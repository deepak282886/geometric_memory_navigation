"""Train the GPT-2 style baseline on WikiText-103 (TASK 20R).

Objective: plain next-token prediction.  No contrastive term — this is the
autoregressive arm of the controlled comparison, and TASK 21R runs the GMN arm
on the *same* token stream, the *same* schedule and the *same* config dict.

Data pipeline (TASK 20R notes 1-3)
----------------------------------
``train_sentences.json`` is streamed **line by line** — 3,837,479 sentences /
1.5 GB of JSON never fit in the 3.1 GB of free RAM this machine has.  Each
sentence becomes ``[BOS] tokens [EOS]`` and everything is concatenated into ONE
long token stream with no sentence boundaries left (standard GPT-2 pretraining
shape), which is then cut into ``seq_len = 64`` chunks.  Each chunk is one
training example.

The stream is cached to ``train_tokens.bin`` (2 bytes/token, ``uint16``) the
first time and memory-mapped afterwards, so

* RAM stays flat (the OS pages the file; nothing is materialised as a Python
  list),
* chunking cannot drift between epochs or between the two models, and
* chunk order can genuinely be shuffled every epoch (``numpy`` permutation of
  ~2M indices) instead of being read sequentially.

Epoch = one pass over all ~3.8M sentences (TASK 20R note 3).  Mid-epoch
validation perplexity is reported every ``--eval-every`` steps because a full
epoch takes hours on CPU; the formal per-epoch numbers are unchanged.

Usage (from the repository root)::

    python environment/training/train_gpt2_wiki.py --smoke
    python environment/training/train_gpt2_wiki.py --epochs 2
    python environment/training/train_gpt2_wiki.py --model medium --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ``environment/``, ``environment/models/``, ``environment/tokenizer/`` and
# ``environment/training/`` are not package roots, so expose them on sys.path
# (TASK 18/19/20 convention).
TRAINING_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = TRAINING_DIR.parent
MODELS_DIR = ENVIRONMENT_DIR / "models"
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR, TRAINING_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import BOS_ID, EOS_ID, PAD_ID, BPETokenizer  # noqa: E402
from gpt2_style import GPT2Style  # noqa: E402

# The LR schedule is imported from the TASK 21 ARC trainer so the autoregressive
# and contrastive runs share one implementation of "linear warmup then cosine
# decay" — TASK 21R does the same, which is what makes the two runs comparable.
from train_gpt2 import build_scheduler  # noqa: E402

#: Log cadence (TASK 20R note 5 asks for every 500 steps, not 200).
LOG_EVERY = 500

#: TASK 20R note 6: if the first epoch's validation perplexity is above this, the
#: run is broken; below GOOD, it is clearly learning.
EARLY_STOP_PPL = 5000.0
EARLY_STOP_PPL_GOOD = 500.0

#: TASK 20R note 7: keep only this many ``*_epoch{N}.pt`` files on disk.
KEEP_EPOCH_CHECKPOINTS = 2

#: Sentences are tokenized in one pass; progress is printed this often.
PROGRESS_EVERY_SENTENCES = 250_000

#: Tokens buffered before a block is appended to the ``.bin`` cache.
WRITE_FLUSH_TOKENS = 2_000_000


def configure_stdout() -> None:
    """Force UTF-8 output so tokens such as ``Ġ`` survive a cp1252 console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-TextIO
            pass


# ---------------------------------------------------------------------------
# Streaming data pipeline (TASK 20R notes 1-2)
# ---------------------------------------------------------------------------
#: Token-stream cache format: little-endian uint16 (vocab 16000 < 65536).
TOKEN_DTYPE = "<u2"


def iter_sentences(path: Path, limit: Optional[int] = None) -> Iterator[str]:
    """Stream sentence strings out of a one-element-per-line JSON array.

    ``json.JSONDecoder.raw_decode`` parses the leading value on each line and
    ignores the array's own separators, so ``train_sentences.json`` (1.5 GB) is
    never materialised — TASK 18R's RAM constraint.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        count = 0
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if count == 0 and line.startswith("["):
                line = line[1:]
            sentence, _ = decoder.raw_decode(line)
            yield sentence
            count += 1
            if limit is not None and count >= limit:
                break


class TokenStreamWriter:
    """Append token ids to a ``uint16`` ``.bin`` file, buffered by token count."""

    def __init__(self, path: Path, flush_tokens: int = WRITE_FLUSH_TOKENS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.flush_tokens = int(flush_tokens)
        self._handle = self.path.open("wb")
        self._buffer: List[int] = []
        self._tokens = 0

    def extend(self, ids: Sequence[int]) -> None:
        self._buffer.extend(ids)
        if len(self._buffer) >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            np.asarray(self._buffer, dtype=TOKEN_DTYPE).tofile(self._handle)
            self._tokens += len(self._buffer)
            self._buffer = []

    @property
    def tokens(self) -> int:
        """Tokens written plus those still buffered."""
        return self._tokens + len(self._buffer)

    def close(self) -> int:
        self.flush()
        self._handle.close()
        return self._tokens


def build_token_stream(
    split: str,
    tokenizer: BPETokenizer,
    path: Path,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Turn one sentence file into ``[BOS] tokens [EOS]`` followed by the next...

    No sentence boundaries survive: the result is the single long token stream
    TASK 20R note 2 asks for, later cut into ``seq_len`` chunks.
    """
    source = config.path(f"wiki_{split}_sentences")
    writer = TokenStreamWriter(path)
    started = time.perf_counter()
    sentences = 0
    sentence_tokens = 0
    longest = 0

    for sentence in iter_sentences(source, limit):
        ids = tokenizer.encode_unpadded(sentence, max_length=None)
        writer.extend((BOS_ID,))
        writer.extend(ids)
        writer.extend((EOS_ID,))
        sentence_tokens += len(ids)
        longest = max(longest, len(ids))
        sentences += 1
        if verbose and sentences % PROGRESS_EVERY_SENTENCES == 0:
            print(f"    {sentences:>9,} sentences  {writer.tokens:>12,} tokens  "
                  f"{time.perf_counter() - started:>6.0f}s", flush=True)

    tokens = writer.close()
    return {
        "sentences": sentences,
        "tokens": tokens,
        "sentence_tokens": sentence_tokens,
        "longest_sentence_tokens": longest,
        "mean_sentence_tokens": (sentence_tokens / sentences) if sentences else 0.0,
        "seconds": round(time.perf_counter() - started, 1),
    }


def read_token_meta() -> Dict[str, Any]:
    """The provenance record for the ``.bin`` caches (empty dict if absent)."""
    path = config.path("wiki_tokens_meta")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_token_meta(meta: Dict[str, Any]) -> None:
    """Persist the token-cache provenance record."""
    path = config.path("wiki_tokens_meta")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def prepare_token_stream(
    split: str,
    tokenizer: BPETokenizer,
    rebuild: bool = False,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Return the cached token-stream stats, building the cache only if needed.

    The cache is rebuilt when ``--rebuild-tokens`` is passed, when the ``.bin`` is
    missing, or when it was written by a different tokenizer (recorded in the meta
    file).  Tokenizing 3.8M sentences is the expensive part of the pipeline, so it
    happens once here and TASK 21R reuses the very same file.
    """
    bin_path = config.path(f"wiki_{split}_tokens")
    if limit is not None:
        # A smoke run must not clobber the full cache: it writes its own file.
        bin_path = bin_path.with_name(f"{bin_path.stem}.smoke.bin")
    meta = read_token_meta()
    recorded = dict(meta.get(split, {}))
    reusable = (
        recorded.get("vocab_size") == tokenizer.vocab_size
        and recorded.get("tokenizer") == str(config.path("tokenizer_wiki"))
        and bin_path.exists()
        and limit is None
    )
    if reusable and not rebuild:
        recorded["cached"] = True
        print(f"  {split:<5s} token cache : {bin_path.name}  {recorded['tokens']:,} tokens  "
              f"({bin_path.stat().st_size / 1e6:.0f} MB)  [reused]")
        return recorded

    print(f"  {split:<5s} token cache : building {bin_path.name} "
          f"(streaming {config.path(f'wiki_{split}_sentences').name}) ...")
    stats = build_token_stream(split, tokenizer, bin_path, limit=limit, verbose=verbose)
    stats["cached"] = False
    stats["vocab_size"] = tokenizer.vocab_size
    stats["tokenizer"] = str(config.path("tokenizer_wiki"))
    if limit is None:
        meta[split] = stats
        meta["bos_id"] = int(BOS_ID)
        meta["eos_id"] = int(EOS_ID)
        write_token_meta(meta)
    print(f"  {split:<5s} token cache : {stats['tokens']:,} tokens from "
          f"{stats['sentences']:,} sentences  ({stats['seconds']:.0f}s, "
          f"{bin_path.stat().st_size / 1e6:.0f} MB)")
    return stats


class ChunkedCorpus:
    """A memory-mapped token stream viewed as ``(n_chunks, seq_len)`` chunks.

    ``n_tokens % seq_len`` trailing tokens are dropped (reported, never padded),
    so every position in every chunk is a real token.  That is why the only
    ignored target — ``[PAD]``, per TASK 20R — can never actually occur here, and
    the trainer verifies the count is zero.
    """

    def __init__(self, path: Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self._stream = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        self.n_tokens = int(self._stream.size)
        self.n_chunks = self.n_tokens // self.seq_len
        self._chunks = self._stream[: self.n_chunks * self.seq_len].reshape(
            self.n_chunks, self.seq_len
        )

    @property
    def dropped_tokens(self) -> int:
        """Trailing tokens that did not fill a whole chunk."""
        return self.n_tokens - self.n_chunks * self.seq_len

    @property
    def bytes_per_epoch(self) -> int:
        return self.n_chunks * self.seq_len * 2

    def chunk_order(self, shuffle_seed: Optional[int] = None) -> np.ndarray:
        """Chunk indices for one epoch, shuffled when a seed is given."""
        order = np.arange(self.n_chunks, dtype=np.int64)
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(order)
        return order

    def batch(self, indices: np.ndarray) -> torch.Tensor:
        """Gather ``indices`` into a ``(len(indices), seq_len)`` long tensor."""
        return torch.from_numpy(np.asarray(self._chunks[indices], dtype=np.int64))

    def close(self) -> None:
        del self._chunks
        self._stream._mmap.close()


# ---------------------------------------------------------------------------
# Evaluation (TASK 20R note 4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_perplexity(
    model: GPT2Style,
    corpus: ChunkedCorpus,
    batch_size: int,
    label: str = "val",
    progress_every: int = 0,
) -> Dict[str, Any]:
    """Mean next-token cross-entropy over a token stream, then ``exp()``.

    The chunks come from the same stream and the same ``seq_len`` as training, so
    this is the standard LM validation the task asks for.  The mean is weighted by
    scored positions, so a short final batch cannot bias it, and the number of
    ``[PAD]`` targets is reported (and is 0 — the stream has no padding).
    """
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    scored = 0
    ignored = 0
    batches = 0
    started = time.perf_counter()

    for start in range(0, corpus.n_chunks, batch_size):
        indices = np.arange(start, min(start + batch_size, corpus.n_chunks), dtype=np.int64)
        ids = corpus.batch(indices)
        logits = model(ids)
        targets = ids[:, 1:]
        per_position = F.cross_entropy(
            logits[:, :-1].reshape(-1, model.vocab_size),
            targets.reshape(-1),
            reduction="none",
            ignore_index=PAD_ID,
        )
        valid = (targets != PAD_ID).reshape(-1)
        loss_sum += float((per_position * valid).sum())
        scored += int(valid.sum())
        ignored += int((~valid).sum())
        batches += 1
        if progress_every and batches % progress_every == 0:
            print(f"      {label}: {batches}/{math.ceil(corpus.n_chunks / batch_size)} "
                  f"batches, ppl so far {math.exp(min(loss_sum / max(1, scored), 20.0)):.2f}",
                  flush=True)

    model.train(was_training)
    mean_loss = loss_sum / max(1, scored)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "scored_tokens": scored,
        "ignored_pad_targets": ignored,
        "batches": batches,
        "chunks": corpus.n_chunks,
        "seconds": round(time.perf_counter() - started, 1),
    }


# ---------------------------------------------------------------------------
# Checkpoints (TASK 20R note 7: every epoch, but keep only the last two)
# ---------------------------------------------------------------------------
def epoch_checkpoint_path(epoch: int) -> Path:
    """``environment/checkpoints/gpt2_wiki_epoch{N}.pt``."""
    return config.path("gpt2_wiki_ckpt").with_name(f"gpt2_wiki_epoch{epoch}.pt")


def prune_epoch_checkpoints(keep: int = KEEP_EPOCH_CHECKPOINTS) -> List[str]:
    """Delete all but the newest ``keep`` ``gpt2_wiki_epoch*.pt`` files."""
    best = config.path("gpt2_wiki_ckpt")
    candidates = sorted(
        best.parent.glob("gpt2_wiki_epoch*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: List[str] = []
    for stale in candidates[keep:]:
        removed.append(stale.name)
        stale.unlink()
    return removed


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> Path:
    """Write a checkpoint, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def stream_path(split: str, limit: Optional[int]) -> Path:
    """Where the ``.bin`` cache for ``split`` lives (smoke runs use their own)."""
    path = config.path(f"wiki_{split}_tokens")
    return path if limit is None else path.with_name(f"{path.stem}.smoke.bin")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the whole TASK 20R training and return the result record."""
    config.ensure_dirs()
    torch.manual_seed(int(args.seed))
    threads = int(args.threads) if args.threads else (os.cpu_count() or 1)
    torch.set_num_threads(threads)
    try:  # CPU-only nicety: denormals are pure overhead here
        torch.set_flush_denormal(True)
    except RuntimeError:  # pragma: no cover - platform dependent
        pass

    model_config = config.model_config(args.model)
    seq_len = int(model_config["max_seq_len"])
    tokenizer = BPETokenizer.load(config.path("tokenizer_wiki"))

    print("=" * 74)
    print("TASK 20R — GPT-2 style pretraining on WikiText-103")
    print("=" * 74)
    print(f"  model config : {args.model}  {model_config}")
    print(f"  tokenizer    : {config.path('tokenizer_wiki')}  "
          f"(vocab {tokenizer.vocab_size:,})")
    print(f"  sequence     : seq_len {seq_len} x batch {args.batch_size} = "
          f"{args.batch_size * seq_len:,} tokens/step")
    print(f"  optimiser    : AdamW lr {args.lr} wd {args.weight_decay} | warmup "
          f"{args.warmup_steps} | cosine decay | clip {args.grad_clip}")
    print(f"  torch        : {torch.__version__}  threads {threads}")

    train_stats = prepare_token_stream(
        "train", tokenizer, rebuild=args.rebuild_tokens, limit=args.limit_sentences
    )
    val_stats = prepare_token_stream(
        "val", tokenizer, rebuild=args.rebuild_tokens, limit=args.limit_sentences
    )

    train_corpus = ChunkedCorpus(stream_path("train", args.limit_sentences), seq_len)
    val_corpus = ChunkedCorpus(stream_path("val", args.limit_sentences), seq_len)

    steps_per_epoch = math.ceil(train_corpus.n_chunks / args.batch_size)
    total_steps = steps_per_epoch * int(args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, int(args.max_steps))
    print(f"  train stream : {train_stats['tokens']:,} tokens -> "
          f"{train_corpus.n_chunks:,} chunks (dropped {train_corpus.dropped_tokens:,} "
          f"trailing) -> {steps_per_epoch:,} steps/epoch")
    print(f"  val stream   : {val_stats['tokens']:,} tokens -> "
          f"{val_corpus.n_chunks:,} chunks (dropped {val_corpus.dropped_tokens:,} trailing)")
    print(f"  schedule     : {total_steps:,} steps total, {args.epochs} epoch(s)")

    model = GPT2Style(config_dict=model_config)
    print(f"  parameters   : {model.count_parameters():,} (LM head tied to the embedding)")
    print(f"  logging      : every {args.log_every} steps"
          + (f", val_ppl every {args.eval_every} steps" if args.eval_every else "")
          + f", checkpoints in {config.path('gpt2_wiki_ckpt').parent}")
    print(flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps)

    history: List[Dict[str, Any]] = []
    mid_epoch_val: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "val_perplexity": float("inf"), "val_loss": float("inf")}
    early_stop: Dict[str, Any] = {}
    stop_reason = f"completed all {args.epochs} requested epoch(s)"
    global_step = 0
    ignored_total = 0
    started = time.perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        order = train_corpus.chunk_order(shuffle_seed=int(args.seed) + epoch)
        epoch_started = time.perf_counter()
        window_started = epoch_started
        epoch_loss_sum = 0.0
        epoch_positions = 0
        window_loss_sum = 0.0
        window_steps = 0
        window_tokens = 0
        steps_this_epoch = 0

        for start in range(0, train_corpus.n_chunks, args.batch_size):
            ids = train_corpus.batch(order[start:start + args.batch_size])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(ids)                     # CE over next token, [PAD] ignored
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            steps_this_epoch += 1
            positions = int(ids[:, 1:].numel())
            ignored_total += int((ids[:, 1:] == PAD_ID).sum())
            step_loss = float(loss.detach())
            epoch_loss_sum += step_loss * positions
            epoch_positions += positions
            window_loss_sum += step_loss
            window_steps += 1
            window_tokens += positions

            if global_step % args.log_every == 0:
                mean = window_loss_sum / max(1, window_steps)
                now = time.perf_counter()
                elapsed = now - started
                epoch_elapsed = now - epoch_started
                eta = epoch_elapsed * (steps_per_epoch - steps_this_epoch) / max(1, steps_this_epoch)
                rate = window_tokens / max(1e-9, now - window_started)
                print(f"  step {global_step:>7,} | loss {mean:7.4f} | ppl "
                      f"{math.exp(min(mean, 20.0)):9.2f} | lr "
                      f"{optimizer.param_groups[0]['lr']:.3e} | {elapsed / 60:7.1f} min | "
                      f"{rate / 1000:5.1f}K tok/s | epoch eta {eta / 3600:5.2f} h", flush=True)
                window_loss_sum = 0.0
                window_steps = 0
                window_tokens = 0
                window_started = now

            if (args.eval_every and global_step % args.eval_every == 0
                    and steps_this_epoch < steps_per_epoch):
                mid = evaluate_perplexity(model, val_corpus, args.eval_batch_size)
                mid_epoch_val.append({
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": round(mid["loss"], 4),
                    "val_ppl": round(mid["perplexity"], 3),
                })
                print(f"  step {global_step:>7,} | mid-epoch val_ppl {mid['perplexity']:9.2f}"
                      f"  (val_loss {mid['loss']:.4f}, {mid['seconds']:.0f}s)", flush=True)

            if args.max_steps and global_step >= int(args.max_steps):
                stop_reason = f"stopped at --max-steps {args.max_steps}"
                break

        epoch_loss = epoch_loss_sum / max(1, epoch_positions)
        train_ppl = math.exp(min(epoch_loss, 20.0))
        val = evaluate_perplexity(model, val_corpus, args.eval_batch_size)
        epoch_seconds = time.perf_counter() - epoch_started

        history.append({
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(epoch_loss, 4),
            "train_ppl": round(train_ppl, 3),
            "val_loss": round(val["loss"], 4),
            "val_ppl": round(val["perplexity"], 3),
            "lr": optimizer.param_groups[0]["lr"],
            "steps": steps_this_epoch,
            "scored_tokens": epoch_positions,
            "epoch_seconds": round(epoch_seconds, 1),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "val_seconds": val["seconds"],
        })

        payload = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss,
            "train_ppl": train_ppl,
            "val_loss": val["loss"],
            "val_perplexity": val["perplexity"],
            "model_config": model_config,
            "train_config": dict(config.TRAIN),
            "tokenizer": str(config.path("tokenizer_wiki")),
            "seq_len": seq_len,
            "args": vars(args),
        }
        saved_epoch = Path("(nothing: --no-save)")
        removed: List[str] = []
        if not args.no_save:
            saved_epoch = save_checkpoint(epoch_checkpoint_path(epoch), payload)
            removed = prune_epoch_checkpoints()

        marker = ""
        if val["perplexity"] < best["val_perplexity"]:
            best = {"epoch": epoch, "val_perplexity": val["perplexity"], "val_loss": val["loss"]}
            if not args.no_save:
                save_checkpoint(config.path("gpt2_wiki_ckpt"), payload)
                marker = "  <- best (also saved to gpt2_wiki.pt)"

        print(f"  epoch {epoch:>2d} | train_loss {epoch_loss:8.4f} | train_ppl "
              f"{train_ppl:9.2f} | val_ppl {val['perplexity']:9.2f} | "
              f"{epoch_seconds / 60:6.1f} min{marker}", flush=True)
        print(f"    checkpoint {saved_epoch.name}  "
              f"(pruned: {', '.join(removed) if removed else 'nothing'})", flush=True)
        if not args.no_save:
            write_progress_snapshot(history, mid_epoch_val, best, stop_reason)

        if epoch == 1 and (not args.max_steps or global_step < int(args.max_steps)):
            passed = val["perplexity"] < EARLY_STOP_PPL
            early_stop = {
                "epoch": 1,
                "val_ppl": round(val["perplexity"], 3),
                "threshold": EARLY_STOP_PPL,
                "passed": passed,
                "good_threshold": EARLY_STOP_PPL_GOOD,
                "clearly_learning": val["perplexity"] < EARLY_STOP_PPL_GOOD,
            }
            verdict = ("clearly learning" if early_stop["clearly_learning"]
                       else f"working, but not yet below {EARLY_STOP_PPL_GOOD:.0f}")
            if passed:
                print(f"  early-stop check (TASK 20R note 6): val_ppl "
                      f"{val['perplexity']:.1f} < {EARLY_STOP_PPL:.0f} -> OK ({verdict})",
                      flush=True)
            else:
                stop_reason = (f"STOPPED after epoch 1: val_ppl {val['perplexity']:.1f} > "
                               f"{EARLY_STOP_PPL:.0f} — training is not working")
                print(f"  {stop_reason}", flush=True)
                break

        if args.max_steps and global_step >= int(args.max_steps):
            break

    total_seconds = time.perf_counter() - started
    result: Dict[str, Any] = {
        "script": "train_gpt2_wiki.py",
        "task": "TASK 20R",
        "objective": "next-token prediction (autoregressive)",
        "model": args.model,
        "model_config": model_config,
        "params": model.count_parameters(),
        "seq_len": seq_len,
        "batch_size": int(args.batch_size),
        "tokens_per_step": int(args.batch_size) * seq_len,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": int(args.warmup_steps),
        "grad_clip": args.grad_clip,
        "seed": int(args.seed),
        "threads": threads,
        "torch": torch.__version__,
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_scheduled": total_steps,
        "global_step": global_step,
        "train_corpus": train_stats,
        "val_corpus": val_stats,
        "train_chunks": train_corpus.n_chunks,
        "val_chunks": val_corpus.n_chunks,
        "dropped_tokens": {
            "train": train_corpus.dropped_tokens,
            "val": val_corpus.dropped_tokens,
        },
        "ignored_pad_targets": ignored_total,
        "history": history,
        "mid_epoch_val": mid_epoch_val,
        "best": {
            "epoch": best["epoch"],
            "val_perplexity": best["val_perplexity"],
            "val_loss": best["val_loss"],
        },
        "early_stop_check": early_stop,
        "stopped_early": stop_reason != f"completed all {args.epochs} requested epoch(s)",
        "stop_reason": stop_reason,
        "total_seconds": round(total_seconds, 1),
        "total_minutes": round(total_seconds / 60, 1),
        "seconds_per_step": round(total_seconds / max(1, global_step), 4),
        "checkpoints": {
            "best": str(config.path("gpt2_wiki_ckpt")),
            "epochs": sorted(
                path.name
                for path in config.path("gpt2_wiki_ckpt").parent.glob("gpt2_wiki_epoch*.pt")
            ),
        },
        "in_progress": False,
    }
    if not args.no_save:
        write_result(result)
    train_corpus.close()
    val_corpus.close()
    return result


def write_result(result: Dict[str, Any]) -> Path:
    """Write the full result record to ``environment/results/gpt2_wiki_history.json``."""
    path = config.path("gpt2_wiki_history")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return path


def write_progress_snapshot(
    history: List[Dict[str, Any]],
    mid_epoch_val: List[Dict[str, Any]],
    best: Dict[str, Any],
    stop_reason: str,
) -> Path:
    """Refresh the history file after each epoch so a crash keeps the curves."""
    path = config.path("gpt2_wiki_history")
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:  # pragma: no cover - only on a torn write
            existing = {}
    existing.update({
        "script": "train_gpt2_wiki.py",
        "task": "TASK 20R",
        "history": history,
        "mid_epoch_val": mid_epoch_val,
        "best": dict(best),
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "in_progress": True,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--model", choices=sorted(config.MODEL_CONFIGS), default="small",
        help="model config to train (DECISION 2: small)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="passes over the whole token stream (TASK 20R: 2 first, then decide)",
    )
    parser.add_argument("--batch-size", type=int, default=int(config.TRAIN["batch_size"]))
    parser.add_argument("--lr", type=float, default=float(config.TRAIN["lr"]))
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=int(config.TRAIN["warmup_steps"]))
    parser.add_argument("--grad-clip", type=float, default=float(config.TRAIN["grad_clip"]))
    parser.add_argument("--seed", type=int, default=int(config.TRAIN["seed"]))
    parser.add_argument("--log-every", type=int, default=LOG_EVERY,
                        help=f"log cadence in steps (default {LOG_EVERY})")
    parser.add_argument("--eval-every", type=int, default=2000,
                        help="mid-epoch val_ppl cadence in steps (0 = per epoch only)")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps")
    parser.add_argument("--limit-sentences", type=int, default=None,
                        help="smoke mode: tokenize only the first N sentences (own .bin)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU thread count")
    parser.add_argument("--rebuild-tokens", action="store_true",
                        help="re-tokenize instead of reusing the .bin cache")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write checkpoints or the history file")
    parser.add_argument("--smoke", action="store_true",
                        help="~1 minute pipeline check: 20k sentences, 30 steps, nothing saved")
    args = parser.parse_args(argv)

    if args.smoke:
        args.limit_sentences = args.limit_sentences or 20_000
        args.max_steps = args.max_steps or 30
        args.epochs = 1
        args.eval_every = 10
        args.no_save = True
    return args


def print_summary(result: Dict[str, Any]) -> None:
    """The DONE block's numbers, straight from the result record."""
    print("\n" + "=" * 74)
    print("SUMMARY — TASK 20R (GPT-2 style, next-token prediction)")
    print("=" * 74)
    print(f"  model                : {result['model']} "
          f"{result['model_config']['n_layers']} x {result['model_config']['hidden_dim']}, "
          f"ffn {result['model_config']['ffn_dim']}  ({result['params']:,} params)")
    print(f"  data                 : {result['train_corpus']['tokens']:,} train tokens "
          f"({result['train_chunks']:,} chunks), {result['val_corpus']['tokens']:,} val tokens "
          f"({result['val_chunks']:,} chunks)")
    print(f"  schedule             : {result['steps_per_epoch']:,} steps/epoch x "
          f"{result['epochs_requested']} epoch(s), lr {result['lr']} -> cosine decay")
    print(f"  {'epoch':>5s} {'train_loss':>11s} {'train_ppl':>10s} {'val_ppl':>9s} "
          f"{'minutes':>8s}")
    for row in result["history"]:
        print(f"  {row['epoch']:>5d} {row['train_loss']:>11.4f} {row['train_ppl']:>10.2f} "
              f"{row['val_ppl']:>9.2f} {row['epoch_seconds'] / 60:>8.1f}")
    if result["mid_epoch_val"]:
        first = result["mid_epoch_val"][0]
        last = result["mid_epoch_val"][-1]
        print(f"  mid-epoch val_ppl    : first {first['val_ppl']:.2f} "
              f"(step {first['global_step']:,}) -> last {last['val_ppl']:.2f} "
              f"(step {last['global_step']:,})")
    best = result["best"]
    print(f"  best val_ppl         : {best['val_perplexity']:.2f} at epoch {best['epoch']} "
          f"-> {result['checkpoints']['best']}")
    print(f"  val_ppl < 100        : {best['val_perplexity'] < 100.0}")
    print(f"  val_ppl < 500        : {best['val_perplexity'] < 500.0}")
    print(f"  total training time  : {result['total_minutes']:.1f} min "
          f"({result['seconds_per_step']:.3f} s/step, {result['threads']} threads)")
    print(f"  stop reason          : {result['stop_reason']}")
    print(f"  checkpoints          : {', '.join(result['checkpoints']['epochs']) or 'none'}")
    print(f"\n  history file         : {config.path('gpt2_wiki_history')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    args = parse_args(argv)
    result = train(args)
    print_summary(result)
    print(f"\nDone in {result['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
