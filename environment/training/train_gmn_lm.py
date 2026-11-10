"""Train the bidirectional-attention LM on WikiText-103 (LLM vs LLM arm).

Same data, same parameters, same next-token objective, same schedule, same
evaluation as ``train_gpt2_wiki.py`` (TASK 20R).  The **only** variable is the
attention direction of the trunk.

Rather than copy the TASK 20R trainer, this file *imports* its pipeline
(``ChunkedCorpus``, ``prepare_token_stream``, ``iter_sentences``,
``build_scheduler``, ``save_checkpoint``, the same ``LOG_EVERY``), so
"everything else identical" is guaranteed by construction and not by
copy-paste.  What differs is the model (``GMNStyleLM`` instead of
``GPT2Style``) and the saved paths.

Read this before interpreting the perplexity
--------------------------------------------
``--attention-mode`` selects how the bidirectional trunk is prevented (or not)
from reading the token it is asked to predict:

``peek``  * the literal recipe: ``hidden = trunk(ids); loss = CE(logits[:, :-1],
          ids[:, 1:])``.  Slicing the **output** does not hide the answer from the
          **input** — ``hidden[t]`` attends to position ``t + 1``, which *is* the
          target — so this mode can reach a very low perplexity by copying and
          its number is not comparable to the causal arm.  It exists to
          demonstrate the leak, and the leak is printed at startup.
``no-peek``  a superdiagonal attention mask (query ``t`` may not attend to key
          ``t + 1``).  This blocks the *direct* route only: with 2+ layers the
          answer still reaches position ``t`` through another position (measured:
          the probe is 0.17, not 0.0).
``mask-target``  the leak-free bidirectional objective: a random ``--mask-rate``
          of the targets is replaced by ``[MASK]`` *in the input* before the
          forward pass and only those targets are scored.  The answer is then
          absent from the sequence, so no attention pattern can leak it, and every
          unmasked token — left and right — is available as context.  This is
          BERT-style denoising rather than pure NTP, which is the one objective
          deviation the bidirectional arm requires.

The startup probe (``gmn_style.peekability``) reports, for each mode, whether
perturbing token ``p`` moves the logit at ``p - 1`` that scores it: 0.0 means the
target is invisible, anything larger means it is visible.

Usage (from the repository root)::

    python environment/training/train_gmn_lm.py --smoke
    python environment/training/train_gmn_lm.py --attention-mode mask-target --epochs 2
    python environment/training/train_gmn_lm.py --compare-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

TRAINING_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = TRAINING_DIR.parent
MODELS_DIR = ENVIRONMENT_DIR / "models"
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR, TRAINING_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import MASK_ID, PAD_ID, BPETokenizer  # noqa: E402
from gmn_style import GMNStyleLM, peekability  # noqa: E402

# The whole training pipeline comes from the TASK 20R trainer: identical corpus,
# identical batches, identical schedule, identical logging cadence.
from train_gpt2_wiki import (  # noqa: E402
    EARLY_STOP_PPL,
    EARLY_STOP_PPL_GOOD,
    KEEP_EPOCH_CHECKPOINTS,
    LOG_EVERY,
    ChunkedCorpus,
    build_scheduler,
    configure_stdout,
    prepare_token_stream,
    save_checkpoint,
    stream_path,
)

#: The three attention modes described in the module docstring.
ATTENTION_MODES: Tuple[str, ...] = ("peek", "no-peek", "mask-target")

#: Share of targets replaced by ``[MASK]`` in the input for ``mask-target``.
DEFAULT_MASK_RATE = 0.15

#: Set per ``--attention-mode``; ``peek``/``no-peek`` enable the superdiagonal mask.
DEFAULT_ATTENTION_MODE = "mask-target"


def mode_peek(mode: str) -> bool:
    """``peek`` keeps the plain bidirectional trunk (the leaky recipe)."""
    return mode == "peek"


def build_model(
    model_config: Dict[str, Any],
    mode: str,
    vocab_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
) -> GMNStyleLM:
    """``GMNStyleLM``, with the trunk built for ``mode``.

    ``peek`` -> plain bidirectional; ``no-peek`` -> superdiagonal mask;
    ``mask-target`` -> plain bidirectional trunk (the input masking is what hides
    the answer, so no attention mask is needed).
    """
    return GMNStyleLM(
        config_dict=model_config,
        peek=mode != "no-peek",
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )


# ---------------------------------------------------------------------------
# Next-token loss per attention mode
# ---------------------------------------------------------------------------
def ntp_loss(
    model: GMNStyleLM,
    token_ids: torch.Tensor,
    mode: str,
    mask_rate: float = DEFAULT_MASK_RATE,
    generator: Optional[torch.Generator] = None,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Next-token cross-entropy for one attention mode (see the module docstring).

    ``mask-target`` is the only mode that touches the input: a random
    ``mask_rate`` of the targets becomes ``[MASK]`` there, and the loss is taken
    over those positions.  Masking a position also removes it from every other
    position's context, which is why it cannot leak — at the cost of scoring only
    a fraction of the sequence.
    """
    if mode == "mask-target":
        inputs = token_ids.clone()
        candidate = torch.zeros_like(inputs, dtype=torch.bool)
        candidate[:, 1:] = True
        drawn = torch.rand(inputs.shape, generator=generator)
        chosen = candidate & (drawn < float(mask_rate))
        bare_rows = (~chosen.any(dim=1)).nonzero(as_tuple=False).flatten()
        if int(bare_rows.numel()):
            chosen[bare_rows, 1] = True     # never leave a row with nothing to score
        inputs[chosen] = MASK_ID

        logits = model(inputs)
        weights = chosen[:, 1:].reshape(-1).to(logits.dtype)
        per_position = F.cross_entropy(
            logits[:, :-1].reshape(-1, model.vocab_size),
            token_ids[:, 1:].reshape(-1),
            reduction="none",
        )
        scored = int(chosen[:, 1:].sum())
        if reduction == "mean":
            loss = (per_position * weights).sum() / max(1, scored)
        else:
            loss = per_position
        return loss, {
            "scored_positions": scored,
            "masked_inputs": int(chosen.sum()),
            "mask_rate_actual": scored / max(1, int(candidate[:, 1:].sum())),
        }

    loss = model.loss(token_ids, reduction=reduction)
    return loss, {
        "scored_positions": int((token_ids[:, 1:] != PAD_ID).sum()),
        "masked_inputs": 0,
        "mask_rate_actual": 0.0,
    }


@torch.no_grad()
def evaluate_perplexity(
    model: GMNStyleLM,
    corpus: ChunkedCorpus,
    batch_size: int,
    mode: str,
    mask_rate: float,
    seed: int = 0,
    label: str = "val",
) -> Dict[str, Any]:
    """Mean next-token cross-entropy over the val stream, then ``exp()``.

    For ``mask-target`` the mask is drawn from a generator seeded by the batch
    index, so exactly the same positions are scored at every evaluation and the
    numbers are comparable across epochs.  ``peek`` / ``no-peek`` score every
    position, exactly as the causal arm does.
    """
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    scored_total = 0
    masked_total = 0
    batches = 0
    started = time.perf_counter()

    for start in range(0, corpus.n_chunks, batch_size):
        indices = np.arange(start, min(start + batch_size, corpus.n_chunks), dtype=np.int64)
        ids = corpus.batch(indices)
        generator = (
            torch.Generator().manual_seed(int(seed) + batches)
            if mode == "mask-target"
            else None
        )
        loss, diag = ntp_loss(model, ids, mode, mask_rate, generator=generator)
        loss_sum += float(loss.detach()) * max(1, diag["scored_positions"])
        scored_total += diag["scored_positions"]
        masked_total += diag["masked_inputs"]
        batches += 1

    model.train(was_training)
    mean_loss = loss_sum / max(1, scored_total)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "scored_positions": scored_total,
        "masked_inputs": masked_total,
        "batches": batches,
        "chunks": corpus.n_chunks,
        "seconds": round(time.perf_counter() - started, 1),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Checkpoints and results (the bidirectional arm's own paths)
# ---------------------------------------------------------------------------
def epoch_checkpoint_path(epoch: int) -> Path:
    """``environment/checkpoints/gmn_lm_wiki_epoch{N}.pt``."""
    return config.path("gmn_lm_ckpt").with_name(f"gmn_lm_wiki_epoch{epoch}.pt")


def prune_epoch_checkpoints(keep: int = KEEP_EPOCH_CHECKPOINTS) -> List[str]:
    """Delete all but the newest ``keep`` ``gmn_lm_wiki_epoch*.pt`` files."""
    best = config.path("gmn_lm_ckpt")
    candidates = sorted(
        best.parent.glob("gmn_lm_wiki_epoch*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: List[str] = []
    for stale in candidates[keep:]:
        removed.append(stale.name)
        stale.unlink()
    return removed


def write_result(result: Dict[str, Any]) -> Path:
    """Write the full result record to ``environment/results/gmn_lm_history.json``."""
    path = config.path("gmn_lm_history")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return path


def write_progress_snapshot(history: List[Dict[str, Any]], best: Dict[str, Any],
                            stop_reason: str,
                            mid_epoch_val: Optional[List[Dict[str, Any]]] = None) -> Path:
    """Refresh the history file after each epoch so a crash keeps the curves."""
    path = config.path("gmn_lm_history")
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:  # pragma: no cover - only on a torn write
            existing = {}
    existing.update({
        "script": "train_gmn_lm.py",
        "task": "TASK 21R-LM",
        "history": history,
        "mid_epoch_val": mid_epoch_val or [],
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
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the whole bidirectional-LM training and return the result record."""
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
    print("TASK 21R-LM — bidirectional-attention LM on WikiText-103")
    print("=" * 74)
    print(f"  model config : {args.model}  {model_config}")
    print(f"  attention    : {args.attention_mode}"
          + ("  (superdiagonal mask)" if args.attention_mode == "no-peek" else "")
          + (f"  (targets masked in the input at rate {args.mask_rate})"
             if args.attention_mode == "mask-target" else ""))
    print(f"  tokenizer    : {config.path('tokenizer_wiki')}  "
          f"(vocab {tokenizer.vocab_size:,})")
    print(f"  sequence     : seq_len {seq_len} x batch {args.batch_size} = "
          f"{args.batch_size * seq_len:,} tokens/step")
    print(f"  optimiser    : AdamW lr {args.lr} wd {args.weight_decay} | warmup "
          f"{args.warmup_steps} | cosine decay | clip {args.grad_clip}")
    print(f"  torch        : {torch.__version__}  threads {threads}")
    print("  pipeline     : imported from train_gpt2_wiki.py (identical corpus, "
          "batches, schedule, logging)")

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
          f"{train_corpus.n_chunks:,} chunks ({steps_per_epoch:,} steps/epoch)")
    print(f"  val stream   : {val_stats['tokens']:,} tokens -> "
          f"{val_corpus.n_chunks:,} chunks")
    print(f"  schedule     : {total_steps:,} steps total, {args.epochs} epoch(s)")

    model = build_model(model_config, args.attention_mode)
    print(f"  parameters   : {model.count_parameters():,} "
          f"(tied LM head, same count as GPT2Style)")
    probe_ids = val_corpus.batch(np.arange(2, dtype=np.int64))
    probe = peekability(model, probe_ids)
    print(f"  leak probe   : logits[p-1] move by {probe['max_abs_change']:.4f} when "
          f"token p changes  (0.0000 = target invisible; GPT-2/causal = 0.0)")
    print(flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps)

    history: List[Dict[str, Any]] = []
    mid_epoch_val: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "val_perplexity": float("inf"),
                            "val_loss": float("inf")}
    early_stop: Dict[str, Any] = {}
    stop_reason = f"completed all {args.epochs} requested epoch(s)"
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        order = train_corpus.chunk_order(shuffle_seed=int(args.seed) + epoch)
        epoch_started = time.perf_counter()
        window_started = epoch_started
        epoch_loss_sum = 0.0
        epoch_scored = 0
        epoch_masked = 0
        window_loss_sum = 0.0
        window_steps = 0
        window_tokens = 0
        steps_this_epoch = 0

        for start in range(0, train_corpus.n_chunks, args.batch_size):
            ids = train_corpus.batch(order[start:start + args.batch_size])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, diag = ntp_loss(
                model, ids, args.attention_mode, args.mask_rate
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            steps_this_epoch += 1
            step_loss = float(loss.detach())
            epoch_loss_sum += step_loss * max(1, diag["scored_positions"])
            epoch_scored += diag["scored_positions"]
            epoch_masked += diag["masked_inputs"]
            window_loss_sum += step_loss
            window_steps += 1
            window_tokens += max(1, diag["scored_positions"])

            if global_step % args.log_every == 0:
                mean = window_loss_sum / max(1, window_steps)
                now = time.perf_counter()
                elapsed = now - started
                epoch_elapsed = now - epoch_started
                eta = (epoch_elapsed * (steps_per_epoch - steps_this_epoch)
                       / max(1, steps_this_epoch))
                rate = window_tokens / max(1e-9, now - window_started)
                print(f"  step {global_step:>7,} | loss {mean:7.4f} | ppl "
                      f"{math.exp(min(mean, 20.0)):9.2f} | lr "
                      f"{optimizer.param_groups[0]['lr']:.3e} | {elapsed / 60:7.1f} min | "
                      f"{rate / 1000:5.1f}K tok/s | epoch eta {eta / 3600:5.2f} h",
                      flush=True)
                window_loss_sum = 0.0
                window_steps = 0
                window_tokens = 0
                window_started = now

            if (args.eval_every and global_step % args.eval_every == 0
                    and steps_this_epoch < steps_per_epoch):
                mid = evaluate_perplexity(
                    model, val_corpus, args.eval_batch_size, args.attention_mode,
                    args.mask_rate, seed=int(args.seed),
                )
                mid_epoch_val.append({
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": round(mid["loss"], 4),
                    "val_ppl": round(mid["perplexity"], 3),
                })
                print(f"  step {global_step:>7,} | mid-epoch val_ppl "
                      f"{mid['perplexity']:9.2f}  (val_loss {mid['loss']:.4f}, "
                      f"{mid['seconds']:.0f}s)", flush=True)

            if args.max_steps and global_step >= int(args.max_steps):
                stop_reason = f"stopped at --max-steps {args.max_steps}"
                break

        epoch_loss = epoch_loss_sum / max(1, epoch_scored)
        train_ppl = math.exp(min(epoch_loss, 20.0))
        val = evaluate_perplexity(
            model, val_corpus, args.eval_batch_size, args.attention_mode,
            args.mask_rate, seed=int(args.seed),
        )
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
            "scored_tokens": epoch_scored,
            "masked_inputs": epoch_masked,
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
            "attention_mode": args.attention_mode,
            "mask_rate": args.mask_rate,
            "no_peek": bool(model.trunk.no_peek),
            "leak_probe_max_abs_change": probe["max_abs_change"],
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
            best = {"epoch": epoch, "val_perplexity": val["perplexity"],
                    "val_loss": val["loss"]}
            if not args.no_save:
                save_checkpoint(config.path("gmn_lm_ckpt"), payload)
                marker = "  <- best (also saved to gmn_lm_wiki.pt)"

        print(f"  epoch {epoch:>2d} | train_loss {epoch_loss:8.4f} | train_ppl "
              f"{train_ppl:9.2f} | val_ppl {val['perplexity']:9.2f} | "
              f"{epoch_seconds / 60:6.1f} min{marker}", flush=True)
        print(f"    checkpoint {saved_epoch.name}  "
              f"(pruned: {', '.join(removed) if removed else 'nothing'})", flush=True)
        if not args.no_save:
            write_progress_snapshot(history, best, stop_reason, mid_epoch_val)

        run_truncated = bool(args.max_steps and global_step >= int(args.max_steps))
        if epoch == 1 and not run_truncated:
            passed = val["perplexity"] < EARLY_STOP_PPL
            early_stop = {
                "epoch": 1,
                "val_ppl": round(val["perplexity"], 3),
                "threshold": EARLY_STOP_PPL,
                "passed": passed,
                "clearly_learning": val["perplexity"] < EARLY_STOP_PPL_GOOD,
            }
            if passed:
                print(f"  sanity check (epoch 1): val_ppl {val['perplexity']:.1f} < "
                      f"{EARLY_STOP_PPL:.0f} -> OK", flush=True)
            else:
                stop_reason = (f"STOPPED after epoch 1: val_ppl "
                               f"{val['perplexity']:.1f} > {EARLY_STOP_PPL:.0f} — "
                               "training is not working")
                print(f"  {stop_reason}", flush=True)
                break

        if run_truncated:
            break

    total_seconds = time.perf_counter() - started
    result: Dict[str, Any] = {
        "script": "train_gmn_lm.py",
        "task": "TASK 21R-LM",
        "objective": "next-token prediction (bidirectional attention)",
        "attention_mode": args.attention_mode,
        "mask_rate": args.mask_rate,
        "no_peek": bool(model.trunk.no_peek),
        "leak_probe_max_abs_change": probe["max_abs_change"],
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
            "best": str(config.path("gmn_lm_ckpt")),
            "epochs": sorted(
                path.name
                for path in config.path("gmn_lm_ckpt").parent.glob("gmn_lm_wiki_epoch*.pt")
            ),
        },
        "in_progress": False,
    }
    if not args.no_save:
        write_result(result)
    train_corpus.close()
    val_corpus.close()
    return result


# ---------------------------------------------------------------------------
# CLI, summary and the STEP 3 comparison table
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Dict[str, Any]:
    """A history/result file, or an empty dict when it does not exist yet."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_comparison() -> None:
    """STEP 3: the causal-vs-bidirectional perplexity table from both histories."""
    columns = (
        ("GPT-2 style", read_json(config.path("gpt2_wiki_history"))),
        ("GMN style (bidir)", read_json(config.path("gmn_lm_history"))),
    )
    print("\n" + "=" * 78)
    print("TASK 21R-LM — comparison: causal vs bidirectional attention")
    print("=" * 78)
    print(f"  {'Model':<19s}| {'Params':<9s}| {'Attention':<14s}| "
          f"{'val_ppl epoch1':<15s}| {'val_ppl epoch2':<15s}| mode")
    print(f"  {'-' * 18}+{'-' * 10}+{'-' * 15}+{'-' * 16}+{'-' * 16}+{'-' * 17}")

    results: Dict[str, Dict[str, Any]] = {}
    for label, payload in columns:
        history = payload.get("history", [])
        params = payload.get("params")
        params_text = f"{params / 1e6:.2f}M" if params else "--"
        attention = ("Causal" if label.startswith("GPT-2") else
                     ("Bidirectional" if payload else "--"))
        if payload and payload.get("attention_mode"):
            attention = f"Bidirectional/{payload['attention_mode']}"
        ppls = {row["epoch"]: row.get("val_ppl") for row in history}
        first = f"{ppls[1]:.2f}" if 1 in ppls and ppls[1] is not None else "--"
        second = f"{ppls[2]:.2f}" if 2 in ppls and ppls[2] is not None else "--"
        mode = payload.get("attention_mode", "-") if payload else "-"
        print(f"  {label:<19s}| {params_text:<9s}| {attention:<14s}| "
              f"{first:<15s}| {second:<15s}| {mode}")
        results[label] = {"ppls": ppls, "params": params, "payload": payload}

    gpt2 = results["GPT-2 style"]["ppls"]
    gmn = results["GMN style (bidir)"]["ppls"]
    print()
    for epoch in (1, 2):
        if epoch in gpt2 and epoch in gmn and gpt2[epoch] is not None and gmn[epoch] is not None:
            gap = gpt2[epoch] - gmn[epoch]
            verdict = ("bidirectional wins" if gmn[epoch] < gpt2[epoch]
                       else "causal wins")
            print(f"  epoch {epoch}: ppl gap = GPT-2 {gpt2[epoch]:.2f} - GMN "
                  f"{gmn[epoch]:.2f} = {gap:+.2f}  -> {verdict}")
        else:
            print(f"  epoch {epoch}: incomplete (GPT-2 {gpt2.get(epoch)}, "
                  f"GMN {gmn.get(epoch)})")
    print("\n  McNemar does not apply to perplexity; the gap above is the test.")
    payload = results["GMN style (bidir)"]["payload"]
    if payload:
        probe = payload.get("leak_probe_max_abs_change")
        print(f"  GMN arm leak probe (logits[p-1] move when token p changes): "
              f"{probe:.4f}" if probe is not None else "  GMN arm leak probe: n/a")
        if probe is not None and probe > 1e-6:
            print("  WARNING: probe > 0 — the bidirectional trunk can see the token "
                  "it is predicting,\n           so its perplexity is NOT comparable "
                  "to the causal arm (see the module docstring).")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", choices=sorted(config.MODEL_CONFIGS), default="small",
                        help="model config to train (DECISION 2: small)")
    parser.add_argument("--attention-mode", choices=list(ATTENTION_MODES),
                        default=DEFAULT_ATTENTION_MODE,
                        help="how the bidirectional trunk is stopped from reading the "
                             "target: 'peek' = the literal sliced recipe (leaky, for "
                             "demonstration), 'no-peek' = superdiagonal mask (still "
                             "leaks after 2 layers), 'mask-target' = the target token "
                             "is replaced by [MASK] in the input (leak-free)")
    parser.add_argument("--mask-rate", type=float, default=DEFAULT_MASK_RATE,
                        help=f"share of targets masked in the input for mask-target "
                             f"(default {DEFAULT_MASK_RATE})")
    parser.add_argument("--epochs", type=int, default=2,
                        help="passes over the whole token stream (STEP 2: 2 epochs)")
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
    parser.add_argument("--compare-only", action="store_true",
                        help="just print the GPT-2 vs GMN comparison table (STEP 3)")
    parser.add_argument("--smoke", action="store_true",
                        help="pipeline check: 20k sentences, 20 steps, nothing saved")
    args = parser.parse_args(argv)

    if args.smoke:
        args.limit_sentences = args.limit_sentences or 20_000
        args.max_steps = args.max_steps or 20
        args.epochs = 1
        args.eval_every = 10
        args.no_save = True
    return args


def print_summary(result: Dict[str, Any]) -> None:
    """The DONE block's numbers, straight from the result record."""
    print("\n" + "=" * 74)
    print("SUMMARY — TASK 21R-LM (bidirectional attention, NTP objective)")
    print("=" * 74)
    print(f"  model                : {result['model']} "
          f"{result['model_config']['n_layers']} x "
          f"{result['model_config']['hidden_dim']}, "
          f"ffn {result['model_config']['ffn_dim']}  ({result['params']:,} params)")
    print(f"  attention            : bidirectional, mode={result['attention_mode']} "
          f"(no_peek={result['no_peek']}, mask_rate={result['mask_rate']})")
    print(f"  leak probe           : {result['leak_probe_max_abs_change']:.4f} "
          f"(0.0000 = target invisible; causal = 0.0)")
    print(f"  data                 : {result['train_corpus']['tokens']:,} train tokens "
          f"({result['train_chunks']:,} chunks), {result['val_corpus']['tokens']:,} val "
          f"tokens ({result['val_chunks']:,} chunks)")
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
    print(f"  best val_ppl         : {best['val_perplexity']:.2f} at epoch "
          f"{best['epoch']} -> {result['checkpoints']['best']}")
    print(f"  total training time  : {result['total_minutes']:.1f} min "
          f"({result['seconds_per_step']:.3f} s/step, {result['threads']} threads)")
    print(f"  stop reason          : {result['stop_reason']}")
    print(f"  checkpoints          : "
          f"{', '.join(result['checkpoints']['epochs']) or 'none'}")
    print(f"\n  history file         : {config.path('gmn_lm_history')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    args = parse_args(argv)
    if args.compare_only:
        print_comparison()
        return 0
    result = train(args)
    print_summary(result)
    print_comparison()
    print(f"\nDone in {result['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
