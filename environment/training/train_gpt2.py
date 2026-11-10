"""Train the GPT-2 style baseline from scratch on ARC + ConceptNet (TASK 21).

Objective: plain next-token prediction.  No contrastive term, no navigation —
that is the whole point of the baseline.

Data (built by :func:`build_training_corpus`)::

    ARC-Challenge train   1,117 x 4  ->  [BOS] question [EOS] choice_k [EOS]
    ConceptNet            99,187    ->  [BOS] subject relation object [EOS]
                                     ------------------------------
                                     103,655 sequences  (~1.08M real tokens)

The four sequences per ARC question deliberately include the *wrong* choices:
the model is asked to model all of them as text, and only the evaluation step
compares their likelihoods.

Sequence handling
-----------------
``encode_unpadded`` is used throughout, and batches are padded to the *batch*
maximum rather than to the global ``max_seq_len`` (TASK 21 note 1).  Batches are
length-bucketed inside a shuffled window so most batches are pure ConceptNet
(~9 tokens) or pure ARC (~36 tokens) instead of every batch paying for the
longest sequence in a random draw.  This changes only padding, never content,
and TASK 22 reuses the same batcher.

Evaluation
----------
For each val question and choice, the full ``[BOS] q [EOS] c [EOS]`` sequence is
scored and the log-probabilities are summed **only over the choice positions**
(everything after the separator ``[EOS]``).  Question positions never contribute,
which is what makes the number comparable with the GMN model's cosine scores.

Usage (from the repository root)::

    python environment/training/train_gpt2.py                      # full 20 epochs
    python environment/training/train_gpt2.py --smoke              # ~1 min sanity run
    python environment/training/train_gpt2.py --epochs 3 --limit-train 5000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# ``environment/``, ``environment/models/`` and ``environment/tokenizer/`` are
# not package roots, so expose them on sys.path (TASK 18/19/20 convention).
TRAINING_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = TRAINING_DIR.parent
MODELS_DIR = ENVIRONMENT_DIR / "models"
TOKENIZER_DIR = ENVIRONMENT_DIR / "tokenizer"
for _directory in (ENVIRONMENT_DIR, MODELS_DIR, TOKENIZER_DIR, TRAINING_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import config  # noqa: E402
from bpe import BOS_ID, BPETokenizer, EOS_ID, PAD_ID  # noqa: E402
from gpt2_style import GPT2Style  # noqa: E402

ARC_TRAIN_SPLIT = "train"
ARC_VAL_SPLIT = "val"

#: Targets never scored: ``[PAD]`` (padding) and ``[BOS]``.  ``[BOS]`` only ever
#: sits at position 0 of a sequence, so after the ``ids[:, 1:]`` shift it is
#: already excluded — the clause is kept so the mask stays correct if sequence
#: packing is ever added, and the script reports how many positions it removed.
IGNORED_TARGET_IDS = (PAD_ID, BOS_ID)

#: Sequences are sorted by length inside windows of this many sequences before
#: batching.  Padding drops sharply; within-window shuffling keeps the order
#: random.  3200 = 100 batches of 32.
BUCKET_WINDOW = 3200

HISTORY_PATH = ENVIRONMENT_DIR / "results" / "gpt2_training_history.json"

# ---------------------------------------------------------------------------
# Data utilities — SHARED with train_gmn.py (TASK 22)
# ---------------------------------------------------------------------------
# These live here rather than in a new module so the file tree stays exactly as
# TASKS_V3.md specifies; TASK 22 imports them so both models are trained on
# byte-identical tokenized data.
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_arc_records(split: str) -> List[Dict[str, Any]]:
    """One of the TASK 18 ARC splits (``train`` / ``val`` / ``test``)."""
    return load_json(config.path(f"arc_{split}"))


def load_conceptnet_triples() -> List[Dict[str, str]]:
    """The TASK 18 ConceptNet triples."""
    return load_json(config.path("conceptnet"))


def conceptnet_sentence(record: Dict[str, str]) -> str:
    """``{"subject": "fire", ...}`` -> ``"fire causes heat"``."""
    return f"{record['subject']} {record['relation']} {record['object']}"


def build_arc_sequence(
    question_ids: Sequence[int],
    choice_ids: Sequence[int],
    max_len: int,
) -> Tuple[List[int], int]:
    """``[BOS] question [EOS] choice [EOS]``.

    Returns ``(ids, choice_start)`` where ``choice_start`` is the index of the
    first choice token, so the evaluator knows which positions to score.

    If the pair is too long the **question** is trimmed first, so the choice and
    its terminating ``[EOS]`` always survive — the choice is what evaluation
    measures.
    """
    suffix = list(choice_ids) + [EOS_ID]
    if len(suffix) > max_len - 2:
        suffix = suffix[: max_len - 2] + [EOS_ID]
    budget = max_len - len(suffix)
    question_keep = list(question_ids)[: max(0, budget - 2)]
    prefix = [BOS_ID, *question_keep, EOS_ID]
    return prefix + suffix, len(prefix)


def build_text_sequence(text_ids: Sequence[int], max_len: int) -> List[int]:
    """``[BOS] tokens [EOS]``, trimmed to ``max_len``."""
    keep = list(text_ids)[: max(0, max_len - 2)]
    return [BOS_ID, *keep, EOS_ID]


def pad_batch(sequences: Sequence[Sequence[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    """Right-pad a list of variable-length id sequences into one ``(B, Tmax)`` tensor."""
    width = max(len(sequence) for sequence in sequences)
    batch = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        batch[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch


class SequenceBatcher:
    """Length-bucketed batches with per-batch dynamic padding.

    Each epoch reshuffles with ``seed + epoch``, sorts by length inside windows
    of ``window`` sequences, then shuffles the resulting batch order.  Content is
    untouched: only how sequences are grouped into batches changes, which cuts
    padded tokens from ~27 to ~11 per sequence on this corpus.
    """

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        batch_size: int,
        seed: int,
        window: int = BUCKET_WINDOW,
    ) -> None:
        self.sequences = sequences
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.window = max(int(window), int(batch_size))

    def __len__(self) -> int:
        return math.ceil(len(self.sequences) / self.batch_size)

    def batches(self, epoch: int = 0) -> Iterator[List[int]]:
        """Yield lists of indices, one per batch, in a seeded random order."""
        order = list(range(len(self.sequences)))
        rng = random.Random(self.seed + epoch)
        rng.shuffle(order)

        batch_indices: List[List[int]] = []
        for start in range(0, len(order), self.window):
            window = order[start: start + self.window]
            window.sort(key=lambda index: len(self.sequences[index]))
            for offset in range(0, len(window), self.batch_size):
                batch_indices.append(window[offset: offset + self.batch_size])
        rng.shuffle(batch_indices)
        yield from batch_indices


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------
def build_training_corpus(
    tokenizer: BPETokenizer,
    max_len: int,
    limit_train: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Tokenize ARC-Challenge train + ConceptNet into variable-length id sequences."""
    started = time.perf_counter()
    arc_records = load_arc_records(ARC_TRAIN_SPLIT)
    sequences: List[List[int]] = []
    arc_truncated = 0

    for record in arc_records:
        question_ids = tokenizer.encode_unpadded(record["question"])
        for choice in record["choices"]:
            choice_ids = tokenizer.encode_unpadded(choice)
            if len(question_ids) + len(choice_ids) + 3 > max_len:
                arc_truncated += 1
            ids, _ = build_arc_sequence(question_ids, choice_ids, max_len)
            sequences.append(ids)
    arc_sequences = len(sequences)

    triples = load_conceptnet_triples()
    conceptnet_truncated = 0
    for record in triples:
        tokens = tokenizer.encode_unpadded(conceptnet_sentence(record))
        if len(tokens) + 2 > max_len:
            conceptnet_truncated += 1
        sequences.append(build_text_sequence(tokens, max_len))
    conceptnet_sequences = len(triples)

    stats: Dict[str, Any] = {
        "arc_records": len(arc_records),
        "arc_sequences": arc_sequences,
        "arc_truncated": arc_truncated,
        "conceptnet_sequences": conceptnet_sequences,
        "conceptnet_truncated": conceptnet_truncated,
        "tokenize_seconds": round(time.perf_counter() - started, 1),
    }

    if limit_train is not None and limit_train < len(sequences):
        rng = random.Random(seed)
        keep = sorted(rng.sample(range(len(sequences)), int(limit_train)))
        sequences = [sequences[index] for index in keep]
        stats["subsampled_to"] = len(sequences)

    lengths = [len(sequence) for sequence in sequences]
    stats.update(
        {
            "total_sequences": len(sequences),
            "total_real_tokens": int(sum(lengths)),
            "mean_length": round(sum(lengths) / len(lengths), 2),
            "max_length": int(max(lengths)),
        }
    )
    return sequences, stats


def print_corpus_stats(stats: Dict[str, Any]) -> None:
    print("Training corpus")
    print("-" * 74)
    print(f"  ARC train records        : {stats['arc_records']:>7,d}")
    print(f"  ARC sequences (x4)       : {stats['arc_sequences']:>7,d}"
          f"   truncated: {stats['arc_truncated']}")
    print(f"  ConceptNet sequences     : {stats['conceptnet_sequences']:>7,d}"
          f"   truncated: {stats['conceptnet_truncated']}")
    print(f"  TOTAL sequences          : {stats['total_sequences']:>7,d}")
    print(f"  real tokens / epoch      : {stats['total_real_tokens']:>7,d}")
    print(f"  mean / max length        : {stats['mean_length']:>7.2f} / {stats['max_length']}")
    if "subsampled_to" in stats:
        print(f"  SUBSAMPLED TO            : {stats['subsampled_to']:>7,d}  (smoke run)")
    print(f"  tokenisation time        : {stats['tokenize_seconds']}s")


def padding_cost(
    sequences: Sequence[Sequence[int]], batcher: SequenceBatcher
) -> Tuple[int, int]:
    """``(padded_tokens, real_tokens)`` for one pass under this batcher."""
    padded = 0
    real = 0
    for batch in batcher.batches(0):
        width = max(len(sequences[index]) for index in batch)
        padded += width * len(batch)
        real += sum(len(sequences[index]) for index in batch)
    return padded, real


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------
def masked_next_token_loss(
    model: GPT2Style,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Mean CE over positions ``1..T-1``, ignoring ``[PAD]`` and ``[BOS]`` targets.

    Input is ``ids[:, :-1]``, target is ``ids[:, 1:]`` (TASK 21 note 2).  The mean
    is taken over scored positions only — masked positions contribute exactly
    nothing to numerator or denominator.
    """
    logits = model(token_ids)
    shift_logits = logits[:, :-1, :]
    targets = token_ids[:, 1:]

    per_position = F.cross_entropy(
        shift_logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    is_pad = targets == PAD_ID
    is_bos = targets == BOS_ID
    valid = ~(is_pad | is_bos)

    loss = (per_position * valid).sum() / valid.sum().clamp(min=1)
    return loss, {
        "scored": int(valid.sum()),
        "masked_pad": int(is_pad.sum()),
        "masked_bos": int(is_bos.sum()),
        "total_targets": int(targets.numel()),
    }


@torch.no_grad()
def score_choices(
    model: GPT2Style,
    val_records: Sequence[Dict[str, Any]],
    tokenizer: BPETokenizer,
    max_len: int,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Sum of log-probabilities over each choice's own positions (TASK 21 note 3).

    Every val question contributes four ``[BOS] q [EOS] c [EOS]`` sequences, and
    only positions **after the separator ``[EOS]``** are summed, so the question
    tokens never enter the comparison.

    Returns ``(scores (N, 4), accuracy, diagnostics)``.
    """
    items: List[Tuple[int, int, List[int], int]] = []
    for record_index, record in enumerate(val_records):
        question_ids = tokenizer.encode_unpadded(record["question"])
        for choice_index, choice in enumerate(record["choices"]):
            choice_ids = tokenizer.encode_unpadded(choice)
            ids, choice_start = build_arc_sequence(question_ids, choice_ids, max_len)
            items.append((record_index, choice_index, ids, choice_start))

    n_choices = len(val_records[0]["choices"])
    score_sums = torch.full((len(val_records), n_choices), float("-inf"))
    score_counts = torch.ones(len(val_records), n_choices)
    scored_positions = 0

    was_training = model.training
    model.eval()
    order = sorted(range(len(items)), key=lambda index: len(items[index][2]))

    for start in range(0, len(order), batch_size):
        chunk = order[start: start + batch_size]
        sequences = [items[index][2] for index in chunk]
        choice_start = torch.tensor([items[index][3] for index in chunk])
        lengths = torch.tensor([len(sequence) for sequence in sequences])

        batch = pad_batch(sequences)
        log_probs = F.log_softmax(model(batch)[:, :-1, :], dim=-1)
        targets = batch[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        positions = torch.arange(gathered.shape[1]).unsqueeze(0)
        scored = (
            (positions >= (choice_start - 1).unsqueeze(1))
            & (positions < (lengths - 1).unsqueeze(1))
            & (targets != PAD_ID)
        )
        scored_positions += int(scored.sum())

        totals = (gathered * scored).sum(-1)
        counts = scored.sum(-1).clamp(min=1)
        for row, item_index in enumerate(chunk):
            record_index, choice_index = items[item_index][0], items[item_index][1]
            score_sums[record_index, choice_index] = totals[row]
            score_counts[record_index, choice_index] = counts[row]

    model.train(was_training)

    labels = torch.tensor([int(record["label"]) for record in val_records])
    predictions = score_sums.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean())

    # Diagnostic only: the TASK 21 metric is a SUM of log-probs, which mechanically
    # favours short choices.  The length-normalised mean is tracked alongside so a
    # length artifact can never be read as an architectural effect.
    means = score_sums / score_counts.clamp(min=1)
    normalized_predictions = means.argmax(dim=-1)
    accuracy_normalized = float((normalized_predictions == labels).float().mean())

    lengths_flat = score_counts.reshape(-1).float()
    scores_flat = score_sums.reshape(-1)
    length_score_correlation = float(
        ((lengths_flat - lengths_flat.mean()) * (scores_flat - scores_flat.mean())).mean()
        / (lengths_flat.std(unbiased=False) * scores_flat.std(unbiased=False)).clamp(min=1e-8)
    )

    diagnostics = {
        "val_records": len(val_records),
        "scored_sequences": len(items),
        "scored_tokens": scored_positions,
        "mean_scored_tokens_per_choice": round(scored_positions / len(items), 2),
        "accuracy_sum_logprob": round(accuracy, 6),
        "accuracy_mean_logprob_normalized": round(accuracy_normalized, 6),
        "length_score_correlation": round(length_score_correlation, 4),
        "mean_choice_logprob": round(float(score_sums.mean()), 3),
        "label_distribution": [int((labels == c).sum()) for c in range(n_choices)],
        "prediction_distribution": [int((predictions == c).sum()) for c in range(n_choices)],
        "normalized_prediction_distribution": [
            int((normalized_predictions == c).sum()) for c in range(n_choices)
        ],
    }
    return score_sums, accuracy, diagnostics


# ---------------------------------------------------------------------------
# Optimiser schedule
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_steps``, then cosine decay to 0 (TASK 21 note 6)."""
    warmup = max(1, int(warmup_steps))
    decay = max(1, int(total_steps) - warmup)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(decay)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def random_batch_padding_cost(
    sequences: Sequence[Sequence[int]], batch_size: int, seed: int
) -> Tuple[int, int]:
    """Padding cost if batches were drawn at random instead of length-bucketed."""
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    padded = real = 0
    for start in range(0, len(order), batch_size):
        batch = order[start: start + batch_size]
        padded += max(len(sequences[index]) for index in batch) * len(batch)
        real += sum(len(sequences[index]) for index in batch)
    return padded, real


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Full TASK 21 training run; returns the history dict."""
    seed = int(args.seed)
    torch.manual_seed(seed)
    random.seed(seed)

    max_len = int(args.max_len)
    tokenizer = BPETokenizer.load()

    sequences, corpus_stats = build_training_corpus(
        tokenizer, max_len, limit_train=args.limit_train, seed=seed
    )
    print_corpus_stats(corpus_stats)

    batcher = SequenceBatcher(sequences, args.batch_size, seed)
    steps_per_epoch = len(batcher)
    total_steps = int(args.epochs) * steps_per_epoch

    bucketed_padded, real_tokens = padding_cost(sequences, batcher)
    random_padded, _ = random_batch_padding_cost(sequences, args.batch_size, seed)
    print("\nBatching (dynamic padding to the batch maximum, per TASK 21 note 1)")
    print("-" * 74)
    print(f"  batches / epoch           : {steps_per_epoch:>7,d}")
    print(f"  real tokens / epoch       : {real_tokens:>7,d}")
    print(f"  padded tokens, bucketed   : {bucketed_padded:>7,d}"
          f"  ({100.0 * bucketed_padded / real_tokens - 100.0:5.1f}% overhead)")
    print(f"  padded tokens, unshuffled : {random_padded:>7,d}"
          f"  ({100.0 * random_padded / real_tokens - 100.0:5.1f}% overhead)")
    print(f"  total optimiser steps     : {total_steps:>7,d}")

    val_records = load_arc_records(ARC_VAL_SPLIT)
    print(f"  val questions             : {len(val_records):>7,d}")

    model = GPT2Style()
    print(f"\nModel: {model.count_parameters():,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_steps)

    print("\nTraining")
    print("-" * 74)
    print(f"  {'step':>7s}  {'loss':>8s}  {'ppl':>10s}  {'lr':>10s}")

    started = time.perf_counter()
    history: List[Dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_diagnostics: Dict[str, Any] = {}
    global_step = 0
    final_epoch_loss = float("nan")
    stopped_early = False

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_scored = 0
        interval_loss_sum = 0.0
        interval_scored = 0
        masked_pad_total = 0
        masked_bos_total = 0

        for batch_indices in batcher.batches(epoch):
            batch = pad_batch([sequences[index] for index in batch_indices])

            optimizer.zero_grad(set_to_none=True)
            loss, diag = masked_next_token_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.detach())
            epoch_loss_sum += loss_value * diag["scored"]
            epoch_scored += diag["scored"]
            interval_loss_sum += loss_value * diag["scored"]
            interval_scored += diag["scored"]
            masked_pad_total += diag["masked_pad"]
            masked_bos_total += diag["masked_bos"]

            if interval_scored and global_step % int(args.log_every) == 0:
                interval_loss = interval_loss_sum / interval_scored
                print(
                    f"  {global_step:>7d}  {interval_loss:>8.4f}  "
                    f"{math.exp(min(interval_loss, 20.0)):>10.2f}  "
                    f"{optimizer.param_groups[0]['lr']:>10.6f}",
                    flush=True,
                )
                interval_loss_sum = 0.0
                interval_scored = 0

            if args.max_steps and global_step >= int(args.max_steps):
                stopped_early = True
                break

        epoch_loss = epoch_loss_sum / max(1, epoch_scored)
        epoch_ppl = math.exp(min(epoch_loss, 20.0))
        final_epoch_loss = epoch_loss

        _, accuracy, eval_diag = score_choices(
            model, val_records, tokenizer, max_len, batch_size=int(args.eval_batch_size)
        )

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": round(epoch_loss, 4),
                "train_ppl": round(epoch_ppl, 3),
                "val_accuracy": round(accuracy, 6),
                "lr": optimizer.param_groups[0]["lr"],
                "scored_tokens": epoch_scored,
                "masked_pad_targets": masked_pad_total,
                "masked_bos_targets": masked_bos_total,
                "elapsed_s": round(time.perf_counter() - started, 1),
                "eval": eval_diag,
            }
        )

        marker = ""
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_diagnostics = eval_diag
            marker = "  <- best (saved)"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_accuracy": accuracy,
                    "train_loss": epoch_loss,
                    "args": vars(args),
                    "model_config": dict(config.MODEL),
                    "train_config": dict(config.TRAIN),
                },
                config.path("gpt2_ckpt"),
            )

        print(
            f"  epoch {epoch:>2d} | train_loss {epoch_loss:8.4f} | "
            f"train_ppl {epoch_ppl:9.2f} | val_acc {100.0 * accuracy:5.2f}% | "
            f"norm {100.0 * eval_diag['accuracy_mean_logprob_normalized']:5.2f}%{marker}",
            flush=True,
        )

        if stopped_early:
            print(f"  (stopped early at --max-steps {args.max_steps})")
            break

    total_time = time.perf_counter() - started

    return {
        "history": history,
        "corpus": corpus_stats,
        "batcher": {
            "batch_size": int(args.batch_size),
            "steps_per_epoch": steps_per_epoch,
            "real_tokens_per_epoch": real_tokens,
            "bucketed_padded_tokens": bucketed_padded,
            "random_padded_tokens": random_padded,
        },
        "best": {
            "val_accuracy": best_accuracy,
            "epoch": best_epoch,
            "diagnostics": best_diagnostics,
        },
        "final_train_loss": final_epoch_loss,
        "final_train_ppl": math.exp(min(final_epoch_loss, 20.0)),
        "parameters": model.count_parameters(),
        "training_seconds": round(total_time, 1),
        "args": vars(args),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def configure_stdout() -> None:
    """Force UTF-8 so the report survives a cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--epochs", type=int, default=int(config.TRAIN["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(config.TRAIN["batch_size"]))
    parser.add_argument("--lr", type=float, default=float(config.TRAIN["lr"]))
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=int(config.TRAIN["warmup_steps"]))
    parser.add_argument("--grad-clip", type=float, default=float(config.TRAIN["grad_clip"]))
    parser.add_argument("--seed", type=int, default=int(config.TRAIN["seed"]))
    parser.add_argument("--max-len", type=int, default=int(config.MODEL["max_seq_len"]))
    parser.add_argument("--log-every", type=int, default=None, help="default 50")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--limit-train", type=int, default=None, help="use only N sequences")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps")
    parser.add_argument("--smoke", action="store_true", help="tiny fast run for sanity checks")
    parser.add_argument("--no-save", action="store_true", help="skip writing checkpoint/history")
    args = parser.parse_args(argv)

    if args.log_every is None:
        args.log_every = 10 if args.smoke else 50
    if args.smoke:
        args.epochs = min(int(args.epochs), 2)
        if args.limit_train is None:
            args.limit_train = 3000
        if args.max_steps is None:
            args.max_steps = 60
    return args




def print_summary(result: Dict[str, Any]) -> None:
    history = result["history"]
    best = result["best"]

    print("\n" + "=" * 74)
    print("PER-EPOCH HISTORY")
    print("=" * 74)
    print(f"  {'epoch':>5s}  {'train_loss':>10s}  {'train_ppl':>12s}  {'val_acc':>8s}  "
          f"{'val_acc_norm':>12s}  {'lr':>10s}")
    for entry in history:
        normalized = entry["eval"].get("accuracy_mean_logprob_normalized", float("nan"))
        print(
            f"  {entry['epoch']:>5d}  {entry['train_loss']:>10.4f}  "
            f"{entry['train_ppl']:>12.2f}  {100.0 * entry['val_accuracy']:>7.2f}%  "
            f"{100.0 * normalized:>11.2f}%  {entry['lr']:>10.6f}"
        )

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  epochs completed         : {len(history)}")
    print(f"  parameters               : {result['parameters']:,}")
    print(f"  BEST VAL ACCURACY        : {100.0 * best['val_accuracy']:.2f}%  "
          f"at epoch {best['epoch']}")
    print(f"  final train loss         : {result['final_train_loss']:.4f}")
    print(f"  FINAL TRAIN PERPLEXITY   : {result['final_train_ppl']:.2f}")
    above_random = best["val_accuracy"] > 0.25
    print(f"  above random (25%)       : {'YES' if above_random else 'NO'}  "
          f"({100.0 * (best['val_accuracy'] - 0.25):+.2f} points)")
    seconds = result["training_seconds"]
    print(f"  training time            : {seconds:.1f}s  ({seconds / 60.0:.1f} min)")
    if best.get("diagnostics"):
        diagnostics = best["diagnostics"]
        print(f"  best-epoch prediction dist : "
              f"{diagnostics.get('prediction_distribution')}")
        print(f"  best-epoch label dist      : {diagnostics.get('label_distribution')}")
        print(f"  mean scored tokens/choice  : "
              f"{diagnostics.get('mean_scored_tokens_per_choice')}")
        print(f"  length/score correlation   : "
              f"{diagnostics.get('length_score_correlation')}  "
              "(negative = sum-of-log-probs favours short choices)")
        print(f"  DIAGNOSTIC (not the metric): length-normalised mean log-prob "
              f"accuracy = "
              f"{100.0 * diagnostics.get('accuracy_mean_logprob_normalized', float('nan')):.2f}%")


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_stdout()
    args = parse_args(argv)
    config.ensure_dirs()

    print("=" * 74)
    print("TASK 21 — Train GPT-2 style from scratch (next-token prediction)")
    print("=" * 74)
    print(f"  model config : {config.MODEL}")
    print(f"  epochs {args.epochs} | batch {args.batch_size} | lr {args.lr} | "
          f"wd {args.weight_decay} | warmup {args.warmup_steps} | clip {args.grad_clip}")
    print(f"  max_len {args.max_len} | log_every {args.log_every} | "
          f"eval_batch {args.eval_batch_size} | seed {args.seed}")
    if args.smoke:
        print("  *** SMOKE RUN — reduced data/steps, NOT the experiment ***")
    print(flush=True)

    result = train(args)
    print_summary(result)

    if not args.no_save:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\n  checkpoint : {config.path('gpt2_ckpt')}")
        print(f"  history    : {HISTORY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
