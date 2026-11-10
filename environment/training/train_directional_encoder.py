"""STEP 2 of the directional encoder: train on continuation DIRECTIONS, not similarity.

The objective
-------------
The V4 experiments left one explanation standing: what the scorer needs is not "is this
ending similar to the context" but "does this continuation point in the right
direction", and no amount of memory or gating fixes a geometry that does not encode it
(TASK 29/30: memory +0.0039 in-corpus, -0.0049 out-of-corpus, and the best gate still
-0.0003 vs no memory). This trainer therefore optimises the DIRECTION between consecutive
sentences directly, with the structural relation from STEP 1 as the supervision:

    e1 = encode(s1), e2 = encode(s2)            (GMNStyle, 6 layers, 256-d, unit norm)
    d_i = normalise(e2 - e1)                    (the continuation direction, unit norm)

    for each anchor i in a batch of 32:
        j = another pair in the batch with the SAME relation     (positive)
        k = a pair in the batch with a DIFFERENT relation         (negative)
        loss_i = max(0, margin - cos(d_i, d_j) + cos(d_i, d_k))
    total_loss = mean(loss_i) over the anchors that have both      margin = 0.3

i.e. "elaboration continues like an elaboration, not like a new topic or a conclusion".
Anchors whose batch lacks a same-relation or a different-relation partner are SKIPPED
and counted in the log - a batch of 32 drawn from a corpus with 18% minority classes can
lack a negative for an elaboration anchor, and silently training on it would mean
optimising a zero gradient.

Cost (why the defaults are what they are)
-----------------------------------------
Measured on this CPU (14 threads): 0.79 s per step at padded length 64, 0.33 s with
dynamic padding. The literal budget (10 epochs x 50,000 pairs / batch 32) is 15,625 steps
= 3.4 h padded, 1.4 h dynamic; on top of that the BPE tokeniser is pure Python, so
tokenising all 100,000 sentences costs ~2.5 min per pass. So the trainer

  * pads dynamically, to the longest sequence IN THE BATCH (the model threads a
    ``key_padding_mask``, so padding never changes a sentence vector),
  * caches the token ids once (``--token-cache``), so restarts are seconds not minutes,
  * takes a per-epoch SUBSAMPLE (``--pairs-per-epoch``, default 8,000 = 2,500 steps,
    ~14 min total), following ``preprocess.py``'s stated convention that a trainer with a
    small step budget subsamples at training time instead of shrinking the artifact.

``--pairs-per-epoch 50000`` runs the literal specification (~1.4 h).

Artifacts
---------
    checkpoints/directional_encoder.pt                    config + weights + training meta
    environment/results/directional_training.json         per-epoch history + geometry probes
    environment/data/wikitext/directional_pairs_tokens.npz cached token ids

Usage (from the repository root)::

    python environment/training/train_directional_encoder.py --smoke
    python environment/training/train_directional_encoder.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ENVIRONMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ENVIRONMENT_DIR.parent
for _directory in (str(ENVIRONMENT_DIR), str(ENVIRONMENT_DIR / "models"), str(ENVIRONMENT_DIR / "tokenizer")):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import config  # noqa: E402
from bpe import PAD_ID, BPETokenizer  # noqa: E402
from gmn_style import GMNStyle  # noqa: E402

RELATIONS: Tuple[str, str, str] = ("elaboration", "new_topic", "conclusion")
RELATION_INDEX = {relation: index for index, relation in enumerate(RELATIONS)}
DEFAULT_PAIRS = config.path("wiki_directional_pairs")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "directional_encoder.pt"
DEFAULT_TOKEN_CACHE = config.path("wiki_directional_pairs").with_name(
    "directional_pairs_tokens.npz"
)
DEFAULT_HISTORY = ENVIRONMENT_DIR / "results" / "directional_training.json"

EPOCHS = 10
BATCH_SIZE = 32
MARGIN = 0.3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
PAIRS_PER_EPOCH = 8000
MAX_SEQ_LEN = 64
LOG_EVERY = 100
SEED = 42
PROBE_PAIRS = 300


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the directional consistency encoder (triplet loss on directions)."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--tokenizer", type=Path, default=config.path("tokenizer_wiki"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--pairs-per-epoch", type=int, default=PAIRS_PER_EPOCH)
    parser.add_argument(
        "--pairs-limit",
        type=int,
        default=0,
        help="use only the first N labeled pairs (0 = all; smoke runs set this)",
    )
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--probe-pairs", type=int, default=PROBE_PAIRS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force-retokenize", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run: 2 epochs x 320 pairs, to check the plumbing end to end",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs = 2
        args.pairs_per_epoch = 320
        args.pairs_limit = 400
        args.token_cache = args.token_cache.with_name(args.token_cache.stem + "_smoke.npz")
        args.log_every = 5
        args.probe_pairs = 90
        args.warmup_steps = 10
    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_pairs(path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Stream ``{"s1", "s2", "relation"}`` objects out of the one-per-line JSON array."""
    pairs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith("["):
                text = text[1:]
            if text.endswith(","):
                text = text[:-1]
            text = text.strip()
            if not text or text == "]" or text == "[]":
                continue
            if text.endswith("]"):
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    item = json.loads(text[:-1])
            else:
                item = json.loads(text)
            if not isinstance(item, dict):
                continue
            relation = str(item["relation"])
            if relation not in RELATION_INDEX:
                continue
            pairs.append({"s1": str(item["s1"]), "s2": str(item["s2"]), "relation": relation})
            if limit and len(pairs) >= limit:
                break
    if not pairs:
        raise ValueError(f"{path} contained no labeled pairs")
    return pairs


def tokenize_pairs(
    pairs: Sequence[Dict[str, str]],
    tokenizer: BPETokenizer,
    cache_path: Path,
    max_seq_len: int,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ids1, ids2, lengths)`` as ``(N, max_seq_len - 1)`` int32 arrays, cached on disk.

    Two conventions follow ``GMNStyle.encode``: it prepends ``[CLS]`` INTERNALLY
    (``prepend_cls=True``), so the tokenizer adds no special tokens and the cap is
    ``max_seq_len - 1`` to leave room for that column; and the sequence cap is only a
    safety net, because batches are padded to the longest sequence in the batch.
    """
    cache_path = Path(cache_path)
    capacity = max(1, int(max_seq_len) - 1)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if int(cached["count"]) == len(pairs) and int(cached["max_seq_len"]) == max_seq_len:
                print(f"  loaded cached token ids from {cache_path.name}")
                return (
                    cached["ids1"].astype(np.int32),
                    cached["ids2"].astype(np.int32),
                    cached["lengths"].astype(np.int32),
                )
    print(f"  tokenising {2 * len(pairs):,} sentences with the BPE tokenizer ...")
    started = time.perf_counter()
    ids1 = np.zeros((len(pairs), capacity), dtype=np.int32)
    ids2 = np.zeros((len(pairs), capacity), dtype=np.int32)
    lengths = np.zeros(len(pairs), dtype=np.int32)
    for index, pair in enumerate(pairs):
        first = tokenizer.encode(pair["s1"], capacity, add_special_tokens=False)
        second = tokenizer.encode(pair["s2"], capacity, add_special_tokens=False)
        ids1[index, : len(first)] = first
        ids2[index, : len(second)] = second
        lengths[index] = max(len(first), len(second))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        ids1=ids1,
        ids2=ids2,
        lengths=lengths,
        count=np.int64(len(pairs)),
        max_seq_len=np.int64(max_seq_len),
    )
    print(f"  tokenised in {time.perf_counter() - started:.1f}s -> {cache_path.name}")
    return ids1, ids2, lengths


def stack_batch(
    ids: np.ndarray, lengths: np.ndarray, indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch to its own longest sequence -> ``(token_ids, padding_mask)``.

    ``padding_mask`` is ``True`` at PADDED positions, the convention
    ``GMNStyle.encode`` documents for ``key_padding_mask`` (``nn.MultiheadAttention``
    style), so padded rows never influence a sentence vector.
    """
    width = max(2, int(lengths[indices].max()))
    batch = np.stack([ids[int(index), :width] for index in indices]).astype(np.int64)
    tensor = torch.from_numpy(batch)
    return tensor, tensor == PAD_ID


def encode_directions(
    model: GMNStyle,
    ids1: np.ndarray,
    ids2: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    """Unit continuation directions ``normalise(encode(s2) - encode(s1))`` for ``indices``."""
    directions: List[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        first_ids, first_mask = stack_batch(ids1, lengths, chunk)
        second_ids, second_mask = stack_batch(ids2, lengths, chunk)
        first = model.encode(first_ids, key_padding_mask=first_mask)
        second = model.encode(second_ids, key_padding_mask=second_mask)
        directions.append(F.normalize(second - first, p=2, dim=-1))
    return torch.cat(directions, dim=0)




# ---------------------------------------------------------------------------
# The directional consistency loss
# ---------------------------------------------------------------------------
def directional_triplet_loss(
    directions: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    rng: np.random.Generator,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """``mean_i max(0, margin - cos(d_i, d_j) + cos(d_i, d_k))`` -> ``(loss, stats)``.

    ``directions`` are unit-norm continuation directions, so the cosine matrix is one
    matmul. ``labels`` are relation indices; for each anchor ``i`` a positive ``j`` is
    drawn uniformly from the same relation and a negative ``k`` from a different one
    (one draw each, as the task specifies - no hard-negative mining). Anchors that have
    no same-relation partner or no different-relation partner in the batch are skipped
    and counted: with ~18% minority classes a random batch of 32 can lack a negative
    for an elaboration anchor, and training on it would mean optimising a zero gradient
    while pretending to learn.

    ``loss`` is ``None`` when fewer than one anchor survives (a batch with one single
    relation), which the caller treats as "no step taken".
    """
    similarity = directions @ directions.t()  # (B, B) cosine
    size = int(directions.shape[0])
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(size, dtype=torch.bool)

    losses: List[torch.Tensor] = []
    positive_cosines: List[float] = []
    negative_cosines: List[float] = []
    violations = 0
    skipped = 0
    for anchor in range(size):
        positives = torch.nonzero(same[anchor] & ~eye[anchor], as_tuple=False).flatten()
        negatives = torch.nonzero(~same[anchor], as_tuple=False).flatten()
        if positives.numel() == 0 or negatives.numel() == 0:
            skipped += 1
            continue
        positive = int(positives[int(rng.integers(positives.numel()))])
        negative = int(negatives[int(rng.integers(negatives.numel()))])
        cosine_positive = similarity[anchor, positive]
        cosine_negative = similarity[anchor, negative]
        hinge = F.relu(margin - cosine_positive + cosine_negative)
        if float(hinge) > 0.0:
            violations += 1
        losses.append(hinge)
        positive_cosines.append(float(cosine_positive.detach()))
        negative_cosines.append(float(cosine_negative.detach()))

    if not losses:
        return None, {
            "anchors": 0,
            "skipped": skipped,
            "violations": 0,
            "mean_cos_positive": 0.0,
            "mean_cos_negative": 0.0,
            "direction_gap": 0.0,
        }

    mean_cos_positive = float(np.mean(positive_cosines))
    mean_cos_negative = float(np.mean(negative_cosines))
    return torch.stack(losses).mean(), {
        "anchors": len(losses),
        "skipped": skipped,
        "violations": violations,
        "violation_rate": violations / len(losses),
        "mean_cos_positive": mean_cos_positive,
        "mean_cos_negative": mean_cos_negative,
        "direction_gap": mean_cos_positive - mean_cos_negative,
    }


# ---------------------------------------------------------------------------
# Geometry probe (a training diagnostic; STEP 3 is the verification)
# ---------------------------------------------------------------------------
def geometry_probe(
    model: GMNStyle,
    ids1: np.ndarray,
    ids2: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Within-relation consistency and cross-relation cosine on a fixed probe set.

    ``labels`` is the ``(N,)`` int array of relation indices aligned with the pairs, so
    ``consistency(relation) = mean cos(d_i, mean_direction(relation))`` (high = that
    relation's directions cluster) and ``cross(a, b)`` is the cosine between the two mean
    directions (low = the relations occupy different directions). This is a TRAINING
    diagnostic run every epoch; STEP 3 does the 500-pair-per-relation verification.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        directions = encode_directions(model, ids1, ids2, lengths, indices, batch_size=64)
    model.train(was_training)

    subgroup_labels = labels[indices]
    means: Dict[str, torch.Tensor] = {}
    consistency: Dict[str, float] = {}
    for index, relation in enumerate(RELATIONS):
        mask = torch.from_numpy(subgroup_labels == index)
        subgroup = directions[mask]
        if subgroup.shape[0] == 0:
            continue
        mean = F.normalize(subgroup.mean(dim=0), p=2, dim=0)
        means[relation] = mean
        consistency[relation] = float((subgroup @ mean).mean())
    cross = {
        f"{first}|{second}": float(means[first] @ means[second])
        for position, first in enumerate(RELATIONS)
        for second in RELATIONS[position + 1 :]
        if first in means and second in means
    }
    return {"consistency": consistency, "cross": cross}




# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def learning_rate_at(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup, then constant - the TASK 18R ``warmup_steps`` convention."""
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / float(warmup_steps))


def save_checkpoint(
    path: Path,
    model: GMNStyle,
    tokenizer_path: Path,
    training: Dict[str, Any],
) -> float:
    """``config`` + weights + metadata, the layout ``GMNStyle.from_model_name`` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": dict(model.config),
            "model_state_dict": model.state_dict(),
            "tokenizer": str(tokenizer_path),
            "name": "directional_encoder",
            "objective": "directional consistency triplet on continuation directions",
            "latent_dim": model.latent_dim,
            "parameter_count": model.count_parameters(),
            "training": training,
        },
        path,
    )
    return path.stat().st_size / (1024 * 1024)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()

    steps_per_epoch = max(1, args.pairs_per_epoch // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print("=" * 92)
    print("STEP 2 - directional consistency encoder (triplet loss on continuation directions)")
    print("=" * 92)
    print(
        f"  pairs          : {args.pairs}\n"
        f"  budget         : {args.epochs} epochs x {args.pairs_per_epoch:,} pairs / batch "
        f"{args.batch_size} = {total_steps:,} steps\n"
        f"  objective      : max(0, {args.margin} - cos(d_i,d_j) + cos(d_i,d_k)), "
        f"d = normalise(e(s2) - e(s1))\n"
        f"  optimiser      : AdamW lr {args.lr} wd {args.weight_decay}, warmup "
        f"{args.warmup_steps} steps, grad clip {args.grad_clip}\n"
        f"  checkpoint     : {args.output}"
    )

    pairs = load_pairs(args.pairs, limit=args.pairs_limit)
    counts = Counter(pair["relation"] for pair in pairs)
    print(f"\n  loaded {len(pairs):,} labeled pairs:")
    for relation in RELATIONS:
        print(f"    {relation:<12} {counts[relation]:>8,}")

    tokenizer = BPETokenizer.load(args.tokenizer)
    ids1, ids2, lengths = tokenize_pairs(
        pairs, tokenizer, args.token_cache, args.max_seq_len, force=args.force_retokenize
    )
    labels = np.asarray([RELATION_INDEX[pair["relation"]] for pair in pairs], dtype=np.int64)

    model = GMNStyle(config_dict=config.MODEL_SMALL).to(device)
    print(
        f"\n  encoder        : GMNStyle(MODEL_SMALL) {model.count_parameters():,} params | "
        f"{model.config['n_layers']} layers, hidden {model.config['hidden_dim']}, "
        f"vocab {model.config['vocab_size']} | latent {model.latent_dim}"
    )

    probe_size = min(args.probe_pairs, len(pairs))
    probe_indices = np.arange(probe_size, dtype=np.int64)
    before = geometry_probe(model, ids1, ids2, lengths, probe_indices, labels)
    print(
        f"  geometry before training (probe {probe_size} pairs): consistency "
        + ", ".join(f"{key} {value:+.3f}" for key, value in before["consistency"].items())
        + " | cross "
        + ", ".join(f"{key} {value:+.3f}" for key, value in before["cross"].items())
    )
    return train_loop(args, model, ids1, ids2, lengths, labels, counts, before, started, rng,
                      steps_per_epoch, total_steps)



def probe_indices_of(args: argparse.Namespace, labels: np.ndarray) -> np.ndarray:
    """The fixed probe set used by the per-epoch geometry diagnostic (first N pairs)."""
    return np.arange(min(args.probe_pairs, labels.shape[0]), dtype=np.int64)


def train_loop(
    args: argparse.Namespace,
    model: GMNStyle,
    ids1: np.ndarray,
    ids2: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    counts: Counter,
    before: Dict[str, Any],
    started: float,
    rng: np.random.Generator,
    steps_per_epoch: int,
    total_steps: int,
) -> Dict[str, Any]:
    """The epoch/batch loop: triplet loss, warmup, per-epoch geometry probe, checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    model.train()
    history: List[Dict[str, Any]] = []
    step = 0
    running: List[float] = []
    skipped_anchors = 0
    logged_anchors = 0
    logged_violations = 0
    positives: List[float] = []
    negatives: List[float] = []

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(labels))[: args.pairs_per_epoch]
        epoch_losses: List[float] = []
        epoch_started = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            if len(batch) < 2:
                continue
            first_ids, first_mask = stack_batch(ids1, lengths, batch)
            second_ids, second_mask = stack_batch(ids2, lengths, batch)
            first = model.encode(first_ids, key_padding_mask=first_mask)
            second = model.encode(second_ids, key_padding_mask=second_mask)
            directions = F.normalize(second - first, p=2, dim=-1)
            loss, stats = directional_triplet_loss(
                directions, torch.from_numpy(labels[batch]), args.margin, rng
            )
            skipped_anchors += int(stats["skipped"])
            if loss is None:  # one single relation in the batch: nothing to learn from
                continue
            lr = learning_rate_at(step, args.lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1

            running.append(float(loss.detach()))
            epoch_losses.append(float(loss.detach()))
            positives.append(stats["mean_cos_positive"])
            negatives.append(stats["mean_cos_negative"])
            logged_anchors += int(stats["anchors"])
            logged_violations += int(stats["violations"])

            if args.log_every and step % args.log_every == 0:
                window = min(args.log_every, len(running))
                print(
                    f"    step {step:>5}/{total_steps} | loss {np.mean(running[-window:]):.4f} | "
                    f"cos(pos) {np.mean(positives[-window:]):+.3f} | "
                    f"cos(neg) {np.mean(negatives[-window:]):+.3f} | "
                    f"violations {100 * logged_violations / max(1, logged_anchors):4.1f}% | "
                    f"lr {lr:.2e} | {(time.perf_counter() - started) / 60:.1f} min"
                )

        probe = geometry_probe(model, ids1, ids2, lengths, probe_indices_of(args, labels), labels)
        entry = {
            "epoch": epoch,
            "steps": step,
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "seconds": time.perf_counter() - epoch_started,
            "probe_consistency": probe["consistency"],
            "probe_cross": probe["cross"],
        }
        history.append(entry)
        print(
            f"  epoch {epoch}/{args.epochs} | loss {entry['mean_loss']:.4f} | consistency "
            + " ".join(
                f"{key[:4]} {value:+.3f}" for key, value in probe["consistency"].items()
            )
            + " | cross "
            + " ".join(f"{key} {value:+.3f}" for key, value in probe["cross"].items())
            + f" | {entry['seconds']:.0f}s"
        )
    return finish(args, model, (ids1, ids2, lengths), labels, counts, before, history,
                  started, step, steps_per_epoch, skipped_anchors, logged_anchors,
                  logged_violations, running, positives, negatives)



def finish(
    args: argparse.Namespace,
    model: GMNStyle,
    probe_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: np.ndarray,
    counts: Counter,
    before: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    started: float,
    step: int,
    steps_per_epoch: int,
    skipped_anchors: int,
    logged_anchors: int,
    logged_violations: int,
    running: Sequence[float],
    positives: Sequence[float],
    negatives: Sequence[float],
) -> Dict[str, Any]:
    """Final geometry probe, checkpoint, history JSON and the summary table."""
    ids1, ids2, lengths = probe_arrays
    final_probe = geometry_probe(
        model, ids1, ids2, lengths, probe_indices_of(args, labels), labels
    )
    training = {
        "pairs": str(args.pairs),
        "pairs_loaded": int(labels.shape[0]),
        "relation_counts": {str(key): int(value) for key, value in counts.items()},
        "epochs": args.epochs,
        "pairs_per_epoch": args.pairs_per_epoch,
        "batch_size": args.batch_size,
        "margin": args.margin,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "warmup_steps": args.warmup_steps,
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "skipped_anchors": skipped_anchors,
        "loss": float(np.mean(running)) if running else 0.0,
        "mean_cos_positive": float(np.mean(positives)) if positives else 0.0,
        "mean_cos_negative": float(np.mean(negatives)) if negatives else 0.0,
        "violation_rate": logged_violations / max(1, logged_anchors),
        "history": list(history),
        "geometry_before": before,
        "geometry_final": final_probe,
        "seed": args.seed,
        "wall_seconds": time.perf_counter() - started,
    }
    size_mb = save_checkpoint(args.output, model, args.tokenizer, training)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("w", encoding="utf-8") as handle:
        json.dump({"task": "STEP 2 - directional encoder", **training}, handle, indent=2)

    print("\n" + "=" * 92)
    print("STEP 2 SUMMARY")
    print("=" * 92)
    header = " | ".join(f"{relation[:11]:>11}" for relation in RELATIONS)
    print(f"  epoch | train loss | {header}   (within-relation consistency, training probe)")
    print("  " + "-" * 78)
    for entry in history:
        row = " | ".join(
            f"{entry['probe_consistency'].get(relation, float('nan')):>+11.3f}"
            for relation in RELATIONS
        )
        print(f"  {entry['epoch']:>5} | {entry['mean_loss']:>10.4f} | {row}")
    print("  " + "-" * 78)
    print(
        f"  training: {step:,} steps | mean loss {training['loss']:.4f} | cos(pos) "
        f"{training['mean_cos_positive']:+.3f} vs cos(neg) {training['mean_cos_negative']:+.3f} "
        f"(gap {training['mean_cos_positive'] - training['mean_cos_negative']:+.3f}) | margin "
        f"violations {100 * training['violation_rate']:.1f}% | skipped anchors "
        f"{skipped_anchors:,}"
    )
    print(f"  saved {args.output} ({size_mb:.1f} MB) and {args.history}")
    print(f"\nDone in {training['wall_seconds'] / 60:.1f} min")
    print("=" * 92)
    return training


if __name__ == "__main__":
    main()

