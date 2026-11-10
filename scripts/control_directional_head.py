"""Diagnostic control for STEP 2: is the failure the OBJECTIVE, or the random encoder?

STEP 2's run ended where it started: mean loss 0.3033 against the 0.3 margin, 94.7% of
anchors violating it, and STEP 3 measured within-relation consistency +0.0373 in-sample
/ -0.0032 held-out - statistically identical to an UNTRAINED GMNStyle. Before blaming
the labels (the "try an RST discourse dataset" branch), it helps to know which of two
very different causes it is:

  A. the objective/labels carry no learnable signal at all, or
  B. the objective is fine but a randomly initialised 8.9M transformer cannot build a
     relation-sensitive sentence representation in 2,500 steps.

This control separates them. It trains the SAME triplet objective and the same margin on
FROZEN sentence latents from the V4 continuation encoder (which already encodes meaning
usefully) through a small trainable head::

    d = normalise(head(e(s2)) - head(e(s1)))          head: 256 -> 256 -> 256

If the geometry organises here (consistency up, cross-relation cosines down) the
objective is learnable and the STEP 2 failure is (B): the encoder needed pretraining,
not different labels. If it does not organise, the labels themselves are the problem and
the RST-discourse branch is the right next step.

Coverage note: only the labeled pairs whose BOTH sentences already have cached latents
can be used (~20% of the 50,001 - the labeled set reaches further into the corpus than
the caches), which is why the control reports its own pair count.

This is a diagnostic, not a deliverable: it writes no checkpoint.

Artifacts
---------
    environment/results/directional_head_control.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
for _directory in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "scripts"),
    str(ENVIRONMENT_DIR),
    str(ENVIRONMENT_DIR / "models"),
    str(ENVIRONMENT_DIR / "tokenizer"),
    str(ENVIRONMENT_DIR / "training"),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import config  # noqa: E402
from check_directional_geometry import RELATIONS, relation_geometry  # noqa: E402
from train_directional_encoder import (  # noqa: E402
    RELATION_INDEX,
    directional_triplet_loss,
    load_pairs,
)

LATENT_CACHES = (
    PROJECT_ROOT / "checkpoints" / "prediction_loop_train_vecs.npz",
    PROJECT_ROOT / "checkpoints" / "wikitext_val_predictor_vecs_continuation.npz",
)
RESULTS_PATH = ENVIRONMENT_DIR / "results" / "directional_head_control.json"
PAIRS_PATH = config.path("wiki_directional_pairs")

HIDDEN = 256
BATCH_SIZE = 32
MARGIN = 0.3
LEARNING_RATE = 1e-3
STEPS = 1500
LOG_EVERY = 250
PROBE_FRACTION = 0.2
PAIRWISE_SAMPLES = 2000
SEED = 42


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control: triplet geometry on frozen latents through a trainable head."
    )
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--probe-fraction", type=float, default=PROBE_FRACTION)
    parser.add_argument(
        "--relations",
        nargs="+",
        default=list(RELATIONS),
        help=(
            "restrict the objective to these relations. Dropping 'conclusion' tests the "
            "hypothesis that an unlearnable (paragraph-position) class contaminates the "
            "objective for the learnable ones"
        ),
    )
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


class DirectionHead(nn.Module):
    """``256 -> 256 -> 256`` MLP over frozen sentence latents (the trainable part)."""

    def __init__(self, dim: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.net(vectors)


def load_latent_lookup() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """``sentence -> unit latent`` from the cached V4 continuation-encoder latents."""
    lookup: Dict[str, np.ndarray] = {}
    stats: Dict[str, Any] = {}
    for path in LATENT_CACHES:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as cached:
            sentences = [str(value) for value in cached["sentences"]]
            vectors = cached["vectors"].astype(np.float32)
        for sentence, vector in zip(sentences, vectors):
            lookup.setdefault(sentence, vector)
        stats[path.name] = len(sentences)
    return lookup, stats


def gather_pairs(
    pairs: Sequence[Dict[str, str]],
    lookup: Dict[str, np.ndarray],
    relations: Sequence[str] = RELATIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(s1_latents, s2_latents, relation_indices)`` for pairs whose sentences are cached.

    ``relations`` restricts the objective to a subset. That matters because a relation can
    be unlearnable from a sentence PAIR: "conclusion" means "s2 is the last sentence of its
    paragraph", which no single sentence reveals - so requiring its direction to separate
    from the other two forces the optimiser to compromise, and that compromise also
    damages the relations that ARE learnable (elaboration vs new_topic, i.e. does s2 stay
    on topic).
    """
    first: List[np.ndarray] = []
    second: List[np.ndarray] = []
    labels: List[int] = []
    for pair in pairs:
        if pair["relation"] not in relations:
            continue
        left = lookup.get(pair["s1"])
        right = lookup.get(pair["s2"])
        if left is None or right is None:
            continue
        first.append(left)
        second.append(right)
        labels.append(RELATION_INDEX[pair["relation"]])
    return (
        np.stack(first) if first else np.zeros((0, HIDDEN), dtype=np.float32),
        np.stack(second) if second else np.zeros((0, HIDDEN), dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def head_directions(
    head: DirectionHead, first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """``normalise(head(e2) - head(e1))`` - the same direction shape as the encoder's."""
    return F.normalize(head(second) - head(first), p=2, dim=-1)


@torch.no_grad()
def measure(
    head: DirectionHead,
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Geometry metrics on a fixed set, reusing the STEP 3 metric definitions."""
    head.eval()
    directions = head_directions(
        head, torch.from_numpy(first), torch.from_numpy(second)
    ).numpy()
    chosen: Dict[str, np.ndarray] = {}
    order: List[int] = []
    for index, relation in enumerate(RELATIONS):
        local = np.flatnonzero(labels == index)
        if local.size == 0:  # a relation absent from this arm
            continue
        chosen[relation] = np.arange(len(order), len(order) + local.size)
        order.extend(local.tolist())
    return relation_geometry(
        directions[np.asarray(order, dtype=np.int64)], chosen, PAIRWISE_SAMPLES, rng
    )


def print_measure(name: str, geometry: Dict[str, Any]) -> None:
    consistency_in = float(np.mean(list(geometry["within_relation_in_sample"].values())))
    consistency_out = float(np.mean(list(geometry["within_relation_held_out"].values())))
    print(
        f"  {name:<26} consistency in-sample {consistency_in:+.4f} | held-out "
        f"{consistency_out:+.4f} | cross max |cos| "
        f"{geometry['max_cross_relation_cosine']:.4f} | separation "
        f"{geometry['relation_separation']:+.4f}"
    )



def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    print("=" * 96)
    print("CONTROL - is the directional objective learnable on USABLE sentence embeddings?")
    print("=" * 96)

    pairs = load_pairs(args.pairs)
    lookup, cache_stats = load_latent_lookup()
    first, second, labels = gather_pairs(pairs, lookup, args.relations)
    print(
        f"\n  arms: {', '.join(args.relations)} | labeled pairs {len(pairs):,} | cached "
        f"sentence latents {sum(cache_stats.values()):,} ({', '.join(cache_stats)}) | "
        f"usable pairs {labels.shape[0]:,} ({100 * labels.shape[0] / len(pairs):.1f}%)"
    )
    if labels.shape[0] < 300:
        raise SystemExit("not enough cached pairs for a meaningful control")
    for index, relation in enumerate(RELATIONS):
        print(f"    {relation:<12} {int((labels == index).sum()):>6,}")

    # split: the head trains on the first part, the probe measures on the rest
    order = rng.permutation(labels.shape[0])
    probe_size = max(60, int(args.probe_fraction * labels.shape[0]))
    probe_idx, train_idx = order[:probe_size], order[probe_size:]
    print(
        f"  train {train_idx.size:,} pairs | probe {probe_idx.size:,} pairs "
        f"(the probe set is never trained on)"
    )

    head = DirectionHead()
    before = measure(head, first[probe_idx], second[probe_idx], labels[probe_idx], rng)
    print("\n  geometry on the probe set")
    print_measure("before training", before)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    head.train()
    first_t = torch.from_numpy(first)
    second_t = torch.from_numpy(second)
    labels_t = torch.from_numpy(labels)
    step = 0
    running: List[float] = []
    gaps: List[float] = []
    while step < args.steps:
        batch = np.asarray(rng.choice(train_idx, size=args.batch_size, replace=False))
        directions = head_directions(head, first_t[batch], second_t[batch])
        loss, stats = directional_triplet_loss(
            directions, labels_t[batch], args.margin, rng
        )
        if loss is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        running.append(float(loss.detach()))
        gaps.append(stats["direction_gap"])
        if step % args.log_every == 0:
            window = min(args.log_every, len(running))
            print(
                f"    step {step:>5}/{args.steps} | loss {np.mean(running[-window:]):.4f} | "
                f"cos(pos)-cos(neg) {np.mean(gaps[-window:]):+.4f} | "
                f"violations {100 * stats['violation_rate']:.1f}%"
            )

    after = measure(head, first[probe_idx], second[probe_idx], labels[probe_idx], rng)
    print_measure("after training", after)

    consistency_before = float(np.mean(list(before["within_relation_held_out"].values())))
    consistency_after = float(np.mean(list(after["within_relation_held_out"].values())))
    organised = consistency_after > consistency_before + 0.1 and after[
        "max_cross_relation_cosine"
    ] < before["max_cross_relation_cosine"]

    print("\n  DIAGNOSIS")
    print(
        f"    held-out consistency {consistency_before:+.4f} -> {consistency_after:+.4f} | "
        f"cross max |cos| {before['max_cross_relation_cosine']:.4f} -> "
        f"{after['max_cross_relation_cosine']:.4f} | separation "
        f"{before['relation_separation']:+.4f} -> {after['relation_separation']:+.4f}"
    )
    if organised:
        print(
            "    the objective IS learnable when the sentence embeddings are usable -> "
            "STEP 2's failure was the randomly initialised encoder (it needed pretraining, "
            "not different labels)"
        )
    else:
        print(
            "    the objective does NOT organise the geometry even on usable embeddings -> "
            "the labels themselves are the bottleneck (the RST-discourse branch is the "
            "right next step)"
        )

    payload = {
        "task": "STEP 2 diagnostic control - frozen latents + trainable direction head",
        "config": {
            "pairs": str(args.pairs),
            "usable_pairs": int(labels.shape[0]),
            "train_pairs": int(train_idx.size),
            "probe_pairs": int(probe_size),
            "steps": step,
            "batch_size": args.batch_size,
            "margin": args.margin,
            "learning_rate": args.lr,
        },
        "latent_caches": cache_stats,
        "geometry_before": before,
        "geometry_after": after,
        "loss": float(np.mean(running)) if running else 0.0,
        "direction_gap": float(np.mean(gaps)) if gaps else 0.0,
        "diagnosis": {
            "consistency_before": consistency_before,
            "consistency_after": consistency_after,
            "organised": bool(organised),
            "interpretation": (
                "learnable objective + random encoder was the bottleneck"
                if organised
                else "labels/objective are the bottleneck"
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  saved {args.results}")
    print(f"  done in {payload['wall_seconds']:.1f}s")
    print("=" * 96)
    return payload


if __name__ == "__main__":
    main()

