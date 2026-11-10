"""STEP 3 of the directional encoder: does the direction geometry actually separate?

The claim to verify: after STEP 2's triplet training, the continuation direction
``normalise(e(s2) - e(s1))`` should depend on the STRUCTURAL RELATION (elaboration /
new_topic / conclusion) rather than on surface similarity, i.e.

  * mean directions of the three relations point in different directions
    (``cos(mean_a, mean_b)`` small - the task's target is < 0.3), and
  * directions of the same relation cluster around their own mean
    (``mean cos(d_i, mean_relation)`` large - the task's target is > 0.5).

Measured three ways, because the headline statistic is easy to fool:

  within_relation_in_sample   ``mean cos(d_i, mean over the same n samples)``. The task
                              asks for this, but it is biased UPWARD: every direction is
                              part of the mean it is compared against, so at n = 500 the
                              self-term alone contributes ~1/500 of the norm and small
                              groups look more consistent than they are.
  within_relation_held_out    split each relation's sample in half: mean from the first
                              half, cosine on the SECOND half. Unbiased, and the number
                              to trust when the in-sample value is close to the target.
  relation_separation         mean cos(d_i, d_j) for same-relation pairs minus the same
                              quantity for different-relation pairs, computed on pairwise
                              samples (no mean involved at all). > 0 means the directions
                              carry the relation; 0 means they do not.

A random / untrained encoder is reported alongside as the reference point, since an
untrained 8.9M transformer already produces *some* separation by accident.

Also saved: the UMAP projection of the direction vectors, coloured by relation.

Artifacts
---------
    notebooks/directional_geometry.png
    environment/results/directional_geometry.json

Usage::

    python scripts/check_directional_geometry.py
    python scripts/check_directional_geometry.py --baseline   # also score an untrained encoder
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
for _directory in (
    str(PROJECT_ROOT),
    str(ENVIRONMENT_DIR),
    str(ENVIRONMENT_DIR / "models"),
    str(ENVIRONMENT_DIR / "tokenizer"),
    str(ENVIRONMENT_DIR / "training"),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import config  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from gmn_style import GMNStyle  # noqa: E402
from train_directional_encoder import (  # noqa: E402
    RELATIONS,
    RELATION_INDEX,
    encode_directions,
    load_pairs,
    stack_batch,
    tokenize_pairs,
)

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "directional_encoder.pt"
PAIRS_PATH = config.path("wiki_directional_pairs")
TOKEN_CACHE = config.path("wiki_directional_pairs").with_name("directional_pairs_tokens.npz")
TOKENIZER_PATH = config.path("tokenizer_wiki")
PLOT_PATH = PROJECT_ROOT / "notebooks" / "directional_geometry.png"
RESULTS_PATH = ENVIRONMENT_DIR / "results" / "directional_geometry.json"

PAIRS_PER_RELATION = 500
PAIRWISE_SAMPLES = 2000
MAX_SEQ_LEN = 64
SEED = 42
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
DPI = 150
#: the task's decision thresholds
VALIDATED_CONSISTENCY = 0.5
WEAK_CONSISTENCY = 0.3
VALIDATED_CROSS = 0.3
PLOT_LIMIT = 1500


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the directional geometry of the trained encoder."
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--plot", type=Path, default=PLOT_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--pairs-per-relation", type=int, default=PAIRS_PER_RELATION)
    parser.add_argument("--pairwise-samples", type=int, default=PAIRWISE_SAMPLES)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also score an untrained GMNStyle(small) as the reference point",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def load_encoder(checkpoint: Path, device: torch.device) -> Tuple[GMNStyle, Dict[str, Any]]:
    """Rebuild the encoder from its checkpoint (``config`` + weights, GMNStyle layout)."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GMNStyle(config_dict=payload["config"]).to(device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model, payload


def sample_indices(
    labels: np.ndarray,
    pairs_per_relation: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """A fixed, class-balanced sample of pair indices (``pairs_per_relation`` per relation)."""
    rng = np.random.default_rng(seed)
    chosen: Dict[str, np.ndarray] = {}
    for index, relation in enumerate(RELATIONS):
        available = np.flatnonzero(labels == index)
        take = min(pairs_per_relation, available.size)
        chosen[relation] = np.sort(rng.choice(available, size=take, replace=False))
    stacked = np.concatenate([chosen[relation] for relation in RELATIONS])
    return stacked, chosen


@torch.no_grad()
def directions_for(
    model: GMNStyle,
    ids1: np.ndarray,
    ids2: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Unit continuation directions for ``indices`` as a numpy array."""
    return encode_directions(model, ids1, ids2, lengths, indices, batch_size=64).cpu().numpy()


def relation_geometry(
    directions: np.ndarray,
    chosen: Dict[str, np.ndarray],
    pairwise_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """In-sample and held-out consistency, mean directions and cross-relation cosines.

    ``chosen`` maps relation -> indices into ``directions``. Relations without samples are
    skipped, so a caller can score a subset of the relations (the diagnostic control runs
    a two-relation arm).
    """
    relations = [relation for relation in RELATIONS if relation in chosen]
    blocks: Dict[str, np.ndarray] = {}
    offset = 0
    for relation in relations:
        count = chosen[relation].size
        blocks[relation] = directions[offset : offset + count]
        offset += count

    means: Dict[str, np.ndarray] = {}
    in_sample: Dict[str, float] = {}
    held_out: Dict[str, float] = {}
    for relation in relations:
        block = blocks[relation]
        half = block.shape[0] // 2
        mean = block.mean(axis=0)
        means[relation] = mean / max(1e-12, float(np.linalg.norm(mean)))
        in_sample[relation] = float((block @ means[relation]).mean())
        # unbiased: the mean comes from the first half, the cosine from the second
        held_mean = block[:half].mean(axis=0)
        held_mean = held_mean / max(1e-12, float(np.linalg.norm(held_mean)))
        held_out[relation] = float((block[half:] @ held_mean).mean())

    cross = {
        f"{first}|{second}": float(means[first] @ means[second])
        for position, first in enumerate(relations)
        for second in relations[position + 1 :]
    }

    def pairwise(relation: str, other: Optional[str], count: int) -> float:
        first_block = blocks[relation]
        second_block = blocks[relation] if other is None else blocks[other]
        left = rng.integers(0, first_block.shape[0], size=count)
        right = rng.integers(0, second_block.shape[0], size=count)
        keep = left != right if other is None else np.ones(count, dtype=bool)
        return float(
            (first_block[left[keep]] * second_block[right[keep]]).sum(axis=1).mean()
        )

    same_relation = float(
        np.mean([pairwise(relation, None, pairwise_samples) for relation in relations])
    )
    different_relation = (
        float(
            np.mean(
                [
                    pairwise(first, second, pairwise_samples)
                    for position, first in enumerate(relations)
                    for second in relations[position + 1 :]
                ]
            )
        )
        if len(relations) > 1
        else 0.0
    )
    return {
        "relations": relations,
        "mean_direction_cosines": cross,
        "within_relation_in_sample": in_sample,
        "within_relation_held_out": held_out,
        "pairwise_same_relation": same_relation,
        "pairwise_different_relation": different_relation,
        "relation_separation": same_relation - different_relation,
        "max_cross_relation_cosine": (
            max(abs(value) for value in cross.values()) if cross else 0.0
        ),
    }



def plot_geometry(
    directions: np.ndarray,
    chosen: Dict[str, np.ndarray],
    means: Dict[str, np.ndarray],
    output_path: Path,
    title_suffix: str,
) -> Dict[str, Any]:
    """UMAP of the direction vectors, coloured by relation; also projects the means.

    Follows ``scripts/visualize_latent.py``: cosine metric, fixed random state, legend,
    and the same 2-d projection for the three mean directions so the cluster centres are
    visible next to the samples they summarise.
    """
    import umap  # imported here so the dependency requirement is explicit

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    stacked_means = np.stack([means[relation] for relation in RELATIONS])
    combined = np.concatenate([directions, stacked_means], axis=0)
    embedding = reducer.fit_transform(combined)
    sample_embedding = embedding[: directions.shape[0]]
    mean_embedding = embedding[directions.shape[0] :]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 7))
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    labels: List[int] = []
    offset = 0
    for color, relation in zip(colors, RELATIONS):
        count = chosen[relation].size
        labels.extend([RELATION_INDEX[relation]] * count)
        axis.scatter(
            sample_embedding[offset : offset + count, 0],
            sample_embedding[offset : offset + count, 1],
            s=14,
            alpha=0.55,
            color=color,
            label=f"{relation} (n={count})",
        )
        offset += count
    for color, relation in zip(colors, RELATIONS):
        index = RELATION_INDEX[relation]
        axis.scatter(
            mean_embedding[index, 0],
            mean_embedding[index, 1],
            s=260,
            marker="X",
            edgecolor="black",
            linewidth=1.2,
            color=color,
            zorder=5,
            label=f"mean {relation}",
        )
    axis.set_title(
        "Continuation directions by structural relation (UMAP, cosine)" + title_suffix,
        fontsize=12,
    )
    axis.set_xlabel("UMAP 1", fontsize=11)
    axis.set_ylabel("UMAP 2", fontsize=11)
    axis.legend(loc="best", fontsize=9, frameon=True)
    axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return {"plot": str(output_path), "projected_points": int(combined.shape[0])}


def print_geometry(name: str, geometry: Dict[str, Any]) -> None:
    """One report block per encoder scored."""
    print(f"\n  {name}")
    print("    within-relation direction consistency (mean cos(d, mean_dir))")
    for relation in RELATIONS:
        print(
            f"      {relation:<12} in-sample {geometry['within_relation_in_sample'][relation]:+.4f}"
            f" | held-out {geometry['within_relation_held_out'][relation]:+.4f}"
        )
    print("    cross-relation cosine between mean directions (target < 0.3)")
    for key, value in geometry["mean_direction_cosines"].items():
        print(f"      {key:<26} {value:+.4f}")
    print(
        f"    pairwise same-relation {geometry['pairwise_same_relation']:+.4f} vs "
        f"different-relation {geometry['pairwise_different_relation']:+.4f} -> separation "
        f"{geometry['relation_separation']:+.4f}"
    )



def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    print("=" * 94)
    print("STEP 3 - directional geometry: do the continuation directions separate relations?")
    print("=" * 94)

    pairs = load_pairs(args.pairs)
    labels = np.asarray([RELATION_INDEX[pair["relation"]] for pair in pairs], dtype=np.int64)
    tokenizer = BPETokenizer.load(args.tokenizer)
    ids1, ids2, lengths = tokenize_pairs(
        pairs, tokenizer, args.token_cache, args.max_seq_len
    )
    indices, chosen = sample_indices(labels, args.pairs_per_relation, args.seed)
    print(
        f"\n  pairs     : {len(pairs):,} labeled pairs | sample: "
        + ", ".join(f"{relation} {chosen[relation].size}" for relation in RELATIONS)
        + f" = {indices.size} directions"
    )

    model, payload = load_encoder(args.checkpoint, device)
    training = payload.get("training", {})
    print(
        f"  encoder   : {args.checkpoint.name} | {payload.get('parameter_count', model.count_parameters()):,} "
        f"params | {model.config['n_layers']} layers, hidden {model.config['hidden_dim']}"
    )
    if training:
        print(
            f"  training  : {training.get('steps', 0):,} steps | mean loss "
            f"{training.get('loss', float('nan')):.4f} | margin {training.get('margin')} | "
            f"cos(pos) {training.get('mean_cos_positive', float('nan')):+.3f} vs cos(neg) "
            f"{training.get('mean_cos_negative', float('nan')):+.3f}"
        )

    models: List[Tuple[str, GMNStyle]] = [("directional encoder (trained, STEP 2)", model)]
    if args.baseline:
        torch.manual_seed(args.seed)
        models.append(("untrained GMNStyle(small) - reference", GMNStyle(config_dict=config.MODEL_SMALL)))

    results: Dict[str, Any] = {}
    embeddings: Dict[str, Dict[str, Any]] = {}
    for name, candidate in models:
        candidate.eval()
        directions = directions_for(candidate, ids1, ids2, lengths, indices)
        geometry = relation_geometry(directions, chosen, args.pairwise_samples, rng)
        means = {}
        offset = 0
        for relation in RELATIONS:
            count = chosen[relation].size
            block = directions[offset : offset + count]
            offset += count
            mean = block.mean(axis=0)
            means[relation] = mean / max(1e-12, float(np.linalg.norm(mean)))
        results[name] = geometry
        embeddings[name] = {"directions": directions, "means": means}
        print_geometry(name, geometry)

    trained = results["directional encoder (trained, STEP 2)"]
    consistency_held_out = float(np.mean(list(trained["within_relation_held_out"].values())))
    consistency_in_sample = float(np.mean(list(trained["within_relation_in_sample"].values())))
    validated = consistency_held_out > VALIDATED_CONSISTENCY and trained[
        "max_cross_relation_cosine"
    ] < VALIDATED_CROSS
    weak = consistency_held_out < WEAK_CONSISTENCY

    print("\n  VERDICT")
    print(
        f"    within-relation consistency: in-sample {consistency_in_sample:+.4f} | "
        f"held-out {consistency_held_out:+.4f} (target > {VALIDATED_CONSISTENCY})"
    )
    print(
        f"    cross-relation cosine (max |.|): {trained['max_cross_relation_cosine']:+.4f} "
        f"(target < {VALIDATED_CROSS})"
    )
    print(f"    relation separation (pairwise): {trained['relation_separation']:+.4f}")
    verdict = (
        "GEOMETRY VALIDATED - proceed to the TASK 29 prediction loop with this encoder"
        if validated
        else "SIGNAL TOO WEAK - try an RST discourse dataset (report and wait for a decision)"
        if weak
        else "BORDERLINE - report and wait for a decision"
    )
    print(f"    {verdict}")

    plot_info: Dict[str, Any] = {}
    if not args.no_plot:
        plot_info = plot_geometry(
            embeddings["directional encoder (trained, STEP 2)"]["directions"],
            chosen,
            embeddings["directional encoder (trained, STEP 2)"]["means"],
            args.plot,
            "",
        )
        print(f"    plot saved: {plot_info['plot']}")

    payload_out = {
        "task": "STEP 3 - directional geometry verification",
        "config": {
            "checkpoint": str(args.checkpoint),
            "pairs": str(args.pairs),
            "pairs_per_relation": args.pairs_per_relation,
            "pairwise_samples": args.pairwise_samples,
            "seed": args.seed,
        },
        "encoder_training": training,
        "results": results,
        "summary": {
            "consistency_in_sample": consistency_in_sample,
            "consistency_held_out": consistency_held_out,
            "max_cross_relation_cosine": trained["max_cross_relation_cosine"],
            "relation_separation": trained["relation_separation"],
            "validated": bool(validated),
            "weak": bool(weak),
            "verdict": verdict,
        },
        "plot": plot_info,
        "wall_seconds": time.perf_counter() - started,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        json.dump(payload_out, handle, indent=2)
    print(f"\n  saved {args.results}")
    print("=" * 94)
    return payload_out


if __name__ == "__main__":
    main()

