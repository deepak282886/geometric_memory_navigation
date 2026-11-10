"""TASK 29 follow-up: WHY does the memory hurt - relevance or the fusion?

The loop measured the effect (ARM A - ARM B = -0.0057 at step 10,000, paired t -4.52)
but not its cause. TASK 27 showed the mechanism is neighbour RELEVANCE: a memory of
topically unrelated experiences pushes the prediction away from the truth (measured
there as -0.0066..-0.0054 for unrelated prefix memories, against +0.0081 for a
relevant one). This script checks that explanation for TASK 29's loop directly, on
the final memory (all 10,000 experiences, regions fitted):

    for in-corpus queries (train pairs, whose neighbours ARE in the memory)
    and out-of-corpus queries (the held-out eval pairs, whose documents are not),

    neighbour relevance = mean cos(P_current, retrieved neighbour)
    context alignment   = cos(C, true tangent direction to P_actual)
    prediction delta    = cos(ARM A, actual) - cos(ARM B, actual)

If the explanation holds, out-of-corpus queries should show LOWER neighbour
relevance and NEGATIVE alignment, and alignment should correlate with the delta
(a context pointing the right way helps; one pointing away hurts).

Nothing is trained; the memory is rebuilt from the same cached latents with the same
config as ``run_prediction_loop.py``, so its neighbour statistics match that run.

Artifacts
---------
    experiments/loop_relevance_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_prediction_loop import (  # noqa: E402
    DEFAULT_PREDICTOR,
    DEFAULT_VAE,
    FUSE_TOP_K,
    KNN_K,
    PROBE_REGIONS,
    SCORE_MODE,
    TRAIN_PAIRS_PATH,
    VAL_PAIRS_PATH,
    encode_sentences_cached,
    fuse_context,
    load_pairs,
    load_vae,
    predict,
    stream_pairs,
    unique_sentences,
)
from src.memory.region_store import RegionStore  # noqa: E402
from src.predictor.latent_predictor import LATENT_DIM, load_latent_predictor  # noqa: E402

TRAIN_VEC_CACHE = PROJECT_ROOT / "checkpoints" / "prediction_loop_train_vecs.npz"
VAL_VEC_CACHE = PROJECT_ROOT / "checkpoints" / "wikitext_val_predictor_vecs_continuation.npz"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "loop_relevance_diagnosis.json"

STEPS = 10000
EVAL_START = 7166
EVAL_SIZE = 500
TRAIN_SAMPLE = 500
NUM_REGIONS = 256
MAX_SIZE = 50000
AUTO_FIT_AFTER = 2048
OUTCOME_FLOOR = 0.1
EVAL_SOURCE_BASE = 1000000
SEED = 42


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Why does memory hurt? Neighbour relevance vs context alignment."
    )
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--train-pairs", type=Path, default=TRAIN_PAIRS_PATH)
    parser.add_argument("--val-pairs", type=Path, default=VAL_PAIRS_PATH)
    parser.add_argument("--train-cache", type=Path, default=TRAIN_VEC_CACHE)
    parser.add_argument("--val-cache", type=Path, default=VAL_VEC_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def tangent_direction(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unit tangent direction from ``query`` toward ``target`` (radial part removed).

    This is the direction the fusion code is trying to approximate, so comparing the
    context against it measures whether the memory pointed the right way.
    """
    direction = target.astype(np.float64) - query.astype(np.float64)
    projected = direction - float(direction @ query) * query
    norm = float(np.linalg.norm(projected))
    return projected / norm if norm > 1e-12 else np.zeros_like(projected)


def describe(name: str, values: np.ndarray) -> Dict[str, Any]:
    return {
        "name": name,
        "num_queries": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "share_positive": float((values > 0).mean()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def query_group(
    memory: RegionStore,
    predictor,
    current: np.ndarray,
    actual: np.ndarray,
    sources: np.ndarray,
    probe_regions: int,
) -> Dict[str, Any]:
    """Per-query neighbour relevance, context alignment and prediction delta."""
    neighbour_top1: List[float] = []
    neighbour_top10: List[float] = []
    alignment: List[float] = []
    contexts = np.zeros((len(current), LATENT_DIM), dtype=np.float32)

    for index in range(len(current)):
        query = current[index]
        records = memory.query(
            query, k=KNN_K, probe_regions=probe_regions, exclude_source=int(sources[index])
        )
        similarities = [float(record["similarity"]) for record in records]
        neighbour_top1.append(similarities[0] if similarities else float("nan"))
        neighbour_top10.append(float(np.mean(similarities)) if similarities else float("nan"))
        context = fuse_context(query, records, probe_regions=probe_regions)
        contexts[index] = context
        alignment.append(float(context.astype(np.float64) @ tangent_direction(query, actual[index])))

    prediction_a = predict(predictor, current, contexts)
    prediction_b = predict(predictor, current, np.zeros_like(contexts))
    cos_a = np.einsum("nd,nd->n", prediction_a.astype(np.float64), actual.astype(np.float64))
    cos_b = np.einsum("nd,nd->n", prediction_b.astype(np.float64), actual.astype(np.float64))
    delta = cos_a - cos_b

    return {
        "neighbour_top1_similarity": describe("top-1 neighbour cosine", np.asarray(neighbour_top1)),
        "neighbour_top10_similarity": describe(
            "top-10 neighbour cosine", np.asarray(neighbour_top10)
        ),
        "context_alignment": describe("cos(C, true tangent direction)", np.asarray(alignment)),
        "prediction_delta": describe("ARM A - ARM B cosine", delta),
        "_arrays": {
            "alignment": np.asarray(alignment),
            "delta": delta,
            "neighbour_top10": np.asarray(neighbour_top10),
        },
    }



def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("TASK 29 DIAGNOSIS - is the damage from irrelevant neighbours, or from fusion?")
    print("=" * 96)

    train_pairs = stream_pairs(args.train_pairs, args.steps)
    val_pairs = load_pairs(args.val_pairs)
    eval_pairs = val_pairs[EVAL_START : EVAL_START + args.eval_size]

    vae = load_vae(args.vae, device)
    train_sentences, train_index = unique_sentences(train_pairs)
    train_latents = encode_sentences_cached(vae, train_sentences, args.train_cache)
    train_current = np.ascontiguousarray(
        train_latents[[train_index[first] for first, _ in train_pairs]]
    )
    train_actual = np.ascontiguousarray(
        train_latents[[train_index[second] for _, second in train_pairs]]
    )
    val_sentences, val_index = unique_sentences(val_pairs)
    val_latents = encode_sentences_cached(vae, val_sentences, args.val_cache)
    eval_current = np.ascontiguousarray(val_latents[[val_index[a] for a, _ in eval_pairs]])
    eval_actual = np.ascontiguousarray(val_latents[[val_index[b] for _, b in eval_pairs]])

    predictor = load_latent_predictor(args.predictor, device)
    predictor.eval()

    print(
        f"\nrebuilding the loop's final memory ({len(train_pairs)} experiences, "
        f"{NUM_REGIONS} regions, same config as run_prediction_loop.py) ..."
    )
    memory = RegionStore(
        num_regions=NUM_REGIONS,
        latent_dim=LATENT_DIM,
        max_size=MAX_SIZE,
        auto_fit_after=AUTO_FIT_AFTER,
        score_mode=SCORE_MODE,
        seed=SEED,
    )
    for step in range(len(train_pairs)):
        memory.add(
            train_current[step],
            max(OUTCOME_FLOOR, 0.5),
            target_vector=train_actual[step],
            source_index=step,
        )
    print(f"  memory: size {memory.size} | regions {memory.num_regions} | fitted {memory.is_fitted}")

    sample = np.sort(
        rng.choice(len(train_pairs), size=min(args.train_sample, len(train_pairs)), replace=False)
    )
    train_group = query_group(
        memory, predictor, train_current[sample], train_actual[sample], sample, PROBE_REGIONS
    )
    eval_sources = np.arange(EVAL_SOURCE_BASE, EVAL_SOURCE_BASE + len(eval_pairs), dtype=np.int64)
    eval_group = query_group(
        memory, predictor, eval_current, eval_actual, eval_sources, PROBE_REGIONS
    )

    print("\n  group                 | top-1 nb | top-10 nb | C alignment | ARM A - ARM B")
    print("  " + "-" * 78)
    for label, group in (("in-corpus (train)", train_group), ("out-of-corpus (eval)", eval_group)):
        print(
            f"  {label:<21} | {group['neighbour_top1_similarity']['mean']:>8.4f} | "
            f"{group['neighbour_top10_similarity']['mean']:>9.4f} | "
            f"{group['context_alignment']['mean']:>+11.4f} | "
            f"{group['prediction_delta']['mean']:>+13.4f}"
        )

    pooled_alignment = np.concatenate(
        [train_group["_arrays"]["alignment"], eval_group["_arrays"]["alignment"]]
    )
    pooled_delta = np.concatenate(
        [train_group["_arrays"]["delta"], eval_group["_arrays"]["delta"]]
    )
    pooled_neighbours = np.concatenate(
        [train_group["_arrays"]["neighbour_top10"], eval_group["_arrays"]["neighbour_top10"]]
    )
    correlation_alignment = float(np.corrcoef(pooled_alignment, pooled_delta)[0, 1])
    correlation_neighbours = float(np.corrcoef(pooled_neighbours, pooled_delta)[0, 1])

    print(f"\n  mechanism checks, all queried pairs pooled (n = {pooled_alignment.size})")
    print(
        f"    corr(context alignment, delta)   : {correlation_alignment:+.3f} "
        f"(positive -> a context that points the right way helps)"
    )
    print(
        f"    corr(neighbour relevance, delta) : {correlation_neighbours:+.3f} "
        f"(positive -> more relevant neighbours help)"
    )
    print(
        f"    neighbour relevance: eval "
        f"{eval_group['neighbour_top10_similarity']['mean']:.4f} vs train "
        f"{train_group['neighbour_top10_similarity']['mean']:.4f}"
    )

    payload = {
        "task": "TASK 29 diagnosis - neighbour relevance vs context alignment",
        "config": {
            "steps": len(train_pairs),
            "num_regions": NUM_REGIONS,
            "probe_regions": PROBE_REGIONS,
            "knn_k": KNN_K,
            "fuse_top_k": FUSE_TOP_K,
            "train_sample": int(sample.size),
            "eval_size": len(eval_pairs),
            "seed": args.seed,
        },
        "memory": memory.summary(),
        "in_corpus": {key: value for key, value in train_group.items() if key != "_arrays"},
        "out_of_corpus": {key: value for key, value in eval_group.items() if key != "_arrays"},
        "mechanism": {
            "corr_alignment_delta": correlation_alignment,
            "corr_neighbour_relevance_delta": correlation_neighbours,
            "pooled_queries": int(pooled_alignment.size),
            "interpretation": (
                "alignment is cos(fused context, true tangent direction to P_actual): a "
                "positive mean means the memory pointed the right way, a negative mean "
                "means it pointed away from the answer"
            ),
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved {args.results}")
    print("=" * 96)
    return payload


if __name__ == "__main__":
    main()

