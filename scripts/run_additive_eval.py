"""STEP 2: lambda sweep for ADDITIVE scoring on all 10,042 validation examples.

Scoring is now additive - the query position is never moved:

    scores = cos(ending, ctx) + lambda_nav * cos(ending, navigation_vector)

Because the tangent navigation vector is orthogonal to the context,
``cos(ending, L2(ctx + nav)) = (cos(ending,ctx) + cos(ending,nav)) / sqrt(2)``,
so the old blend ranked identically but diluted the context term by sqrt(2).
Removing that factor should recover accuracy.

``lambda_nav = 0.0`` must reproduce the no_navigation baseline exactly.

Writes ``experiments/results_additive.json`` and caches the per-example
context/navigation score matrices to ``experiments/additive_scores.npz``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import precompute_val_full_vectors as vecache  # noqa: E402
from run_ablations import paired_comparison  # noqa: E402
from src.fusion.fuse import fuse_directions  # noqa: E402
from src.memory.store import MemoryStore  # noqa: E402
from src.sampler.jev import parallel_sample  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
V2_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full_v2.pkl"
CHORD_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_full_val.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_additive.json"
SCORES_PATH = PROJECT_ROOT / "experiments" / "additive_scores.npz"

LAMBDAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEED = 42
K = 10
TOP_K = 3
TARGET = "correct_ending"


def rename_pairs(stats: Dict[str, Any], first: str, second: str) -> Dict[str, Any]:
    """Re-key ``paired_comparison`` output so ``first``/``second`` are explicit."""
    return {
        "first": first,
        "second": second,
        f"{first}_correct_{second}_wrong": stats["full_correct_ablation_wrong"],
        f"{first}_wrong_{second}_correct": stats["full_wrong_ablation_correct"],
        "discordant_pairs": stats["discordant_pairs"],
        "mcnemar_p_value": stats["mcnemar_p_value"],
        "significant_at_0.05": stats["significant_at_0.05"],
    }


def precompute_score_terms(
    val_data: List[Dict[str, Any]],
    queries: np.ndarray,
    endings: np.ndarray,
    store: MemoryStore,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-example ``(context_scores, navigation_scores)``, each ``(N, 4)``.

    Computed once so the whole lambda sweep is just ``argmax(ctx + lam*nav)`` -
    every lambda therefore sees exactly the same retrieval and navigation vector.
    """
    total = len(val_data)
    context_scores = np.zeros((total, 4), dtype=np.float64)
    navigation_scores = np.zeros((total, 4), dtype=np.float64)

    with contextlib.redirect_stdout(io.StringIO()):
        for index in range(total):
            query_vector = queries[index].astype(np.float64)
            ending_vectors = endings[index].astype(np.float64)

            memories = store.query(query_vector, k=K)
            weighted = parallel_sample(
                query_vector, memories, [1.0] * len(memories), target=TARGET
            )
            navigation_vector = fuse_directions(
                weighted, top_k=TOP_K, use_tangent=True, query_vector=query_vector
            )

            context_scores[index] = ending_vectors @ query_vector
            navigation_scores[index] = ending_vectors @ navigation_vector
    return context_scores, navigation_scores


def discriminative_gap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean score of the correct ending minus the mean of the incorrect ones."""
    rows = np.arange(len(labels))
    correct = scores[rows, labels]
    mask = np.ones_like(scores, dtype=bool)
    mask[rows, labels] = False
    return float(correct.mean() - scores[mask].mean())


def main() -> None:
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)
    total = len(val_data)

    store = MemoryStore.load(V2_STORE_PATH)
    queries, endings = vecache.load_all_chunks(total)
    with GMN_PRED_PATH.open(encoding="utf-8") as handle:
        cached_gmn = json.load(handle)
    with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
        gpt2 = json.load(handle)
    with CHORD_RESULTS_PATH.open(encoding="utf-8") as handle:
        chord_results = json.load(handle)

    labels = np.asarray(cached_gmn["no_navigation"]["labels"])
    no_navigation_predictions = cached_gmn["no_navigation"]["predictions"]
    no_navigation_accuracy = cached_gmn["no_navigation"]["accuracy"]
    gpt2_predictions = gpt2["predictions"]
    gpt2_accuracy = chord_results["gpt2_small"]["accuracy"]

    print(f"val examples {total} | v2 store {store.size} | target={TARGET} | "
          f"use_tangent=True | k={K} | top_k={TOP_K}")
    print(f"cached: no_navigation {100 * no_navigation_accuracy:.2f}% | "
          f"GPT-2 small {100 * gpt2_accuracy:.2f}%")

    print("\nprecomputing context and navigation score terms (once, for all lambdas) ...")
    started = time.perf_counter()
    context_scores, navigation_scores = precompute_score_terms(
        val_data, queries, endings, store
    )
    precompute_seconds = time.perf_counter() - started
    print(f"  done in {precompute_seconds:.1f}s")
    np.savez(
        SCORES_PATH,
        context_scores=context_scores,
        navigation_scores=navigation_scores,
        labels=labels,
    )
    print(f"  score matrices cached to {SCORES_PATH}")

    gap_context = discriminative_gap(context_scores, labels)
    gap_navigation = discriminative_gap(navigation_scores, labels)
    print(f"\n  gap(cos ctx) = {gap_context:+.4f} | gap(cos nav) = {gap_navigation:+.4f}")
    print(f"  => predicted gap(lambda) = {gap_context:+.4f} + lambda * "
          f"{gap_navigation:+.4f}")

    # ------------------------------------------------------------- sweep
    print("\n" + "=" * 76)
    print("LAMBDA SWEEP (additive scoring, all 10,042 val examples)")
    print("=" * 76)
    header = (f"  {'lambda':>6} | {'Accuracy':>9} | {'vs no_nav':>10} | "
              f"{'vs GPT-2':>9} | {'gap':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: List[Dict[str, Any]] = []
    for lam in LAMBDAS:
        scores = context_scores + lam * navigation_scores
        predictions = np.argmax(scores, axis=1)
        accuracy = float((predictions == labels).mean())
        gap = discriminative_gap(scores, labels)
        rows.append(
            {
                "lambda_nav": lam,
                "accuracy": accuracy,
                "correct": int((predictions == labels).sum()),
                "delta_vs_no_navigation": accuracy - no_navigation_accuracy,
                "delta_vs_gpt2": accuracy - gpt2_accuracy,
                "discriminative_gap": gap,
                "predictions": [int(value) for value in predictions],
            }
        )
        print(f"  {lam:>6.2f} | {100 * accuracy:>8.2f}% | "
              f"{100 * (accuracy - no_navigation_accuracy):>+9.2f} | "
              f"{100 * (accuracy - gpt2_accuracy):>+8.2f} | {gap:>+8.4f}")
    print("  " + "-" * (len(header) - 2))

    # ------------------------------------ lambda=0.0 must equal no_navigation
    zero_row = next(row for row in rows if row["lambda_nav"] == 0.0)
    differing = sum(
        1
        for a, b in zip(zero_row["predictions"], no_navigation_predictions)
        if a != b
    )
    bit_identical = differing == 0 and zero_row["accuracy"] == no_navigation_accuracy
    print("\n  lambda=0.00 vs no_navigation baseline:")
    print(f"    accuracy {100 * zero_row['accuracy']:.4f}% vs cached "
          f"{100 * no_navigation_accuracy:.4f}% "
          f"-> equal: {zero_row['accuracy'] == no_navigation_accuracy}")
    print(f"    predictions differ on {differing}/{total} examples")
    print(f"    bit-identical to no_navigation: {bit_identical}")

    # --------------------------------------------------------- best lambda
    best = max(rows, key=lambda row: row["accuracy"])
    best_lambda = best["lambda_nav"]
    print(f"\n  BEST lambda_nav = {best_lambda:.2f} -> {100 * best['accuracy']:.2f}% "
          f"({best['correct']}/{total})")
    print(f"    vs no_navigation    : {100 * best['delta_vs_no_navigation']:+.2f} points")
    print(f"    vs GPT-2 small      : {100 * best['delta_vs_gpt2']:+.2f} points")
    print(f"    discriminative gap  : {best['discriminative_gap']:+.4f}")
    print(f"      context-only {gap_context:+.4f} | additive sum "
          f"{gap_context + gap_navigation:+.4f} | old sqrt2-blend "
          f"{(gap_context + gap_navigation) / np.sqrt(2.0):+.4f}")

    # -------------------------------------------------------- McNemar tests
    comparisons: Dict[str, Dict[str, Any]] = {
        "best_lambda_vs_no_navigation": rename_pairs(
            paired_comparison(
                best["predictions"], no_navigation_predictions, labels.tolist()
            ),
            f"additive_lambda_{best_lambda:g}",
            "no_navigation",
        ),
        "best_lambda_vs_gpt2_small": rename_pairs(
            paired_comparison(best["predictions"], gpt2_predictions, labels.tolist()),
            f"additive_lambda_{best_lambda:g}",
            "gpt2_small",
        ),
    }

    print("\n" + "=" * 76)
    print(f"McNEMAR EXACT TESTS (best lambda = {best_lambda:.2f}, all {total} examples)")
    print("=" * 76)
    for name, stats in comparisons.items():
        a_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        b_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(f"  {name}")
        print(f"    {a_key:<52} {stats[a_key]:>6}")
        print(f"    {b_key:<52} {stats[b_key]:>6}")
        print(f"    discordant {stats['discordant_pairs']:>5} | "
              f"p = {stats['mcnemar_p_value']:.4g} | significant: "
              f"{stats['significant_at_0.05']}")

    # -------------------------------------------------------------- save
    beats_no_navigation = bool(
        comparisons["best_lambda_vs_no_navigation"]["significant_at_0.05"]
        and best["accuracy"] > no_navigation_accuracy
    )
    beats_gpt2 = bool(
        comparisons["best_lambda_vs_gpt2_small"]["significant_at_0.05"]
        and best["accuracy"] > gpt2_accuracy
    )
    print(f"\n  best lambda beats no_navigation (significant): {beats_no_navigation}")
    print(f"  best lambda beats GPT-2 small   (significant): {beats_gpt2}")

    result = {
        "experiment": "additive scoring + lambda sweep",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": total,
        "seed": SEED,
        "config": {
            "store": "checkpoints/memory_store_full_v2.pkl",
            "store_size": store.size,
            "target": TARGET,
            "use_tangent": True,
            "retrieval_k": K,
            "fuse_top_k": TOP_K,
            "scoring": "cos(ending, ctx) + lambda_nav * cos(ending, nav_vector)",
            "position_moved": False,
        },
        "sweep": [{k: v for k, v in row.items() if k != "predictions"} for row in rows],
        "gap_terms": {
            "gap_context": gap_context,
            "gap_navigation": gap_navigation,
            "gap_additive_sum_lambda1": gap_context + gap_navigation,
            "gap_old_sqrt2_blend_lambda1": (gap_context + gap_navigation) / np.sqrt(2.0),
        },
        "best": {
            "lambda_nav": best_lambda,
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "delta_vs_no_navigation": best["delta_vs_no_navigation"],
            "delta_vs_gpt2": best["delta_vs_gpt2"],
            "discriminative_gap": best["discriminative_gap"],
        },
        "lambda_zero_equals_no_navigation": {
            "accuracy_equal": bool(zero_row["accuracy"] == no_navigation_accuracy),
            "predictions_differ": differing,
            "bit_identical": bool(bit_identical),
        },
        "mcnemar": comparisons,
        "best_lambda_beats_no_navigation": beats_no_navigation,
        "best_lambda_beats_gpt2_small": beats_gpt2,
        "reference_accuracies": {
            "no_navigation": no_navigation_accuracy,
            "gpt2_small": gpt2_accuracy,
            "chord_ctx_target": chord_results["conditions"]["full_model"]["accuracy"],
            "tangent_ctx_target": 0.2897,
            "continuation_sqrt2_blend": 0.2939,
        },
        "precompute_seconds": precompute_seconds,
        "scores_cache": "experiments/additive_scores.npz",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("Additive evaluation complete.")


if __name__ == "__main__":
    main()
