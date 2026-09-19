"""STEP 4: does targeting correct CONTINUATIONS fix navigation?

Compares, on all 10,042 validation examples and the v2 (39,905-example) store:

    no_navigation                  31.31%  cached
    chord navigation (ctx target)  27.75%  cached
    tangent          (ctx target)  28.97%  recomputed here (needed for McNemar)
    continuation     (ce  target)  <- computed here
    GPT-2 small                    28.58%  cached

Navigation targets each memory's KNOWN-CORRECT continuation
(``correct_ending_vector``) instead of the neighbouring context, with the radial
component projected out (``use_tangent=True``) since the chord version is known
to hurt.

Also repeats the 200-example diagnosis with the continuation target and checks
each predicted symptom.

Writes ``experiments/results_continuation.json`` and
``experiments/navigation_diagnosis_continuation.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import precompute_val_full_vectors as vecache  # noqa: E402
from diagnose_navigation import run_diagnosis  # noqa: E402
from run_ablations import paired_comparison, run_ablation  # noqa: E402
from src.memory.store import MemoryStore  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
V2_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full_v2.pkl"
CHORD_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_full_val.json"
TANGENT_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_tangent.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_continuation.json"
CONTINUATION_DIAG_PATH = (
    PROJECT_ROOT / "experiments" / "navigation_diagnosis_continuation.json"
)

SEED = 42
GPT2_PARAMETERS = 124_439_808

#: tangent-target reference values the continuation target should improve on
TANGENT_DIAGNOSIS = {
    "mean_score_correct_navigated": 0.5501,
    "score_gap_navigated": 0.0078,
    "improved": 51,
    "hurt": 55,
    "mean_cos_ctx_nav": 0.0,
}


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


def main() -> None:
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)
    total = len(val_data)

    store = MemoryStore.load(V2_STORE_PATH)
    if not store.has_correct_ending_vectors:
        raise RuntimeError(f"{V2_STORE_PATH.name} has no correct_ending_vectors")
    queries, endings = vecache.load_all_chunks(total)
    print(f"val examples {total} | v2 store {store.size} | "
          f"correct_ending_vectors {store.correct_ending_vectors.shape} | "
          f"queries {queries.shape} | endings {endings.shape}")

    # ------------------------------------------- recompute tangent (for McNemar)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    tangent = run_ablation(val_data, queries, endings, store, "full_tangent", rng)
    tangent_seconds = time.perf_counter() - started
    with TANGENT_RESULTS_PATH.open(encoding="utf-8") as handle:
        cached_tangent = json.load(handle)["conditions"]["full_model_tangent"]["accuracy"]
    print(f"  [tangent, ctx target]      {100 * tangent['accuracy']:.2f}% "
          f"({tangent['correct']}/{total}) in {tangent_seconds:.1f}s "
          f"-> reproduces cached {100 * cached_tangent:.2f}%: "
          f"{abs(tangent['accuracy'] - cached_tangent) < 1e-12}")

    # ------------------------------------------- the new continuation target
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    continuation = run_ablation(
        val_data, queries, endings, store, "full_continuation", rng
    )
    continuation_seconds = time.perf_counter() - started
    print(f"  [continuation, ce target]  {100 * continuation['accuracy']:.2f}% "
          f"({continuation['correct']}/{total}) in {continuation_seconds:.1f}s")

    # ---------------------------------------------------------- cached baselines
    with GMN_PRED_PATH.open(encoding="utf-8") as handle:
        cached_gmn = json.load(handle)
    with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
        gpt2 = json.load(handle)
    with CHORD_RESULTS_PATH.open(encoding="utf-8") as handle:
        chord_results = json.load(handle)

    labels = cached_gmn["full_model"]["labels"]
    no_navigation = cached_gmn["no_navigation"]
    chord = cached_gmn["full_model"]
    gpt2_accuracy = chord_results["gpt2_small"]["accuracy"]

    for name, outcome in (("tangent", tangent), ("continuation", continuation)):
        if outcome["labels"] != labels:
            raise RuntimeError(f"{name} label array disagrees with the cached runs")

    # ------------------------------------------------------------- table
    print("\n" + "=" * 70)
    print("CONTINUATION TARGET vs EVERYTHING ELSE (all 10,042 val examples)")
    print("=" * 70)
    print(f"  {'Condition':<30} | {'Accuracy':>9} | source")
    print("  " + "-" * 62)
    table = [
        ("no_navigation", no_navigation["accuracy"], "cached"),
        ("chord navigation (ctx target)", chord["accuracy"], "cached"),
        ("tangent navigation (ctx target)", tangent["accuracy"], "recomputed"),
        ("continuation navigation (ce target)", continuation["accuracy"], "computed now"),
        ("GPT-2 small", gpt2_accuracy, "cached"),
    ]
    for name, accuracy, source in table:
        print(f"  {name:<30} | {100 * accuracy:>8.2f}% | {source}")
    print("  " + "-" * 62)
    for label, other in (
        ("no_navigation", no_navigation["accuracy"]),
        ("tangent (ctx target)", tangent["accuracy"]),
        ("chord  (ctx target)", chord["accuracy"]),
        ("GPT-2 small", gpt2_accuracy),
    ):
        print(f"  continuation - {label:<20}: "
              f"{100 * (continuation['accuracy'] - other):+.2f} points")

    # ---------------------------------------------------- McNemar tests
    continuation_predictions = continuation["predictions"]
    comparisons: Dict[str, Dict[str, Any]] = {
        "continuation_vs_no_navigation": rename_pairs(
            paired_comparison(
                continuation_predictions, no_navigation["predictions"], labels
            ),
            "continuation",
            "no_navigation",
        ),
        "continuation_vs_tangent": rename_pairs(
            paired_comparison(continuation_predictions, tangent["predictions"], labels),
            "continuation",
            "tangent",
        ),
        "continuation_vs_gpt2_small": rename_pairs(
            paired_comparison(continuation_predictions, gpt2["predictions"], labels),
            "continuation",
            "gpt2_small",
        ),
    }

    print("\n" + "=" * 70)
    print("McNEMAR EXACT TESTS (paired, all 10,042 examples)")
    print("=" * 70)
    for name, stats in comparisons.items():
        a_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        b_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(f"  {name}")
        print(f"    {a_key:<50} {stats[a_key]:>6}")
        print(f"    {b_key:<50} {stats[b_key]:>6}")
        print(f"    discordant {stats['discordant_pairs']:>5} | "
              f"p = {stats['mcnemar_p_value']:.4g} | significant: "
              f"{stats['significant_at_0.05']}")

    def beats(key: str, other_accuracy: float) -> bool:
        stats = comparisons[key]
        return bool(
            stats["significant_at_0.05"] and continuation["accuracy"] > other_accuracy
        )

    beats_no_navigation = beats(
        "continuation_vs_no_navigation", no_navigation["accuracy"]
    )
    beats_tangent = beats("continuation_vs_tangent", tangent["accuracy"])
    beats_gpt2 = beats("continuation_vs_gpt2_small", gpt2_accuracy)

    # ------------------------------------- diagnosis with the continuation target
    print("\n" + "=" * 70)
    print("DIAGNOSIS WITH target='correct_ending' (200 random examples)")
    print("=" * 70)
    diagnosis = run_diagnosis(use_tangent=True, target="correct_ending")
    CONTINUATION_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONTINUATION_DIAG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(diagnosis, handle, indent=2)

    summary = diagnosis["summary"]
    expectations = {
        "cos_ctx_nav_positive": {
            "value": summary["mean_cos_ctx_nav"],
            "target": "> 0.0  (tangent was exactly 0.0)",
            "passed": summary["mean_cos_ctx_nav"] > 0.0,
            "note": (
                "NOT ACHIEVABLE with use_tangent=True: the tangent projection is "
                "precisely the operation that zeroes this cosine. A positive value "
                "would require keeping a radial component, i.e. undoing the fix."
            ),
        },
        "mean_score_correct_beats_tangent": {
            "value": summary["mean_score_correct_navigated"],
            "target": f"> {TANGENT_DIAGNOSIS['mean_score_correct_navigated']}",
            "passed": summary["mean_score_correct_navigated"]
            > TANGENT_DIAGNOSIS["mean_score_correct_navigated"],
        },
        "score_gap_beats_tangent": {
            "value": summary["score_gap_navigated"],
            "target": f"> {TANGENT_DIAGNOSIS['score_gap_navigated']}",
            "passed": summary["score_gap_navigated"]
            > TANGENT_DIAGNOSIS["score_gap_navigated"],
        },
        "improved_beats_tangent": {
            "value": summary["improved"],
            "target": f"> {TANGENT_DIAGNOSIS['improved']}",
            "passed": summary["improved"] > TANGENT_DIAGNOSIS["improved"],
        },
        "hurt_below_tangent": {
            "value": summary["hurt"],
            "target": f"< {TANGENT_DIAGNOSIS['hurt']}",
            "passed": summary["hurt"] < TANGENT_DIAGNOSIS["hurt"],
        },
    }

    print("\n" + "=" * 70)
    print("EXPECTED SYMPTOMS OF THE CONTINUATION TARGET")
    print("=" * 70)
    for name, check in expectations.items():
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {name:<34} "
              f"value {check['value']}  target {check['target']}")
    print("\n  gap comparison: chord 0.0031 -> tangent 0.0078 -> continuation "
          f"{summary['score_gap_navigated']:.4f}")

    # -------------------------------------------------------------- save
    result = {
        "experiment": "continuation navigation (correct-ending target)",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": total,
        "seed": SEED,
        "config": {
            "store": "checkpoints/memory_store_full_v2.pkl",
            "store_size": store.size,
            "store_has_correct_ending_vectors": store.has_correct_ending_vectors,
            "retrieval_k": 10,
            "fuse_top_k": 3,
            "use_tangent": True,
            "target": "correct_ending",
            "scoring": (
                "cosine(L2(encode(ending)), L2(ctx + fuse_directions(..., "
                "use_tangent=True, query_vector=ctx)))"
            ),
        },
        "table": [
            {"condition": name, "accuracy": accuracy, "source": source}
            for name, accuracy, source in table
        ],
        "conditions": {
            "no_navigation": {
                "accuracy": no_navigation["accuracy"],
                "source": "cached",
            },
            "chord_navigation_ctx_target": {
                "accuracy": chord["accuracy"],
                "source": "cached",
            },
            "tangent_navigation_ctx_target": {
                "accuracy": tangent["accuracy"],
                "correct": tangent["correct"],
                "seconds": tangent_seconds,
                "source": "recomputed (matches results_tangent.json)",
            },
            "continuation_navigation_ce_target": {
                "accuracy": continuation["accuracy"],
                "correct": continuation["correct"],
                "num_examples": continuation["num_examples"],
                "seconds": continuation_seconds,
                "ms_per_example": continuation["ms_per_example"],
            },
            "gpt2_small": {
                "accuracy": gpt2_accuracy,
                "params": GPT2_PARAMETERS,
                "source": "cached",
            },
        },
        "mcnemar": comparisons,
        "continuation_beats_no_navigation": bool(beats_no_navigation),
        "continuation_beats_tangent": bool(beats_tangent),
        "continuation_beats_gpt2_small": bool(beats_gpt2),
        "diagnosis": {
            "file": "experiments/navigation_diagnosis_continuation.json",
            "summary": summary,
            "tangent_reference": TANGENT_DIAGNOSIS,
            "expectations": expectations,
            "expectations_passed": [
                name for name, check in expectations.items() if check["passed"]
            ],
            "expectations_failed": [
                name for name, check in expectations.items() if not check["passed"]
            ],
        },
        "gap_progression": {
            "no_navigation_ctx_only": 0.0195,
            "chord_ctx_target": 0.0031,
            "tangent_ctx_target": 0.0078,
            "continuation_target": summary["score_gap_navigated"],
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Diagnosis saved to {CONTINUATION_DIAG_PATH}")
    print("Continuation evaluation complete.")


if __name__ == "__main__":
    main()

