"""STEP 2: does removing the radial component fix navigation?

Compares, on all 10,042 validation examples and the full 39,905-example store:

    no_navigation        31.3%  cached (TASK "full validation" run)
    full_model (chord)   27.8%  cached
    full_model (tangent) <- computed here
    GPT-2 small          28.6%  cached

then runs three exact McNemar tests against the tangent variant, repeats the
200-example navigation diagnosis with ``use_tangent=True``, and checks each of the
predicted symptoms of the fix.

Writes ``experiments/results_tangent.json`` and
``experiments/navigation_diagnosis_tangent.json``.
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
FULL_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full.pkl"
CHORD_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_full_val.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_tangent.json"
TANGENT_DIAG_PATH = PROJECT_ROOT / "experiments" / "navigation_diagnosis_tangent.json"

SEED = 42
GPT2_PARAMETERS = 124_439_808

#: chord-diagnosis reference values the fix should improve on (TASK 9 STEP 2)
CHORD_DIAGNOSIS = {
    "mean_score_correct_navigated": 0.4519,
    "score_gap_navigated": 0.0031,
    "improved": 59,
    "hurt": 65,
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

    store = MemoryStore.load(FULL_STORE_PATH)
    queries, endings = vecache.load_all_chunks(total)
    print(f"val examples {total} | full store {store.size} | "
          f"queries {queries.shape} | endings {endings.shape}")

    # ------------------------------------------------ tangent full model
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    tangent = run_ablation(val_data, queries, endings, store, "full_tangent", rng)
    tangent_seconds = time.perf_counter() - started
    print(f"  [full_model (tangent)] {100 * tangent['accuracy']:.2f}% "
          f"({tangent['correct']}/{total}) in {tangent_seconds:.1f}s")

    # ------------------------------------------------ cached conditions
    with GMN_PRED_PATH.open(encoding="utf-8") as handle:
        cached_gmn = json.load(handle)
    with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
        cached_gpt2 = json.load(handle)
    with CHORD_RESULTS_PATH.open(encoding="utf-8") as handle:
        chord_results = json.load(handle)

    labels = cached_gmn["full_model"]["labels"]
    no_navigation = cached_gmn["no_navigation"]
    chord = cached_gmn["full_model"]
    gpt2 = cached_gpt2
    gpt2_accuracy = chord_results["gpt2_small"]["accuracy"]

    if tangent["labels"] != labels:
        raise RuntimeError("label arrays disagree between cached and new runs")

    # ------------------------------------------------------------- table
    print("\n" + "=" * 66)
    print("TANGENT vs CHORD NAVIGATION (all 10,042 val examples)")
    print("=" * 66)
    print(f"  {'Condition':<22} | {'Accuracy':>9} | source")
    print("  " + "-" * 58)
    table = [
        ("no_navigation", no_navigation["accuracy"], "cached"),
        ("full_model (chord)", chord["accuracy"], "cached"),
        ("full_model (tangent)", tangent["accuracy"], "computed now"),
        ("GPT-2 small", gpt2_accuracy, "cached"),
    ]
    for name, accuracy, source in table:
        print(f"  {name:<22} | {100 * accuracy:>8.2f}% | {source}")
    print("  " + "-" * 58)
    print(f"  tangent - no_navigation : "
          f"{100 * (tangent['accuracy'] - no_navigation['accuracy']):+.2f} points")
    print(f"  tangent - chord         : "
          f"{100 * (tangent['accuracy'] - chord['accuracy']):+.2f} points")
    print(f"  tangent - GPT-2 small   : "
          f"{100 * (tangent['accuracy'] - gpt2_accuracy):+.2f} points")

    # ---------------------------------------------------- McNemar tests
    tangent_predictions = tangent["predictions"]
    comparisons: Dict[str, Dict[str, Any]] = {
        "tangent_vs_no_navigation": rename_pairs(
            paired_comparison(tangent_predictions, no_navigation["predictions"], labels),
            "tangent",
            "no_navigation",
        ),
        "tangent_vs_chord": rename_pairs(
            paired_comparison(tangent_predictions, chord["predictions"], labels),
            "tangent",
            "chord",
        ),
        "tangent_vs_gpt2_small": rename_pairs(
            paired_comparison(tangent_predictions, gpt2["predictions"], labels),
            "tangent",
            "gpt2_small",
        ),
    }

    print("\n" + "=" * 66)
    print("McNEMAR EXACT TESTS (paired, all 10,042 examples)")
    print("=" * 66)
    for name, stats in comparisons.items():
        a_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        b_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(f"  {name}")
        print(f"    {a_key:<48} {stats[a_key]:>6}")
        print(f"    {b_key:<48} {stats[b_key]:>6}")
        print(f"    discordant {stats['discordant_pairs']:>5} | "
              f"p = {stats['mcnemar_p_value']:.4g} | significant: "
              f"{stats['significant_at_0.05']}")

    def beats(key: str, other_accuracy: float) -> bool:
        stats = comparisons[key]
        return bool(stats["significant_at_0.05"] and tangent["accuracy"] > other_accuracy)

    beats_no_navigation = beats("tangent_vs_no_navigation", no_navigation["accuracy"])
    beats_chord = beats("tangent_vs_chord", chord["accuracy"])
    beats_gpt2 = beats("tangent_vs_gpt2_small", gpt2_accuracy)

    # ------------------------------------------ diagnosis with the fix
    print("\n" + "=" * 66)
    print("DIAGNOSIS WITH use_tangent=True (200 random examples)")
    print("=" * 66)
    diagnosis = run_diagnosis(use_tangent=True)
    TANGENT_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TANGENT_DIAG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(diagnosis, handle, indent=2)

    summary = diagnosis["summary"]
    expectations = {
        "cos_ctx_nav_is_zero": {
            "value": summary["mean_cos_ctx_nav"],
            "target": "~0.0  (chord was -0.4486)",
            "passed": abs(summary["mean_cos_ctx_nav"]) < 0.05,
        },
        "mean_score_correct_beats_chord": {
            "value": summary["mean_score_correct_navigated"],
            "target": f"> {CHORD_DIAGNOSIS['mean_score_correct_navigated']}",
            "passed": summary["mean_score_correct_navigated"]
            > CHORD_DIAGNOSIS["mean_score_correct_navigated"],
        },
        "score_gap_beats_chord": {
            "value": summary["score_gap_navigated"],
            "target": f"> {CHORD_DIAGNOSIS['score_gap_navigated']}",
            "passed": summary["score_gap_navigated"] > CHORD_DIAGNOSIS["score_gap_navigated"],
        },
        "improved_beats_chord": {
            "value": summary["improved"],
            "target": f"> {CHORD_DIAGNOSIS['improved']}",
            "passed": summary["improved"] > CHORD_DIAGNOSIS["improved"],
        },
        "hurt_below_chord": {
            "value": summary["hurt"],
            "target": f"< {CHORD_DIAGNOSIS['hurt']}",
            "passed": summary["hurt"] < CHORD_DIAGNOSIS["hurt"],
        },
    }

    print("\n" + "=" * 66)
    print("EXPECTED SYMPTOMS OF THE FIX")
    print("=" * 66)
    for name, check in expectations.items():
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {name:<32} "
              f"value {check['value']}  target {check['target']}")

    # -------------------------------------------------------------- save
    result = {
        "experiment": "tangent navigation (radial component removed before fusion)",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": total,
        "seed": SEED,
        "config": {
            "store": "checkpoints/memory_store_full.pkl",
            "store_size": store.size,
            "retrieval_k": 10,
            "fuse_top_k": 3,
            "scoring": (
                "cosine(L2(encode(ending)), L2(ctx + fuse_directions(..., "
                "use_tangent=True, query_vector=ctx)))"
            ),
            "fix": "tangent_direction: d - (d.q) q, renormalised",
        },
        "table": [
            {"condition": name, "accuracy": accuracy, "source": source}
            for name, accuracy, source in table
        ],
        "conditions": {
            "no_navigation": {"accuracy": no_navigation["accuracy"], "source": "cached"},
            "full_model_chord": {"accuracy": chord["accuracy"], "source": "cached"},
            "full_model_tangent": {
                "accuracy": tangent["accuracy"],
                "correct": tangent["correct"],
                "num_examples": tangent["num_examples"],
                "seconds": tangent_seconds,
                "ms_per_example": tangent["ms_per_example"],
            },
            "gpt2_small": {
                "accuracy": gpt2_accuracy,
                "params": GPT2_PARAMETERS,
                "source": "cached",
            },
        },
        "mcnemar": comparisons,
        "tangent_beats_no_navigation": bool(beats_no_navigation),
        "tangent_beats_chord": bool(beats_chord),
        "tangent_beats_gpt2_small": bool(beats_gpt2),
        "diagnosis": {
            "file": "experiments/navigation_diagnosis_tangent.json",
            "summary": summary,
            "chord_reference": CHORD_DIAGNOSIS,
            "expectations": expectations,
            "all_expectations_passed": bool(all(c["passed"] for c in expectations.values())),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Diagnosis saved to {TANGENT_DIAG_PATH}")
    print("Tangent evaluation complete.")


if __name__ == "__main__":
    main()

