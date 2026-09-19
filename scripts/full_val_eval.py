"""STEP 1: full-validation (10,042) comparison of no_navigation / full_model / GPT-2.

At n=500 the unpaired standard error is 2.11 points, which cannot separate a
4-point effect; at n=10,042 it is ~0.47 points. This script re-runs the two GMN
conditions on the whole split and pairs them against GPT-2 with an exact McNemar
test.

It reuses the verified ablation machinery (``run_ablations.run_ablation`` and
``run_ablations.paired_comparison``) so the semantics are identical to TASK 9, and
reads the precomputed vector cache, so no VAE is needed here.

Writes ``experiments/results_full_val.json``.
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
from run_ablations import paired_comparison, run_ablation  # noqa: E402
from src.memory.store import MemoryStore  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
FULL_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full.pkl"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_full_val.json"

SUBSET_SIZE = 500  # data/val.json == the first 500 rows of the full split
SEED = 42


def accuracy_on(predictions: List[int], labels: List[int], count: int | None = None) -> float:
    preds = np.asarray(predictions if count is None else predictions[:count])
    truth = np.asarray(labels if count is None else labels[:count])
    return float((preds == truth).mean())


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
    print(f"full validation examples: {total}")

    store = MemoryStore.load(FULL_STORE_PATH)
    print(f"full memory store: {store.size} vectors")

    print("loading cached query/ending vectors ...")
    queries, endings = vecache.load_all_chunks(total)
    print(f"  queries {queries.shape} | endings {endings.shape}")

    rng = np.random.default_rng(SEED)
    conditions: Dict[str, Dict[str, Any]] = {}
    for name, mode in (("full_model", "full"), ("no_navigation", "no_navigation")):
        started = time.perf_counter()
        outcome = run_ablation(val_data, queries, endings, store, mode, rng)
        conditions[name] = outcome
        print(f"  [{name}] {100 * outcome['accuracy']:.2f}% "
              f"({outcome['correct']}/{outcome['num_examples']}) in "
              f"{time.perf_counter() - started:.1f}s")

    # cache the GMN predictions so re-runs (e.g. once GPT-2 finishes) are instant
    with GMN_PRED_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                name: {
                    "predictions": outcome["predictions"],
                    "labels": outcome["labels"],
                    "accuracy": outcome["accuracy"],
                }
                for name, outcome in conditions.items()
            },
            handle,
        )
    print(f"GMN predictions cached to {GMN_PRED_PATH}")

    gpt2: Dict[str, Any] | None = None
    if GPT2_PRED_PATH.exists():
        with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
            gpt2 = json.load(handle)
        print(f"GPT-2 predictions loaded: {gpt2['accuracy']:.4f} "
              f"({gpt2['num_examples']} examples)")
    else:
        print(f"GPT-2 predictions not found at {GPT2_PRED_PATH} - skipping that row")

    # ------------------------------------------------------------- table
    subset_accuracy = {
        name: accuracy_on(outcome["predictions"], outcome["labels"], SUBSET_SIZE)
        for name, outcome in conditions.items()
    }

    print("\n" + "=" * 64)
    print("FULL VALIDATION COMPARISON (same examples for every condition)")
    print("=" * 64)
    print(f"  {'Condition':<16} | {'n=500':>7} | {'n=10,042':>9} | delta")
    print("  " + "-" * 52)
    table: List[Dict[str, Any]] = []
    for name in ("no_navigation", "full_model"):
        full_acc = conditions[name]["accuracy"]
        row = {
            "condition": name,
            "accuracy_n500": subset_accuracy[name],
            "accuracy_full": full_acc,
            "delta": full_acc - subset_accuracy[name],
        }
        table.append(row)
        print(f"  {name:<16} | {100 * subset_accuracy[name]:>6.1f}% | "
              f"{100 * full_acc:>8.1f}% | {100 * row['delta']:>+5.1f}")

    gpt2_accuracy_full = None
    if gpt2 is not None:
        gpt2_accuracy_full = accuracy_on(gpt2["predictions"], gpt2["labels"])
        gpt2_accuracy_500 = accuracy_on(gpt2["predictions"], gpt2["labels"], SUBSET_SIZE)
        row = {
            "condition": "gpt2_small",
            "accuracy_n500": gpt2_accuracy_500,
            "accuracy_full": gpt2_accuracy_full,
            "delta": gpt2_accuracy_full - gpt2_accuracy_500,
        }
        table.append(row)
        print(f"  {'GPT-2 small':<16} | {100 * gpt2_accuracy_500:>6.1f}% | "
              f"{100 * gpt2_accuracy_full:>8.1f}% | {100 * row['delta']:>+5.1f}")
    print("  " + "-" * 52)

    unpaired_se = 100.0 * (
        conditions["full_model"]["accuracy"]
        * (1.0 - conditions["full_model"]["accuracy"])
        / total
    ) ** 0.5
    print(f"  unpaired std error at n={total}: {unpaired_se:.2f} points "
          f"(was 2.11 at n=500)")

    # -------------------------------------------------- McNemar tests
    labels = conditions["full_model"]["labels"]
    no_nav_preds = conditions["no_navigation"]["predictions"]
    full_preds = conditions["full_model"]["predictions"]

    comparisons: Dict[str, Dict[str, Any]] = {}
    comparisons["no_navigation_vs_full_model"] = rename_pairs(
        paired_comparison(no_nav_preds, full_preds, labels), "no_navigation", "full_model"
    )
    if gpt2 is not None:
        comparisons["no_navigation_vs_gpt2_small"] = rename_pairs(
            paired_comparison(no_nav_preds, gpt2["predictions"], labels),
            "no_navigation",
            "gpt2_small",
        )
        comparisons["full_model_vs_gpt2_small"] = rename_pairs(
            paired_comparison(full_preds, gpt2["predictions"], labels),
            "full_model",
            "gpt2_small",
        )

    print("\n" + "=" * 64)
    print("McNEMAR EXACT TESTS (paired, same examples)")
    print("=" * 64)
    for name, stats in comparisons.items():
        a_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        b_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(f"  {name}")
        print(f"    {a_key:<52} {stats[a_key]:>6}")
        print(f"    {b_key:<52} {stats[b_key]:>6}")
        print(f"    discordant {stats['discordant_pairs']:>5} | "
              f"p = {stats['mcnemar_p_value']:.4g} | "
              f"significant at 0.05: {stats['significant_at_0.05']}")

    # -------------------------------------------------------------- save
    results = {
        "experiment": "full validation comparison (n=10,042)",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": total,
        "subset_size": SUBSET_SIZE,
        "seed": SEED,
        "gmn_config": {
            "store": "checkpoints/memory_store_full.pkl",
            "store_size": store.size,
            "scoring": "cosine(L2(encode(ending)), navigated position)",
            "classifier_head": "none",
            "retrieval_k": 10,
            "fuse_top_k": 3,
        },
        "table": table,
        "conditions": {
            name: {
                "accuracy": conditions[name]["accuracy"],
                "accuracy_n500": subset_accuracy[name],
                "correct": conditions[name]["correct"],
                "num_examples": conditions[name]["num_examples"],
                "seconds": conditions[name]["seconds"],
                "ms_per_example": conditions[name]["ms_per_example"],
            }
            for name in conditions
        },
        "gpt2_small": None
        if gpt2 is None
        else {
            "accuracy": gpt2_accuracy_full,
            "accuracy_length_normalized": gpt2.get("accuracy_length_normalized"),
            "params": 124439808,
            "model_name": gpt2.get("model_name"),
            "seconds": gpt2.get("seconds"),
        },
        "mcnemar": comparisons,
        "unpaired_std_error_points": unpaired_se,
        "gpt2_pending": gpt2 is None,
        "notes": {
            "n500_subset": "data/val.json is the first 500 rows of the full split",
            "encoding": (
                "vectors cached by scripts/precompute_val_full_vectors.py using the "
                "same per-example batch shapes as score_example"
            ),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    if gpt2 is None:
        print("NOTE: GPT-2 row pending - re-run this script once "
              "scripts/gpt2_full_val.py has written its predictions.")
    print("Full-validation comparison complete.")


if __name__ == "__main__":
    main()

