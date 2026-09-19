"""STEP 1: honest lambda tuning - the reported number must not be fitted.

The lambda sweep in ``results_additive.json`` chose the best lambda on the same
10,042 examples it reported, which leaks. Here the split is deterministic:

    tune_set = first 5,000 examples     -> choose lambda
    eval_set = last 5,042 examples      -> report ONLY that lambda

No re-encoding is needed: ``experiments/additive_scores.npz`` already holds the
per-example context and navigation score terms in validation order, so the two
sets share exactly the same retrieval, navigation vectors and encodings.

Writes ``experiments/results_honest.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_ablations import paired_comparison  # noqa: E402

SCORES_PATH = PROJECT_ROOT / "experiments" / "additive_scores.npz"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_honest.json"

TUNE_SIZE = 5000
LAMBDAS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.25, 0.3, 0.5, 1.0)


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


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def main() -> None:
    scores = np.load(SCORES_PATH)
    context_scores = scores["context_scores"]
    navigation_scores = scores["navigation_scores"]
    labels = scores["labels"]
    total = len(labels)

    with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
        gpt2 = json.load(handle)
    gpt2_predictions = np.asarray(gpt2["predictions"])

    tune_slice = slice(0, TUNE_SIZE)
    eval_slice = slice(TUNE_SIZE, total)
    print(f"val examples {total} -> tune {TUNE_SIZE} (first) | "
          f"eval {total - TUNE_SIZE} (last)")
    print(f"score matrices {context_scores.shape} | labels {labels.shape}")
    print(f"GPT-2 predictions {gpt2_predictions.shape} ({gpt2['num_examples']} examples)")

    # ------------------------------------------------------------- tuning
    print("\n" + "=" * 62)
    print(f"TUNE SET ({TUNE_SIZE} examples) - choosing lambda")
    print("=" * 62)
    print(f"  {'lambda':>6} | {'accuracy':>9}")
    print("  " + "-" * 20)
    tune_rows: List[Dict[str, float]] = []
    for lam in LAMBDAS:
        predictions = np.argmax(
            context_scores[tune_slice] + lam * navigation_scores[tune_slice], axis=1
        )
        tune_accuracy = accuracy(predictions, labels[tune_slice])
        tune_rows.append({"lambda_nav": lam, "accuracy": tune_accuracy})
        print(f"  {lam:>6.2f} | {100 * tune_accuracy:>8.2f}%")
    print("  " + "-" * 20)

    best_row = max(tune_rows, key=lambda row: row["accuracy"])
    best_lambda = best_row["lambda_nav"]
    print(f"\n  best_lambda = {best_lambda:.2f} "
          f"(tune accuracy {100 * best_row['accuracy']:.2f}%)")
    print(f"  NOTE: this lambda is now FROZEN and only evaluated below.")

    # --------------------------------------------------- honest evaluation
    eval_labels = labels[eval_slice]
    no_navigation_predictions = np.argmax(context_scores[eval_slice], axis=1)
    gmn_predictions = np.argmax(
        context_scores[eval_slice] + best_lambda * navigation_scores[eval_slice], axis=1
    )
    gpt2_eval_predictions = gpt2_predictions[eval_slice]

    no_navigation_accuracy = accuracy(no_navigation_predictions, eval_labels)
    gmn_accuracy = accuracy(gmn_predictions, eval_labels)
    gpt2_accuracy = accuracy(gpt2_eval_predictions, eval_labels)

    print("\n" + "=" * 74)
    print(f"EVAL SET ({total - TUNE_SIZE} examples, honest - lambda chosen on tune only)")
    print("=" * 74)
    header = (f"  {'Condition':<20} | {'Accuracy':>9} | {'vs no_nav':>10} | "
              f"{'vs GPT-2':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'no_navigation':<20} | {100 * no_navigation_accuracy:>8.2f}% | "
          f"{'baseline':>10} | "
          f"{100 * (no_navigation_accuracy - gpt2_accuracy):>+8.2f}")
    print(f"  {'GMN (lambda=' + f'{best_lambda:.2f})':<20} | "
          f"{100 * gmn_accuracy:>8.2f}% | "
          f"{100 * (gmn_accuracy - no_navigation_accuracy):>+9.2f} | "
          f"{100 * (gmn_accuracy - gpt2_accuracy):>+8.2f}")
    print(f"  {'GPT-2 small':<20} | {100 * gpt2_accuracy:>8.2f}% | "
          f"{100 * (gpt2_accuracy - no_navigation_accuracy):>+9.2f} | "
          f"{'baseline':>9}")
    print("  " + "-" * (len(header) - 2))
    print(f"  eval set: {len(eval_labels)} examples | unpaired std error ~"
          f"{100 * (gmn_accuracy * (1 - gmn_accuracy) / len(eval_labels)) ** 0.5:.2f} points")

    # -------------------------------------------------------- McNemar tests
    eval_label_list = [int(value) for value in eval_labels]
    comparisons: Dict[str, Dict[str, Any]] = {
        "gmn_vs_no_navigation": rename_pairs(
            paired_comparison(
                [int(value) for value in gmn_predictions],
                [int(value) for value in no_navigation_predictions],
                eval_label_list,
            ),
            f"gmn_lambda_{best_lambda:g}",
            "no_navigation",
        ),
        "gmn_vs_gpt2_small": rename_pairs(
            paired_comparison(
                [int(value) for value in gmn_predictions],
                [int(value) for value in gpt2_eval_predictions],
                eval_label_list,
            ),
            f"gmn_lambda_{best_lambda:g}",
            "gpt2_small",
        ),
        "no_navigation_vs_gpt2_small": rename_pairs(
            paired_comparison(
                [int(value) for value in no_navigation_predictions],
                [int(value) for value in gpt2_eval_predictions],
                eval_label_list,
            ),
            "no_navigation",
            "gpt2_small",
        ),
    }

    print("\n" + "=" * 74)
    print("McNEMAR EXACT TESTS (eval set only)")
    print("=" * 74)
    for name, stats in comparisons.items():
        a_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        b_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(f"  {name}")
        print(f"    {a_key:<52} {stats[a_key]:>6}")
        print(f"    {b_key:<52} {stats[b_key]:>6}")
        print(f"    discordant {stats['discordant_pairs']:>5} | "
              f"p = {stats['mcnemar_p_value']:.4f} | significant: "
              f"{stats['significant_at_0.05']}")

    navigation_helps = bool(
        comparisons["gmn_vs_no_navigation"]["significant_at_0.05"]
        and gmn_accuracy > no_navigation_accuracy
    )
    beats_gpt2 = bool(
        comparisons["gmn_vs_gpt2_small"]["significant_at_0.05"]
        and gmn_accuracy > gpt2_accuracy
    )
    no_navigation_beats_gpt2 = bool(
        comparisons["no_navigation_vs_gpt2_small"]["significant_at_0.05"]
        and no_navigation_accuracy > gpt2_accuracy
    )
    print(f"\n  navigation helps significantly on the honest eval: {navigation_helps}")
    print(f"  GMN beats GPT-2 small significantly               : {beats_gpt2}")
    print(f"  no_navigation beats GPT-2 small significantly     : {no_navigation_beats_gpt2}")
    print(f"  GMN parameters 624,256 vs GPT-2 124,439,808 "
          f"({124439808 / 624256:.0f}x smaller)")

    # ------------------------------------- how big was the leak? (diagnostic)
    leak_rows: List[Dict[str, float]] = []
    for lam in LAMBDAS:
        predictions = np.argmax(
            context_scores[eval_slice] + lam * navigation_scores[eval_slice], axis=1
        )
        leak_rows.append(
            {"lambda_nav": lam, "eval_accuracy": accuracy(predictions, eval_labels)}
        )
    leaky_best = max(leak_rows, key=lambda row: row["eval_accuracy"])
    tune_spread = max(row["accuracy"] for row in tune_rows) - min(
        row["accuracy"] for row in tune_rows
    )
    lambda_gain_is_noise = bool(
        leaky_best["eval_accuracy"] - gmn_accuracy > 0.005 and tune_spread < 0.01
    )
    print("\n  POST-HOC (leaky) eval-set sweep, for reference only:")
    for row in leak_rows:
        marker = "  <- honest choice" if row["lambda_nav"] == best_lambda else ""
        print(f"    lambda {row['lambda_nav']:>4.2f} -> {100 * row['eval_accuracy']:>5.2f}%{marker}")
    print(f"    choosing lambda on the eval set would have given "
          f"{100 * leaky_best['eval_accuracy']:.2f}% at lambda "
          f"{leaky_best['lambda_nav']:.2f}, i.e. "
          f"{100 * (leaky_best['eval_accuracy'] - gmn_accuracy):+.2f} points of "
          f"optimistic bias vs the honest {100 * gmn_accuracy:.2f}%")
    print(f"    tune-set spread across lambda: {100 * tune_spread:.2f} points "
          f"(under 1 point -> lambda is chosen by noise)")
    print(f"    lambda_is_untunable: {lambda_gain_is_noise}")

    # -------------------------------------------------------------- save
    result = {
        "experiment": "honest lambda evaluation (tune/eval split)",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": total,
        "tune_size": TUNE_SIZE,
        "eval_size": total - TUNE_SIZE,
        "split_definition": (
            f"first {TUNE_SIZE} = tune, remaining {total - TUNE_SIZE} = eval (deterministic)"
        ),
        "config": {
            "store": "checkpoints/memory_store_full_v2.pkl",
            "target": "correct_ending",
            "use_tangent": True,
            "retrieval_k": 10,
            "fuse_top_k": 3,
            "scoring": "cos(ending, ctx) + lambda_nav * cos(ending, nav_vector)",
            "position_moved": False,
        },
        "tune": {
            "lambdas": tune_rows,
            "best_lambda": best_lambda,
            "best_lambda_tune_accuracy": best_row["accuracy"],
        },
        "eval": {
            "no_navigation": {
                "accuracy": no_navigation_accuracy,
                "correct": int((no_navigation_predictions == eval_labels).sum()),
            },
            "gmn": {
                "lambda_nav": best_lambda,
                "accuracy": gmn_accuracy,
                "correct": int((gmn_predictions == eval_labels).sum()),
                "delta_vs_no_navigation": gmn_accuracy - no_navigation_accuracy,
                "delta_vs_gpt2": gmn_accuracy - gpt2_accuracy,
            },
            "gpt2_small": {
                "accuracy": gpt2_accuracy,
                "params": 124439808,
            },
        },
        "lambda_is_untunable": lambda_gain_is_noise,
        "no_navigation_beats_gpt2_significantly": no_navigation_beats_gpt2,
        "mcnemar": comparisons,
        "post_hoc_leaky_eval_sweep": {
            "note": (
                "Diagnostic only - these lambdas were NOT chosen honestly. Included "
                "to show the size of the optimistic bias the tune/eval split removes."
            ),
            "rows": leak_rows,
            "best_if_lambda_chosen_on_eval": leaky_best,
            "optimistic_bias_points": leaky_best["eval_accuracy"] - gmn_accuracy,
        },
        "navigation_helps_significantly": navigation_helps,
        "gmn_beats_gpt2_significantly": beats_gpt2,
        "gmn_parameters": 624256,
        "gpt2_parameters": 124439808,
        "caveat": (
            "Paper number: lambda was chosen on the first 5,000 examples and this "
            "accuracy is measured only on the remaining 5,041."
        ),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("Honest lambda evaluation complete.")


if __name__ == "__main__":
    main()
