"""TASK 16 EXTENDED: squeeze more out of CLIP-style training.

TASK 16 showed the first significant gain (32.15% vs 31.31%, p = 0.024) but the
training loss was still falling at epoch 5 and the 500-example val curve was noisy
and trending up. This script runs three variants, each starting from a FRESH
``checkpoints/vae.pt`` head (the base encoder stays frozen), so each is a clean
comparison against the 5-epoch run:

    A  15 epochs, T=0.07, batch=32      -> checkpoints/vae_clip_15ep.pt
    B  T in {0.03, 0.07, 0.10, 0.20}, 10 epochs, batch=32
       best temperature                 -> checkpoints/vae_clip_best_T.pt
    C  10 epochs, T=0.07, batch=128     -> checkpoints/vae_clip_b128.pt
       (127 negatives per example instead of 31 = harder negatives, but note the
        loss here is a 4-way softmax over the endings of ONE example, so batch size
        changes the *gradient noise*, not the negative count - this is measured,
        not assumed.)

The overall best config then gets the full 10,042 evaluation plus McNemar tests
against the original VAE and against the TASK 16 5-epoch CLIP model.

Every variant runs on the cached frozen base embeddings (no sentence encoding), so
an epoch costs a few seconds.

Artifacts
---------
    checkpoints/vae_clip_{15ep,T0.03,T0.07,T0.10,T0.20,b128}.pt
    checkpoints/vae_clip_best_T.pt          best of the temperature sweep (B)
    checkpoints/vae_clip_best_overall.pt    best of everything
    experiments/results_clip_extended.json
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.autoencoder.vae import BASE_MODEL_NAME  # noqa: E402
from train_vae_clip import (  # noqa: E402
    EVAL_SIZE,
    GPT2_10042,
    LOAD_CHECKPOINT,
    ORIGINAL_NO_NAV_10042,
    SEED,
    evaluate,
    load_vae,
    rename_pairs,
    save_checkpoint,
    set_seed,
    train_head,
)
from run_ablations import paired_comparison  # noqa: E402

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_clip_extended.json"
TASK16_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_clip.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
TRAIN_CACHE = CHECKPOINTS_DIR / "train_full_base_embeddings.npz"
VAL_CACHE = CHECKPOINTS_DIR / "val_full_base_embeddings.npz"
BEST_T_PATH = CHECKPOINTS_DIR / "vae_clip_best_T.pt"
BEST_OVERALL_PATH = CHECKPOINTS_DIR / "vae_clip_best_overall.pt"

LEARNING_RATE = 1e-4
LOG_EVERY = 0  # per-step logging would drown a 65-epoch sweep; epochs are logged

VARIANTS: List[Dict[str, Any]] = [
    {"label": "15 epochs (A)", "tag": "15ep", "group": "A",
     "epochs": 15, "temperature": 0.07, "batch_size": 32},
    {"label": "T=0.03 (B)", "tag": "T0.03", "group": "B",
     "epochs": 10, "temperature": 0.03, "batch_size": 32},
    {"label": "T=0.07 (B)", "tag": "T0.07", "group": "B",
     "epochs": 10, "temperature": 0.07, "batch_size": 32},
    {"label": "T=0.10 (B)", "tag": "T0.10", "group": "B",
     "epochs": 10, "temperature": 0.10, "batch_size": 32},
    {"label": "T=0.20 (B)", "tag": "T0.20", "group": "B",
     "epochs": 10, "temperature": 0.20, "batch_size": 32},
    {"label": "batch=128 (C)", "tag": "b128", "group": "C",
     "epochs": 10, "temperature": 0.07, "batch_size": 128},
]


def main() -> None:
    set_seed(SEED)
    device = torch.device("cpu")

    with np.load(TRAIN_CACHE) as data:
        train_ctx = torch.from_numpy(data["ctx"])
        train_end = torch.from_numpy(data["ending"])
        train_labels = torch.from_numpy(data["labels"]).long()
    with np.load(VAL_CACHE) as data:
        val_ctx = torch.from_numpy(data["ctx"])
        val_end = torch.from_numpy(data["endings"])
        val_labels = torch.from_numpy(data["labels"]).long()

    print("TASK 16 EXTENDED - CLIP-style hyperparameter sweep")
    print(f"  train cache {tuple(train_ctx.shape)} | val cache {tuple(val_ctx.shape)}")
    print(f"  frozen base encoder : {BASE_MODEL_NAME}")
    print(f"  start checkpoint    : {LOAD_CHECKPOINT.name}")
    print(
        f"  {len(VARIANTS)} variants | per-epoch eval on the first {EVAL_SIZE} val examples"
    )

    results: List[Dict[str, Any]] = []
    overall_started = time.perf_counter()

    for spec in VARIANTS:
        print("\n" + "=" * 88)
        print(
            f"VARIANT {spec['label']} | tag={spec['tag']} | epochs={spec['epochs']} "
            f"| T={spec['temperature']} | batch={spec['batch_size']}"
        )
        print("=" * 88)
        vae, _ = load_vae(LOAD_CHECKPOINT, device)
        pretrain = evaluate(vae, val_ctx, val_end, val_labels, EVAL_SIZE)
        training = train_head(
            vae,
            train_ctx,
            train_end,
            train_labels,
            val_ctx,
            val_end,
            val_labels,
            epochs=spec["epochs"],
            batch_size=spec["batch_size"],
            learning_rate=LEARNING_RATE,
            temperature=spec["temperature"],
            eval_size=EVAL_SIZE,
            log_every=LOG_EVERY,
            seed=SEED,
            verbose=True,
        )

        if training["best_state"] is not None:
            vae.load_state_dict(training["best_state"], strict=False)
        full = evaluate(vae, val_ctx, val_end, val_labels)
        path = CHECKPOINTS_DIR / f"vae_clip_{spec['tag']}.pt"
        meta: Dict[str, Any] = {
            "epochs": spec["epochs"],
            "batch_size": spec["batch_size"],
            "learning_rate": LEARNING_RATE,
            "temperature": spec["temperature"],
            "history": training["history"],
            "best_epoch": training["best_epoch"],
            "best_val_500_accuracy": training["best_accuracy"],
            "variant_label": spec["label"],
            "pretrain_val_500_accuracy": pretrain["accuracy"],
            "full_10042_accuracy": full["accuracy"],
        }
        size_mb = save_checkpoint(
            vae,
            path,
            {"saved": "best", "variant": spec["tag"]},
            spec["temperature"],
            meta,
        )
        record: Dict[str, Any] = {
            "label": spec["label"],
            "tag": spec["tag"],
            "group": spec["group"],
            "epochs": spec["epochs"],
            "temperature": spec["temperature"],
            "batch_size": spec["batch_size"],
            "best_val_500_accuracy": training["best_accuracy"],
            "best_val_500_epoch": training["best_epoch"],
            "final_val_500_accuracy": training["history"][-1]["val_500_accuracy"],
            "full_10042_accuracy": full["accuracy"],
            "full_10042_correct": full["correct"],
            "full_10042_val_gap": full["val_gap"],
            "full_10042_gap_std": full["gap_std"],
            "full_10042_predictions": full["predictions"],
            "history": training["history"],
            "checkpoint": str(path.relative_to(PROJECT_ROOT)),
            "size_mb": size_mb,
            "train_seconds": training["seconds"],
        }
        results.append(record)
        print(
            f"\n  -> best val{EVAL_SIZE} {100 * training['best_accuracy']:.2f}% "
            f"(epoch {training['best_epoch']}) | full 10,042 "
            f"{100 * full['accuracy']:.4f}% | saved {path.name} ({size_mb:.2f} MB)",
            flush=True,
        )

    print(f"\nSweep wall time: {time.perf_counter() - overall_started:.1f}s")

    # ------------------------------------------------------- selection ----
    group_b = [r for r in results if r["group"] == "B"]
    best_b = (
        max(
            group_b,
            key=lambda r: (r["best_val_500_accuracy"], r["full_10042_accuracy"]),
        )
        if group_b
        else None
    )
    if best_b is not None:
        shutil.copyfile(CHECKPOINTS_DIR / f"vae_clip_{best_b['tag']}.pt", BEST_T_PATH)
        print(
            f"\nBest temperature (variant B): T={best_b['temperature']} "
            f"-> val{EVAL_SIZE} {100 * best_b['best_val_500_accuracy']:.2f}% "
            f"| full {100 * best_b['full_10042_accuracy']:.4f}% | saved {BEST_T_PATH.name}"
        )

    overall = max(
        results, key=lambda r: (r["best_val_500_accuracy"], r["full_10042_accuracy"])
    )
    shutil.copyfile(CHECKPOINTS_DIR / f"vae_clip_{overall['tag']}.pt", BEST_OVERALL_PATH)

    best_vae, best_checkpoint = load_vae(BEST_OVERALL_PATH, device)
    best_eval = evaluate(best_vae, val_ctx, val_end, val_labels)
    del best_vae
    print(
        f"Best overall            : {overall['label']} (epochs={overall['epochs']}, "
        f"T={overall['temperature']}, batch={overall['batch_size']}) "
        f"| saved {BEST_OVERALL_PATH.name}"
    )

    # ------------------------------------------------ variant comparison ----
    with TASK16_RESULTS_PATH.open(encoding="utf-8") as handle:
        task16 = json.load(handle)
    clip5_history = task16["training_history"]
    clip5_best = max(clip5_history, key=lambda r: r["val_500_accuracy"])
    clip5_acc = float(task16["results"]["CLIP-style no_nav"]["accuracy"])
    clip5_pred = [int(v) for v in task16["results"]["CLIP-style no_nav"]["predictions"]]

    print("\n" + "=" * 100)
    print(
        f"VARIANT COMPARISON (val{EVAL_SIZE} picks the checkpoint; full = all 10,042)"
    )
    print("=" * 100)
    print(
        f"  {'Config':<20} | {'val500 best':>11} | {'epoch':>6} | {'val500 final':>12} "
        f"| {'full 10,042':>11} | notes"
    )
    print("  " + "-" * 96)
    print(
        f"  {'Original (5ep)':<20} | {100 * clip5_best['val_500_accuracy']:>10.2f}% "
        f"| {clip5_best['epoch']:>6} | {100 * clip5_history[-1]['val_500_accuracy']:>11.2f}% "
        f"| {100 * clip5_acc:>10.4f}% | TASK 16 baseline"
    )
    for record in results:
        notes = ""
        if record["tag"] == overall["tag"]:
            notes = "BEST OVERALL"
        elif best_b is not None and record["group"] == "B" and record["tag"] == best_b["tag"]:
            notes = "best T"
        elif record["group"] == "C":
            notes = "batch 128"
        print(
            f"  {record['label']:<20} | {100 * record['best_val_500_accuracy']:>10.2f}% "
            f"| {record['best_val_500_epoch']:>6} "
            f"| {100 * record['final_val_500_accuracy']:>11.2f}% "
            f"| {100 * record['full_10042_accuracy']:>10.4f}% | {notes}"
        )
    print("  " + "-" * 96)

    # ---------------------------------------------------------- McNemar ----
    with GMN_PRED_PATH.open(encoding="utf-8") as handle:
        gmn = json.load(handle)
    original_pred = [int(value) for value in gmn["no_navigation"]["predictions"]]
    original_acc = float(gmn["no_navigation"]["accuracy"])

    mcnemar: Dict[str, Dict[str, Any]] = {
        "best_vs_original": rename_pairs(
            paired_comparison(
                best_eval["predictions"], original_pred, val_labels.tolist()
            ),
            "best_clip",
            "original",
        ),
        "best_vs_clip_5ep": rename_pairs(
            paired_comparison(best_eval["predictions"], clip5_pred, val_labels.tolist()),
            "best_clip",
            "clip_5ep",
        ),
    }
    primary = mcnemar["best_vs_original"]
    secondary = mcnemar["best_vs_clip_5ep"]
    beats_original = bool(
        primary["significant_at_0.05"] and best_eval["accuracy"] > original_acc
    )
    beats_clip5 = bool(
        secondary["significant_at_0.05"] and best_eval["accuracy"] > clip5_acc
    )

    print("\n" + "=" * 96)
    print("FULL 10,042-EXAMPLE EVALUATION - BEST CONFIG")
    print("=" * 96)
    print(
        f"  config     : {overall['label']} (epochs={overall['epochs']}, "
        f"T={overall['temperature']}, batch={overall['batch_size']})"
    )
    print(
        f"  accuracy   : {100 * best_eval['accuracy']:.4f}% "
        f"({best_eval['correct']}/{best_eval['num_examples']})"
    )
    print(
        f"  val gap    : {best_eval['val_gap']:.6f} "
        f"(cos+ {best_eval['mean_cos_correct']:.4f} | "
        f"cos- {best_eval['mean_cos_wrong']:.4f})"
    )
    print(f"  val gap std: {best_eval['gap_std']:.4f}")

    print("\n" + "=" * 96)
    print("FINAL COMPARISON (all 10,042 val examples)")
    print("=" * 96)
    print(
        f"  {'Condition':<26} | {'Accuracy':>9} | {'vs original':>12} | {'p-value':>10}"
    )
    print("  " + "-" * 68)
    print(
        f"  {'Original VAE no_nav':<26} | {100 * original_acc:>8.4f}% "
        f"| {'baseline':>12} | {'-':>10}"
    )
    print(
        f"  {'CLIP 5ep (T16)':<26} | {100 * clip5_acc:>8.4f}% "
        f"| {100 * (clip5_acc - original_acc):>+11.4f}% | {'0.0240':>10}"
    )
    print(
        f"  {'CLIP best config':<26} | {100 * best_eval['accuracy']:>8.4f}% "
        f"| {100 * (best_eval['accuracy'] - original_acc):>+11.4f}% "
        f"| {primary['mcnemar_p_value']:>10.4f}"
    )
    print(
        f"  {'GPT-2 small (cached)':<26} | {100 * GPT2_10042:>8.4f}% "
        f"| {100 * (GPT2_10042 - original_acc):>+11.4f}% | {'cached':>10}"
    )
    print("  " + "-" * 68)

    print("\n" + "=" * 96)
    print("McNEMAR EXACT TESTS")
    print("=" * 96)
    for name, stats in mcnemar.items():
        first_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        second_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(
            f"  {name:<20} | {first_key} {stats[first_key]:>5} "
            f"| {second_key} {stats[second_key]:>5} "
            f"| discordant {stats['discordant_pairs']:>5} "
            f"| p = {stats['mcnemar_p_value']:.4f} "
            f"| significant {stats['significant_at_0.05']}"
        )

    a_record = next((r for r in results if r["group"] == "A"), None)
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(
        f"  best config          : {overall['label']} | epochs={overall['epochs']} "
        f"| T={overall['temperature']} | batch={overall['batch_size']}"
    )
    print(
        f"  best full 10,042 acc : {100 * best_eval['accuracy']:.4f}% "
        f"({best_eval['correct']}/{best_eval['num_examples']})"
    )
    print(
        f"  vs original 31.31%   : {100 * (best_eval['accuracy'] - original_acc):+.4f} "
        f"points | p = {primary['mcnemar_p_value']:.4f} "
        f"| significant {primary['significant_at_0.05']}"
    )
    print(
        f"  vs CLIP 5ep 32.15%   : {100 * (best_eval['accuracy'] - clip5_acc):+.4f} "
        f"points | p = {secondary['mcnemar_p_value']:.4f} "
        f"| significant {secondary['significant_at_0.05']}"
    )
    print(f"  beats original sig   : {beats_original}")
    print(f"  beats CLIP 5ep sig   : {beats_clip5}")
    if a_record is not None:
        print(f"\n  15-epoch val{EVAL_SIZE} curve (variant A):")
        for entry in a_record["history"]:
            print(
                f"    epoch {entry['epoch']:>2}: {100 * entry['val_500_accuracy']:.2f}% "
                f"({entry['val_500_correct']}/{EVAL_SIZE}) | loss {entry['train_loss']:.4f}"
            )

    payload = {
        "task": "TASK 16 EXTENDED - CLIP-style hyperparameter sweep",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": best_eval["num_examples"],
        "method": {
            "objective": "CrossEntropy(cos(ending, ctx) / temperature, label)",
            "base_encoder": "frozen all-MiniLM-L6-v2",
            "fine_tuned": "VAE head (fc, mu_layer, logvar_layer)",
            "decoder": "ignored",
            "start_checkpoint": "checkpoints/vae.pt",
            "learning_rate": LEARNING_RATE,
        },
        "selection_metric": (
            f"rank_1 on the first {EVAL_SIZE} val examples - noisy (SE ~2.2 points), so "
            "the full 10,042 column is the trustworthy one"
        ),
        "variants": results,
        "best_temperature": (
            {
                "tag": best_b["tag"],
                "temperature": best_b["temperature"],
                "best_val_500_accuracy": best_b["best_val_500_accuracy"],
                "full_10042_accuracy": best_b["full_10042_accuracy"],
                "checkpoint": str(BEST_T_PATH.relative_to(PROJECT_ROOT)),
            }
            if best_b is not None
            else None
        ),
        "best_overall": {
            "label": overall["label"],
            "tag": overall["tag"],
            "epochs": overall["epochs"],
            "temperature": overall["temperature"],
            "batch_size": overall["batch_size"],
            "best_val_500_accuracy": overall["best_val_500_accuracy"],
            "best_val_500_epoch": overall["best_val_500_epoch"],
            "checkpoint": str(BEST_OVERALL_PATH.relative_to(PROJECT_ROOT)),
            "checkpoint_temperature": best_checkpoint.get("temperature"),
        },
        "full_eval": {
            "accuracy": best_eval["accuracy"],
            "correct": best_eval["correct"],
            "val_gap": best_eval["val_gap"],
            "gap_std": best_eval["gap_std"],
            "mean_cos_correct": best_eval["mean_cos_correct"],
            "mean_cos_wrong": best_eval["mean_cos_wrong"],
            "rank_counts": best_eval["rank_counts"],
            "predictions": best_eval["predictions"],
        },
        "baselines": {
            "original_no_nav_10042": original_acc,
            "clip_5ep_10042": clip5_acc,
            "gpt2_small_10042": GPT2_10042,
        },
        "task16_5ep_reference": {
            "best_val_500_accuracy": clip5_best["val_500_accuracy"],
            "best_val_500_epoch": clip5_best["epoch"],
            "val_500_curve": [
                {
                    "epoch": entry["epoch"],
                    "val_500_accuracy": entry["val_500_accuracy"],
                    "train_loss": entry["train_loss"],
                }
                for entry in clip5_history
            ],
            "full_10042_accuracy": clip5_acc,
        },
        "comparison": {
            "best_config": (
                f"{overall['label']}: epochs={overall['epochs']}, "
                f"T={overall['temperature']}, batch={overall['batch_size']}"
            ),
            "best_full_10042_accuracy": best_eval["accuracy"],
            "delta_vs_original": best_eval["accuracy"] - original_acc,
            "delta_vs_clip_5ep": best_eval["accuracy"] - clip5_acc,
            "delta_vs_gpt2": best_eval["accuracy"] - GPT2_10042,
            "mcnemar_p_value_vs_original": primary["mcnemar_p_value"],
            "significant_vs_original": bool(primary["significant_at_0.05"]),
            "mcnemar_p_value_vs_clip_5ep": secondary["mcnemar_p_value"],
            "significant_vs_clip_5ep": bool(secondary["significant_at_0.05"]),
            "beats_original_significantly": beats_original,
            "beats_clip_5ep_significantly": beats_clip5,
            "val_gap": best_eval["val_gap"],
            "val_gap_std": best_eval["gap_std"],
        },
        "mcnemar": mcnemar,
        "artifacts": {
            "per_variant_checkpoints": {
                record["tag"]: record["checkpoint"] for record in results
            },
            "best_temperature": str(BEST_T_PATH.relative_to(PROJECT_ROOT)),
            "best_overall": str(BEST_OVERALL_PATH.relative_to(PROJECT_ROOT)),
        },
        "caveats": [
            "checkpoints are selected on 500 val examples (SE ~2.2 points); the full "
            "10,042 column is the reliable signal",
            "6 variants were compared on the same 10,042 examples with no "
            "multiple-comparison correction, so p-values are optimistic",
            f"single seed ({SEED}); run-to-run variance was not measured",
        ],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {RESULTS_PATH} ({RESULTS_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()



