"""TASK 8: evaluate GMN geometric ending scoring against random and GPT-2 small.

All three models are scored on the same 500 validation examples. GMN uses no
trained classifier head: each of the 4 candidate endings is scored by its cosine
similarity to the navigated latent position (see ``score_example``).

Writes ``experiments/results.json`` (consumed by TASK 12) and
``experiments/gmn_val_predictions.json`` (per-example predictions and scores).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.eval.evaluate import (  # noqa: E402
    DEFAULT_FUSE_TOP_K,
    DEFAULT_K,
    evaluate_gmn,
    evaluate_gpt2_variants,
    evaluate_random,
)
from src.memory.store import MemoryStore  # noqa: E402

VAL_PATH = PROJECT_ROOT / "data" / "val.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store.pkl"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results.json"
PREDICTIONS_PATH = PROJECT_ROOT / "experiments" / "gmn_val_predictions.json"

GPT2_MODEL_NAME = "gpt2"
#: measured parameter count of the loaded gpt2 checkpoint
GPT2_PARAMETERS = 124_439_808
#: the figure quoted in TASKS.md/README
GPT2_QUOTED_PARAMETERS = "117M"
GMN_QUOTED_PARAMETERS = "<10M"
SEED = 42


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config (same pattern as TASK 3/4/6/7)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vae = VAE(
        base_model_name=config["base_model_name"],
        input_dim=config["base_embed_dim"],
        hidden_dim=config["head_hidden_dim"],
        latent_dim=config["latent_dim"],
        decoder_hidden_dim=config["decoder_hidden_dim"],
        logvar_bias=config["logvar_bias"],
        device=device,
    ).to(device)
    missing, unexpected = vae.load_state_dict(checkpoint["model_state_dict"], strict=False)
    real_missing = [key for key in missing if not key.startswith("encoder.")]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={real_missing} unexpected={unexpected}")
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae, checkpoint


def main() -> None:
    with VAL_PATH.open(encoding="utf-8") as handle:
        val_data = json.load(handle)

    vae, vae_checkpoint = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    store = MemoryStore.load(STORE_PATH)

    vae_head_parameters = sum(p.numel() for p in vae.head.parameters())
    vae_decoder_parameters = sum(p.numel() for p in vae.decoder.parameters())
    gmn_parameters = vae_head_parameters + vae_decoder_parameters

    print(f"Validation examples: {len(val_data)} | store size: {store.size}")
    print(f"GMN parameters: VAE head {vae_head_parameters:,} + "
          f"decoder {vae_decoder_parameters:,} = {gmn_parameters:,} "
          f"(base encoder frozen, not counted)")
    print(f"\nGMN uses NO trained classifier head - geometric ending scoring only.")

    # ------------------------------------------------------------ random
    print("\n[1/3] random baseline ...")
    random_started = time.perf_counter()
    random_accuracy = evaluate_random(val_data)
    random_seconds = time.perf_counter() - random_started

    # ------------------------------------------------------------ gpt-2
    print("[2/3] GPT-2 small baseline (single pass, both aggregations; ~1.5 min on CPU) ...")
    gpt2_variants = evaluate_gpt2_variants(val_data, model_name=GPT2_MODEL_NAME)
    gpt2_accuracy = gpt2_variants["sum"]  # TASK 8 specification: raw summed log prob
    gpt2_accuracy_normalized = gpt2_variants["length_normalized"]
    gpt2_length_bias = gpt2_variants["longest_ending_picked_by_sum_rate"]
    gpt2_seconds = gpt2_variants["seconds"]

    # ------------------------------------------------------------ gmn
    print("[3/3] GMN geometric ending scoring ...")
    gmn_started = time.perf_counter()
    gmn_accuracy = evaluate_gmn(
        val_data, vae, store, predictions_path=PREDICTIONS_PATH
    )
    gmn_seconds = time.perf_counter() - gmn_started
    gmn_ms_per_example = 1000.0 * gmn_seconds / max(len(val_data), 1)

    # ------------------------------------------------------- diagnostics
    with PREDICTIONS_PATH.open(encoding="utf-8") as handle:
        gmn_detail = json.load(handle)

    labels_array = np.asarray(gmn_detail["labels"])
    scores_array = np.asarray(gmn_detail["scores"])
    predictions_array = np.asarray(gmn_detail["predictions"])

    order = np.argsort(-scores_array, axis=1)
    ranks = np.asarray(
        [
            int(np.where(order[row] == labels_array[row])[0][0]) + 1
            for row in range(len(labels_array))
        ]
    )
    rank_counts = Counter(ranks.tolist())
    sorted_scores = np.sort(scores_array, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    print("\nDiagnostics (GMN):")
    print(f"  prediction distribution  : {np.bincount(predictions_array, minlength=4).tolist()}")
    print(f"  truth distribution       : {np.bincount(labels_array, minlength=4).tolist()}")
    print(f"  rank of correct ending   : {dict(sorted(rank_counts.items()))}  (1 = picked)")
    print(f"  top-2 contains correct   : {(ranks <= 2).mean():.4f}")
    print(f"  mean score margin (1st-2nd): {margins.mean():.6f}")

    # ------------------------------------------------------------- table
    print("\n" + "=" * 46)
    print("RESULTS ON 500 HELLASWAG VALIDATION EXAMPLES")
    print("=" * 46)
    header = f"{'Model':<14} | {'Params':<7} | {'Accuracy':>8}"
    print(header)
    print("-" * len(header))
    print(f"{'Random':<14} | {'-':<7} | {100 * random_accuracy:>7.1f}%")
    print(f"{'GPT-2 small':<14} | {GPT2_QUOTED_PARAMETERS:<7} | {100 * gpt2_accuracy:>7.1f}%")
    print(f"{'GMN (ours)':<14} | {GMN_QUOTED_PARAMETERS:<7} | {100 * gmn_accuracy:>7.1f}%")
    print("-" * len(header))

    beats_random = gmn_accuracy > 0.25
    beats_gpt2 = gmn_accuracy > gpt2_accuracy
    beats_gpt2_fair = gmn_accuracy > gpt2_accuracy_normalized

    print("\n  GPT-2 aggregation variants (same single forward pass):")
    print(f"    raw summed log prob  (TASK 8 spec) : {100 * gpt2_accuracy:.1f}%")
    print(f"    length-normalised                  : {100 * gpt2_accuracy_normalized:.1f}%"
          f"   <- closest to the ~50% quoted in the README")
    print(f"    raw sum picked the LONGEST ending  : {100 * gpt2_length_bias:.1f}% of examples"
          f"   <- length bias in the raw-sum objective")

    print(f"\n  GMN exact parameter count : {gmn_parameters:,}")
    print(f"  GMN beats random (>25%)   : {beats_random}")
    print(f"  GMN beats GPT-2 (raw sum) : {beats_gpt2} "
          f"({100 * gmn_accuracy:.1f}% vs {100 * gpt2_accuracy:.1f}%)")
    print(f"  GMN beats GPT-2 (length-normalised, honest): {beats_gpt2_fair} "
          f"({100 * gmn_accuracy:.1f}% vs {100 * gpt2_accuracy_normalized:.1f}%)")
    print(f"  GMN time per example      : {gmn_ms_per_example:.2f} ms "
          f"({gmn_seconds:.1f}s total)")
    print(f"  GPT-2 time per example    : "
          f"{1000.0 * gpt2_seconds / max(len(val_data), 1):.2f} ms")

    # -------------------------------------------------------------- save
    results = {
        "dataset": "hellaswag",
        "split": "validation",
        "val_examples": len(val_data),
        "seed": SEED,
        "gmn_config": {
            "scoring": "cosine(L2(encode(ending)), L2(query_vector + nav_vector))",
            "classifier_head": "none",
            "retrieval_k": DEFAULT_K,
            "fuse_top_k": DEFAULT_FUSE_TOP_K,
            "outcome_weights_at_inference": 1.0,
            "latent_dim": vae_checkpoint["config"]["latent_dim"],
            "base_encoder": vae_checkpoint["config"]["base_model_name"],
        },
        "models": {
            "random": {
                "accuracy": random_accuracy,
                "params": None,
                "seconds": random_seconds,
            },
            "gpt2_small": {
                "accuracy": gpt2_accuracy,
                "accuracy_length_normalized": gpt2_accuracy_normalized,
                "longest_ending_picked_by_sum_rate": gpt2_length_bias,
                "params": GPT2_PARAMETERS,
                "params_quoted": GPT2_QUOTED_PARAMETERS,
                "model_name": GPT2_MODEL_NAME,
                "scoring": "sum of ending-token log probs (length_normalized variant also computed)",
                "seconds": gpt2_seconds,
            },
            "gmn": {
                "accuracy": gmn_accuracy,
                "params": gmn_parameters,
                "params_quoted": GMN_QUOTED_PARAMETERS,
                "params_breakdown": {
                    "vae_head": vae_head_parameters,
                    "vae_decoder": vae_decoder_parameters,
                    "base_encoder_frozen": "excluded",
                },
                "seconds": gmn_seconds,
                "ms_per_example": gmn_ms_per_example,
            },
        },
        "gmn_diagnostics": {
            "prediction_distribution": np.bincount(predictions_array, minlength=4).tolist(),
            "truth_distribution": np.bincount(labels_array, minlength=4).tolist(),
            "rank_of_correct_ending": {str(k): int(v) for k, v in sorted(rank_counts.items())},
            "top2_correct_rate": float((ranks <= 2).mean()),
            "mean_score_margin": float(margins.mean()),
        },
        "beats_random": bool(beats_random),
        "beats_gpt2": bool(beats_gpt2),
        "beats_gpt2_length_normalized": bool(beats_gpt2_fair),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
