"""EXPERIMENT: does a 39,905-example memory store beat the 2,000-example one?

Runs the same geometric ending scoring on the same 500 validation examples,
changing only the store size and ``k``. No retraining: the VAE, sampler and
fusion are untouched.

Writes ``experiments/results_full.json`` and, for the best configuration,
``experiments/gmn_val_predictions_full.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.eval.evaluate import (  # noqa: E402
    DEFAULT_FUSE_TOP_K,
    evaluate_gmn,
    score_val_set,
)
from src.memory.store import MemoryStore  # noqa: E402

VAL_PATH = PROJECT_ROOT / "data" / "val.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
FULL_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full.pkl"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "experiments" / "results.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_full.json"
PREDICTIONS_PATH = PROJECT_ROOT / "experiments" / "gmn_val_predictions_full.json"

#: required by the experiment
K_VALUES: Tuple[int, ...] = (10, 20)
#: cheap extras (encoding dominates the per-example cost), to pick a default
EXTRA_K_VALUES: Tuple[int, ...] = (5, 50)

BASELINE_STORE_SIZE = 2000


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config (same pattern as TASK 3-8)."""
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
    with BASELINE_RESULTS_PATH.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    baseline_accuracy = float(baseline["models"]["gmn"]["accuracy"])
    gpt2_accuracy = float(baseline["models"]["gpt2_small"]["accuracy"])

    vae, _checkpoint = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    store = MemoryStore.load(FULL_STORE_PATH)

    print(f"\nval examples {len(val_data)} | full store {store.size} vectors "
          f"(vs {BASELINE_STORE_SIZE} in the TASK 8 baseline)")
    print(f"baseline from results.json : {100 * baseline_accuracy:.1f}% "
          f"(store {BASELINE_STORE_SIZE}, k=10)")
    print(f"GPT-2 small reference      : {100 * gpt2_accuracy:.1f}%")

    print("\nRunning geometric ending scoring on the full store ...")
    rows: List[Dict[str, Any]] = []
    for k in sorted(set(K_VALUES) | set(EXTRA_K_VALUES)):
        started = time.perf_counter()
        accuracy = evaluate_gmn(val_data, vae, store, predictions_path=None, k=k)
        seconds = time.perf_counter() - started
        required = k in K_VALUES
        rows.append(
            {
                "k": k,
                "accuracy": accuracy,
                "seconds": seconds,
                "ms_per_example": 1000.0 * seconds / max(len(val_data), 1),
                "required": required,
            }
        )
        print(f"  store {store.size:>5d} k={k:>2d} -> {100 * accuracy:>5.1f}%"
              f"  ({seconds:.1f}s){'' if required else '   (extra)'}")

    # ------------------------------------------------------------- table
    baseline_k = int(baseline["gmn_config"]["retrieval_k"])
    required_rows = {row["k"]: row for row in rows if row["required"]}
    best_required = max(required_rows.values(), key=lambda row: row["accuracy"])
    best_any = max(rows, key=lambda row: row["accuracy"])

    print("\n" + "=" * 52)
    print("STORE SIZE SWEEP (500 HellaSwag val examples)")
    print("=" * 52)
    header = f"{'Store size':<11} | {'k':>3} | GMN accuracy"
    print(header)
    print("-" * len(header))
    print(f"{BASELINE_STORE_SIZE:>10,} | {baseline_k:>3} | {100 * baseline_accuracy:>5.1f}%"
          f"  (baseline, TASK 8)")
    for k in K_VALUES:
        row = required_rows[k]
        print(f"{store.size:>10,} | {k:>3} | {100 * row['accuracy']:>5.1f}%")
    print("-" * len(header))

    extras = [row for row in rows if not row["required"]]
    if extras:
        print("extra k values on the full store (not required):")
        for row in extras:
            print(f"{store.size:>10,} | {row['k']:>3} | {100 * row['accuracy']:>5.1f}%")

    beats_2000 = best_required["accuracy"] > baseline_accuracy
    beats_gpt2 = best_required["accuracy"] > gpt2_accuracy
    delta_2000 = best_required["accuracy"] - baseline_accuracy
    delta_gpt2 = best_required["accuracy"] - gpt2_accuracy

    print(f"\n  best required config : k={best_required['k']} -> "
          f"{100 * best_required['accuracy']:.1f}%")
    print(f"  best over all k tried: k={best_any['k']} -> {100 * best_any['accuracy']:.1f}%")
    print(f"  vs 2000-store GMN ({100 * baseline_accuracy:.1f}%): beats={beats_2000} "
          f"({100 * delta_2000:+.1f} points)")
    print(f"  vs GPT-2 small ({100 * gpt2_accuracy:.1f}%)   : beats={beats_gpt2} "
          f"({100 * delta_gpt2:+.1f} points)")

    if beats_gpt2:
        decision = "PROCEED to TASK 9 with the FULL store (default)"
    elif best_required["accuracy"] < 0.33:
        decision = "PROCEED to TASK 9 with the 2000 store (story: 199x smaller, matches GPT-2 small)"
    else:
        decision = "between the two thresholds - your call"
    print(f"\n  DECISION GATE -> {decision}")

    # ------------------------------------------- best config predictions
    detail = score_val_set(val_data, vae, store, k=best_any["k"], fuse_top_k=DEFAULT_FUSE_TOP_K)
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(detail, handle, indent=2)
    print(f"\n  best-config predictions -> {PREDICTIONS_PATH.name} "
          f"(k={best_any['k']}, accuracy {100 * detail['accuracy']:.1f}%)")

    # -------------------------------------------------------------- save
    results = {
        "experiment": "full training-set memory store",
        "dataset": "hellaswag",
        "split": "validation",
        "val_examples": len(val_data),
        "seed": 42,
        "gmn_config": {
            "scoring": "cosine(L2(encode(ending)), L2(query_vector + nav_vector))",
            "classifier_head": "none",
            "fuse_top_k": DEFAULT_FUSE_TOP_K,
            "outcome_weights_at_inference": 1.0,
        },
        "stores": {
            "baseline_2000": {
                "store_size": BASELINE_STORE_SIZE,
                "k": baseline_k,
                "accuracy": baseline_accuracy,
                "source": "experiments/results.json (TASK 8)",
            },
            "full": {
                "store_size": store.size,
                "source": "checkpoints/memory_store_full.pkl",
                "k_sweep": {
                    str(row["k"]): {
                        "accuracy": row["accuracy"],
                        "seconds": row["seconds"],
                        "ms_per_example": row["ms_per_example"],
                        "required": row["required"],
                    }
                    for row in rows
                },
                "best_required": {
                    "k": best_required["k"],
                    "accuracy": best_required["accuracy"],
                },
                "best_overall": {
                    "k": best_any["k"],
                    "accuracy": best_any["accuracy"],
                },
            },
        },
        "gpt2_small_accuracy": gpt2_accuracy,
        "full_store_beats_2000_store": bool(beats_2000),
        "full_store_beats_gpt2": bool(beats_gpt2),
        "decision": decision,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("Full-store experiment complete.")


if __name__ == "__main__":
    main()
