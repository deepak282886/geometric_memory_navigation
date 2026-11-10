"""TASK 13 step 3 / TASK 14 step 2: compare VAE latent geometries.

Encodes the SAME 500 validation examples with every available VAE and, for each
example, measures the cosine similarity between the context and each of the four
candidate endings, then ranks the correct ending.

Two encoding conventions are reported:

* ``ending_alone``      - ``cos(mu(ctx), mu(ending))``. This is exactly what the
  scorer in ``src/eval/evaluate.py`` computes, so its rank-1 rate *is* the
  ``no_navigation`` accuracy and is the headline number.
* ``ctx_plus_ending``   - ``cos(mu(ctx), mu(ctx + " " + ending))``, the text form
  the VAE was trained to reconstruct (TASK 2). Reported for completeness.

Models compared (missing files are skipped):
    1. ``checkpoints/vae.pt``              original VAE (recon + KL only)
    2. ``checkpoints/vae_structured.pt``   TASK 13 v1 (1 sampled wrong ending)
    3. ``--checkpoint``                    default ``vae_structured_v2.pt``
                                           TASK 14 v2 (ALL 3 wrong endings)

The original and v1 are re-measured on the same slice rather than quoted, so the
comparison is same-set and fair.

Default output: ``experiments/structured_geometry_v2.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.memory.store import normalize_rows  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
ORIGINAL_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
V1_PATH = PROJECT_ROOT / "checkpoints" / "vae_structured.pt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vae_structured_v2.pt"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "experiments" / "structured_geometry_v2.json"

NUM_EXAMPLES = 500
NUM_ENDINGS = 4
# Published honest no_navigation figure from experiments/results_honest.json
# (5,042-example eval split).
ORIGINAL_HONEST_NO_NAV = 0.2973026576755256
ORIGINAL_HONEST_EXAMPLES = 5042
# TASK 13 v1 references on the same 500 examples (from structured_geometry.json).
V1_RANK1_REFERENCE = 0.36
V1_GAP_STD_REFERENCE = 0.3107
# STEP 3 decision thresholds (rank-1 on the 500 examples).
PROCEED_THRESHOLD = 0.38
SMALLER_MARGIN_THRESHOLD = 0.37


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original / v1 / v2 VAE geometry on one val slice."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    return parser.parse_args(argv)


def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
    """Rebuild a VAE from a ``vae*.pt`` checkpoint (same recipe as TASK 2/8)."""
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
    return vae


def encode_example(
    vae: VAE, example: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ctx_vec (256,), ending_alone (4,256), ctx_plus_ending (4,256))`` unit-norm.

    Context is encoded alone and the endings four at a time, mirroring
    ``scripts/precompute_val_full_vectors.py`` so the numbers line up with the
    published evaluation.
    """
    ctx = example["ctx"].strip()
    endings = [ending.strip() for ending in example["endings"]]
    with torch.no_grad():
        ctx_vec = normalize_rows(vae.encode_texts([ctx]).cpu().numpy())[0]
        alone = normalize_rows(vae.encode_texts(endings).cpu().numpy())
        concat = normalize_rows(
            vae.encode_texts([f"{ctx} {ending}" for ending in endings]).cpu().numpy()
        )
    return ctx_vec, alone, concat


def collect_scores(
    vae: VAE, val_data: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(scores_alone (N,4), scores_concat (N,4), labels (N,))``."""
    total = len(val_data)
    scores_alone = np.zeros((total, NUM_ENDINGS), dtype=np.float64)
    scores_concat = np.zeros((total, NUM_ENDINGS), dtype=np.float64)
    labels = np.zeros(total, dtype=np.int64)
    for index, example in enumerate(val_data):
        ctx_vec, alone, concat = encode_example(vae, example)
        scores_alone[index] = alone @ ctx_vec
        scores_concat[index] = concat @ ctx_vec
        labels[index] = int(example["label"])
    return scores_alone, scores_concat, labels


def scores_to_geometry(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Cosine statistics and correct-ending rank distribution for one convention."""
    total = scores.shape[0]
    rows = np.arange(total)
    cos_correct = scores[rows, labels]
    wrong_mask = np.ones(scores.shape, dtype=bool)
    wrong_mask[rows, labels] = False
    cos_wrong_all = scores[wrong_mask].reshape(total, NUM_ENDINGS - 1)

    # rank 1 = best; rank = 1 + number of endings scoring strictly higher
    ranks = 1 + (scores > cos_correct[:, None]).sum(axis=1)
    rank_counts = {int(rank): int((ranks == rank).sum()) for rank in range(1, NUM_ENDINGS + 1)}
    predictions = np.argmax(scores, axis=1)

    return {
        "mean_cos_correct": float(cos_correct.mean()),
        "mean_cos_wrong_all3": float(cos_wrong_all.mean()),
        "gap_all3": float(cos_correct.mean() - cos_wrong_all.mean()),
        "cos_correct_std": float(cos_correct.std()),
        "gap_std": float((cos_correct - cos_wrong_all.mean(axis=1)).std()),
        "rank_counts": rank_counts,
        "rank_percentages": {
            f"rank_{rank}": 100.0 * count / total for rank, count in rank_counts.items()
        },
        "rank1_accuracy": float((predictions == labels).mean()),
        "rank1_correct": int((predictions == labels).sum()),
        "num_examples": total,
    }


def print_geometry(title: str, geometry: Dict[str, Any]) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 58)
    print(f"    Mean cos(ctx, correct):  {geometry['mean_cos_correct']:.4f}")
    print(f"    Mean cos(ctx, wrong):    {geometry['mean_cos_wrong_all3']:.4f}")
    print(f"    Gap:                     {geometry['gap_all3']:.4f}")
    percentages = geometry["rank_percentages"]
    for rank in range(1, NUM_ENDINGS + 1):
        suffix = "   <- no_navigation accuracy" if rank == 1 else ""
        print(f"    Correct ending rank {rank}:   {percentages[f'rank_{rank}']:.1f}%{suffix}")
    print(
        f"    (rank1 {geometry['rank1_correct']}/{geometry['num_examples']}, "
        f"per-example gap std {geometry['gap_std']:.4f})"
    )


def model_label(checkpoint: Dict[str, Any], path: Path, fallback: Tuple[str, str]) -> str:
    """Label used in the table: 'Original VAE' / 'Structured v1 (1x)' / '... v2 (3x)'."""
    if path == ORIGINAL_PATH:
        return "Original VAE"
    tag = checkpoint.get("tag", fallback[0])
    mode = checkpoint.get("wrong_mode", fallback[1])
    return f"Structured {tag} ({'3x' if mode == 'all' else '1x'})"


def main() -> None:
    args = parse_args()
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)
    examples = val_data[: args.num_examples]
    labels = np.asarray([int(e["label"]) for e in examples], dtype=np.int64)
    device = torch.device("cpu")

    print(f"Geometry check on the first {len(examples)} examples of {VAL_FULL_PATH.name}")
    print(f"label distribution: {np.bincount(labels, minlength=NUM_ENDINGS).tolist()}")

    # (path, (fallback_tag, fallback_wrong_mode)); the original has no structure info
    candidates = [
        (ORIGINAL_PATH, ("original", "none")),
        (V1_PATH, ("v1", "one")),
        (Path(args.checkpoint), ("v2", "all")),
    ]
    seen: set = set()
    specs: List[Dict[str, Any]] = []
    for path, fallback in candidates:
        if not path.exists():
            print(f"  (skipping missing checkpoint {path})")
            continue
        if path in seen:
            continue
        seen.add(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        specs.append(
            {
                "label": model_label(checkpoint, path, fallback),
                "path": path,
                "margin": checkpoint.get("structure_margin"),
                "wrong_mode": checkpoint.get("wrong_mode"),
                "best_epoch": checkpoint.get("best_epoch"),
                "checkpoint": checkpoint,
            }
        )

    results: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        print(f"\nEncoding {len(examples)} examples with {spec['path'].name} ...", flush=True)
        vae = load_vae(spec["path"], device)
        alone, concat, _ = collect_scores(vae, examples)
        del vae
        snapshot_keys = (
            "final_probe_gap",
            "final_probe1_gap",
            "final_probe_per_example_gap_std",
            "final_probe_mean_cos_correct",
            "final_probe_mean_cos_wrong",
        )
        results[spec["label"]] = {
            "checkpoint": str(spec["path"].relative_to(PROJECT_ROOT)),
            "margin": spec["margin"],
            "wrong_mode": spec["wrong_mode"],
            "best_epoch": spec["best_epoch"],
            "ending_alone": scores_to_geometry(alone, labels),
            "ctx_plus_ending": scores_to_geometry(concat, labels),
            "training_snapshot": {
                key: spec["checkpoint"].get(key) for key in snapshot_keys
            },
        }
        geometry = results[spec["label"]]
        print("\n" + "=" * 70)
        print(f"{spec['label']}  ({spec['path'].name}, margin={spec['margin']})")
        print("=" * 70)
        print_geometry(
            "scorer convention: cos(ctx, ending encoded ALONE)", geometry["ending_alone"]
        )
        print_geometry(
            "training convention: cos(ctx, ctx + ending)", geometry["ctx_plus_ending"]
        )

    # ------------------------------------------------------------- table
    print("\n" + "=" * 82)
    print(f"COMPARISON - scorer convention, same {len(examples)} val examples")
    print("=" * 82)
    print(
        f"  {'Method':<22} | {'cos_correct':>11} | {'cos_wrong':>9} | {'gap':>7} "
        f"| {'gap_std':>7} | {'rank_1':>7}"
    )
    print("  " + "-" * 78)
    for label, geometry in results.items():
        geo = geometry["ending_alone"]
        print(
            f"  {label:<22} | {geo['mean_cos_correct']:>11.4f} "
            f"| {geo['mean_cos_wrong_all3']:>9.4f} | {geo['gap_all3']:>7.4f} "
            f"| {geo['gap_std']:>7.4f} | {100 * geo['rank1_accuracy']:>6.1f}%"
        )
    print("  " + "-" * 78)

    original_acc = results["Original VAE"]["ending_alone"]["rank1_accuracy"]
    original_std = results["Original VAE"]["ending_alone"]["gap_std"]
    target_label = specs[-1]["label"]
    target_geo = results[target_label]["ending_alone"]
    target_acc = target_geo["rank1_accuracy"]

    print("\n" + "=" * 82)
    print("KEY QUESTION + STEP 3 DECISION")
    print("=" * 82)
    print(f"  original rank-1 (same set)    : {100 * original_acc:.1f}%")
    print(f"  v1 rank-1 (TASK 13 reference) : {100 * V1_RANK1_REFERENCE:.1f}%")
    print(f"  {target_label} rank-1        : {100 * target_acc:.1f}%")
    print(f"  target vs original            : {100 * (target_acc - original_acc):+.1f} points")
    print(f"  per-example gap std           : target {target_geo['gap_std']:.4f} | "
          f"v1 {V1_GAP_STD_REFERENCE:.4f} | original {original_std:.4f}")
    print(f"  gap std lower than v1         : "
          f"{target_geo['gap_std'] < V1_GAP_STD_REFERENCE}  (TASK 14 target)")
    if target_acc >= PROCEED_THRESHOLD:
        decision = "PROCEED to full 10,042 evaluation (TASK 14)"
    elif target_acc < SMALLER_MARGIN_THRESHOLD:
        decision = "RETRY with margin=0.3 (less saturation)"
    else:
        decision = "BORDERLINE (37-38%): report and wait for confirmation"
    print(f"  rank-1 >= {100 * PROCEED_THRESHOLD:.0f}% -> full eval   : "
          f"{target_acc >= PROCEED_THRESHOLD}")
    print(f"  rank-1 <  {100 * SMALLER_MARGIN_THRESHOLD:.0f}% -> margin 0.3: "
          f"{target_acc < SMALLER_MARGIN_THRESHOLD}")
    print(f"  DECISION                      : {decision}")

    payload = {
        "task": "TASK 14 STEP 2 - structured VAE v2 geometry check",
        "val_file": str(VAL_FULL_PATH.relative_to(PROJECT_ROOT)),
        "num_examples": len(examples),
        "label_distribution": np.bincount(labels, minlength=NUM_ENDINGS).tolist(),
        "conventions": {
            "ending_alone": "cos(mu(ctx), mu(ending)) - what src/eval/evaluate.py scores",
            "ctx_plus_ending": "cos(mu(ctx), mu(ctx + ' ' + ending)) - TASK 2 training text",
        },
        "thresholds": {
            "proceed": PROCEED_THRESHOLD,
            "try_smaller_margin": SMALLER_MARGIN_THRESHOLD,
        },
        "original_honest_no_navigation_5042": ORIGINAL_HONEST_NO_NAV,
        "v1_references_same_500": {
            "rank1": V1_RANK1_REFERENCE,
            "gap_std": V1_GAP_STD_REFERENCE,
        },
        "models": results,
        "comparison": {
            "target": target_label,
            "original_rank1_same_set": original_acc,
            "target_rank1": target_acc,
            "delta_vs_original": target_acc - original_acc,
            "target_gap_std": target_geo["gap_std"],
            "gap_std_lower_than_v1": bool(target_geo["gap_std"] < V1_GAP_STD_REFERENCE),
            "beats_original": bool(target_acc > original_acc),
            "decision": decision,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
