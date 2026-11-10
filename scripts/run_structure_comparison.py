"""TASK 14 EXPERIMENT A/B: full 10,042-example ``no_navigation`` evaluation.

The 500-example geometry check has a standard error of ~2.2 points, so it cannot
resolve the differences at stake. This script evaluates the geometric
``no_navigation`` scorer (``lambda_nav = 0``, i.e. pure
``argmax_k cos(mu(ending_k), mu(ctx))``) on ALL 10,042 validation examples.

Baselines are the published cached numbers, so the comparison is against the same
examples the paper reports:
    original VAE (``checkpoints/vae.pt``)   31.31%   (0.31308504)
    GPT-2 small                             28.58%   (0.28579964)

Encoding note: ``experiments/additive_scores.npz`` holds the ORIGINAL model's
per-example scores, so nothing needs re-encoding for the baseline. The structured
models are different networks, so their val vectors are re-encoded here (and
cached per tag, so re-runs are free). The original model is ALSO re-encoded once
as a pipeline self-check: if ``--validate-original`` reproduces 31.31% exactly,
the batched encoder and the scorer agree with the published pipeline.

Batched encoding was verified equivalent to the per-example pipeline used by
``scripts/precompute_val_full_vectors.py``: max |delta| 3.4e-07 on the first 500
examples, 500/500 identical argmax.

Artifacts
---------
    checkpoints/structured_val_vecs_<tag>.npz   cached (N,256)/(N,4,256) vectors
    experiments/results_structure_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.autoencoder.vae import VAE  # noqa: E402
from src.memory.store import normalize_rows  # noqa: E402
from run_ablations import paired_comparison  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
ORIGINAL_SCORES_PATH = PROJECT_ROOT / "experiments" / "additive_scores.npz"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_structure_comparison.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
VEC_CACHE_DIR = CHECKPOINTS_DIR / "structured_val_vecs"

EMBED_BATCH_SIZE = 128
NUM_ENDINGS = 4
# Published cached baselines on all 10,042 val examples.
ORIGINAL_NO_NAV_10042 = 0.31308504282015537
GPT2_10042 = 0.28579964150567616
ORIGINAL_LABEL = "Original VAE no_nav"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 10,042 no_navigation eval for structured VAEs."
    )
    parser.add_argument("--tags", nargs="+", default=["v2", "v3"])
    parser.add_argument(
        "--validate-original",
        action="store_true",
        help="also re-encode checkpoints/vae.pt and check it reproduces 31.31%%",
    )
    parser.add_argument("--force-reencode", action="store_true")
    return parser.parse_args(argv)


def checkpoint_path_for(tag: str) -> Path:
    if tag == "original":
        return CHECKPOINTS_DIR / "vae.pt"
    return CHECKPOINTS_DIR / f"vae_structured_{tag}.pt"


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild a VAE from a checkpoint (same recipe as TASK 2/8/13)."""
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
    return vae, checkpoint


def encode_val_vectors(
    vae: VAE,
    ctx_texts: List[str],
    ending_texts: List[str],
    cache_path: Path,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit ``(N, 256)`` ctx vectors and ``(N, 4, 256)`` ending vectors, cached."""
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            if cached["queries"].shape[0] == len(ctx_texts):
                return cached["queries"], cached["endings"]
            print(f"  stale cache {cache_path.name} - re-encoding")
    started = time.perf_counter()
    with torch.no_grad():
        queries = normalize_rows(
            vae.encode_texts(ctx_texts, batch_size=EMBED_BATCH_SIZE).cpu().numpy()
        )
        endings = normalize_rows(
            vae.encode_texts(ending_texts, batch_size=EMBED_BATCH_SIZE).cpu().numpy()
        ).reshape(len(ctx_texts), NUM_ENDINGS, queries.shape[1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, queries=queries, endings=endings)
    elapsed = time.perf_counter() - started
    print(
        f"  encoded {len(ctx_texts) + len(ending_texts)} texts in {elapsed:.1f}s "
        f"({1000 * elapsed / (len(ctx_texts) + len(ending_texts)):.2f} ms/text) "
        f"-> {cache_path.name}"
    )
    return queries, endings


def scores_from_vectors(queries: np.ndarray, endings: np.ndarray) -> np.ndarray:
    """``cos(ending, ctx)`` for every (example, ending) -> ``(N, 4)``."""
    return np.einsum("nd,nkd->nk", queries, endings)


def geometry_from_scores(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Gap / accuracy / rank distribution for one model on the full val split."""
    total = scores.shape[0]
    rows = np.arange(total)
    cos_correct = scores[rows, labels]
    wrong_mask = np.ones(scores.shape, dtype=bool)
    wrong_mask[rows, labels] = False
    cos_wrong = scores[wrong_mask].reshape(total, NUM_ENDINGS - 1)
    ranks = 1 + (scores > cos_correct[:, None]).sum(axis=1)
    predictions = np.argmax(scores, axis=1)
    return {
        "num_examples": total,
        "accuracy": float((predictions == labels).mean()),
        "correct": int((predictions == labels).sum()),
        "mean_cos_correct": float(cos_correct.mean()),
        "mean_cos_wrong": float(cos_wrong.mean()),
        "val_gap": float(cos_correct.mean() - cos_wrong.mean()),
        "gap_std": float((cos_correct - cos_wrong.mean(axis=1)).std()),
        "rank_counts": {int(r): int((ranks == r).sum()) for r in range(1, NUM_ENDINGS + 1)},
        "predictions": [int(value) for value in predictions],
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
    args = parse_args()
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)
    total = len(val_data)
    labels = np.asarray([int(e["label"]) for e in val_data], dtype=np.int64)
    ctx_texts = [e["ctx"].strip() for e in val_data]
    ending_texts = [
        e["endings"][k].strip() for e in val_data for k in range(NUM_ENDINGS)
    ]
    device = torch.device("cpu")
    print(f"Full val split: {total} examples | {len(ending_texts)} ending texts")

    # ------------------------------------------------ cached baselines ----
    with GMN_PRED_PATH.open(encoding="utf-8") as handle:
        gmn = json.load(handle)
    with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
        gpt2 = json.load(handle)
    original_pred = [int(value) for value in gmn["no_navigation"]["predictions"]]
    gpt2_pred = [int(value) for value in gpt2["predictions"]]
    if len(original_pred) != total or len(gpt2_pred) != total:
        raise RuntimeError("cached prediction files do not cover the full val split")

    with np.load(ORIGINAL_SCORES_PATH) as cached_scores:
        context_scores = cached_scores["context_scores"]
        cached_labels = cached_scores["labels"].astype(np.int64)
    if not np.array_equal(cached_labels, labels):
        raise RuntimeError("additive_scores.npz labels disagree with data/val_full.json")
    original_geometry = geometry_from_scores(context_scores, labels)
    predictions_agree = np.array_equal(
        np.asarray(original_geometry["predictions"], dtype=np.int64),
        np.asarray(original_pred, dtype=np.int64),
    )
    original_acc = float(gmn["no_navigation"]["accuracy"])
    print(
        f"  cached original no_nav : {100 * original_acc:.4f}% "
        f"({gmn['no_navigation']['accuracy']:.8f}) | val gap "
        f"{original_geometry['val_gap']:.6f} | cos+ {original_geometry['mean_cos_correct']:.4f} "
        f"| cos- {original_geometry['mean_cos_wrong']:.4f}"
    )
    print(
        f"  cached scores' argmax matches cached no_nav predictions: {predictions_agree}"
    )
    print(f"  cached GPT-2 small     : {100 * GPT2_10042:.4f}%")

    models: Dict[str, Dict[str, Any]] = {
        ORIGINAL_LABEL: {
            "checkpoint": "checkpoints/vae.pt",
            "source": "cached experiments/additive_scores.npz + gmn_full_val_predictions.json",
            "margin": None,
            "wrong_mode": None,
            "accuracy": original_acc,
            "correct": original_geometry["correct"],
            "val_gap": original_geometry["val_gap"],
            "gap_std": original_geometry["gap_std"],
            "mean_cos_correct": original_geometry["mean_cos_correct"],
            "mean_cos_wrong": original_geometry["mean_cos_wrong"],
            "rank_counts": original_geometry["rank_counts"],
            "predictions": original_pred,
        }
    }

    # ------------------------------------------- structured models ----
    tags = list(args.tags)
    if args.validate_original:
        tags = tags + ["original"]
    for raw_tag in tags:
        checkpoint_path = checkpoint_path_for(raw_tag)
        if not checkpoint_path.exists():
            print(f"\n[{raw_tag}] SKIPPED - {checkpoint_path.name} not found")
            continue
        vae, checkpoint = load_vae(checkpoint_path, device)
        margin = checkpoint.get("structure_margin")
        wrong_mode = checkpoint.get("wrong_mode")
        if raw_tag == "original":
            label = "Original VAE (re-encoded)"
        else:
            label = f"Structured {raw_tag} (margin={margin:g})" if margin is not None else (
                f"Structured {raw_tag}"
            )
        print(f"\n[{raw_tag}] {label} | {checkpoint_path.name} | wrong_mode={wrong_mode}")
        queries, endings = encode_val_vectors(
            vae,
            ctx_texts,
            ending_texts,
            VEC_CACHE_DIR / f"{raw_tag}.npz",
            force=args.force_reencode,
        )
        del vae
        geometry = geometry_from_scores(scores_from_vectors(queries, endings), labels)
        models[label] = {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "source": "re-encoded on all 10,042 val examples (batched, cached)",
            "tag": raw_tag,
            "margin": margin,
            "wrong_mode": wrong_mode,
            "best_epoch": checkpoint.get("best_epoch"),
            "final_probe_gap": checkpoint.get("final_probe_gap"),
            "accuracy": geometry["accuracy"],
            "correct": geometry["correct"],
            "val_gap": geometry["val_gap"],
            "gap_std": geometry["gap_std"],
            "mean_cos_correct": geometry["mean_cos_correct"],
            "mean_cos_wrong": geometry["mean_cos_wrong"],
            "rank_counts": geometry["rank_counts"],
            "predictions": geometry["predictions"],
        }
        print(
            f"  no_navigation {100 * geometry['accuracy']:.4f}% "
            f"({geometry['correct']}/{total}) | val gap {geometry['val_gap']:.6f} "
            f"| cos+ {geometry['mean_cos_correct']:.4f} | cos- {geometry['mean_cos_wrong']:.4f}"
        )
        if raw_tag == "original":
            print(
                f"  self-check: reproduces cached no_nav exactly: "
                f"{abs(geometry['accuracy'] - original_acc) < 1e-9} "
                f"(delta {100 * (geometry['accuracy'] - original_acc):+.4f} points)"
            )

    # ---------------------------------------------------------- McNemar ----
    structured_labels = [label for label in models if label.startswith("Structured")]
    tag_to_label = {models[label]["tag"]: label for label in structured_labels}
    structured_tags = sorted(tag_to_label)
    mcnemar: Dict[str, Dict[str, Any]] = {}
    for tag in structured_tags:
        label = tag_to_label[tag]
        mcnemar[f"{tag}_vs_original"] = rename_pairs(
            paired_comparison(models[label]["predictions"], original_pred, labels.tolist()),
            tag,
            "original",
        )
    for index, first in enumerate(structured_tags):
        for second in structured_tags[index + 1 :]:
            mcnemar[f"{first}_vs_{second}"] = rename_pairs(
                paired_comparison(
                    models[tag_to_label[first]]["predictions"],
                    models[tag_to_label[second]]["predictions"],
                    labels.tolist(),
                ),
                first,
                second,
            )
    # Bonus (not requested): does a structured model still beat GPT-2 small?
    for tag in structured_tags:
        mcnemar[f"{tag}_vs_gpt2_small"] = rename_pairs(
            paired_comparison(
                models[tag_to_label[tag]]["predictions"], gpt2_pred, labels.tolist()
            ),
            tag,
            "gpt2_small",
        )

    # ------------------------------------------------------------- table ----
    original_row_acc = models[ORIGINAL_LABEL]["accuracy"]
    print("\n" + "=" * 92)
    print("FULL VALIDATION no_navigation ACCURACY (all 10,042 val examples)")
    print("=" * 92)
    header = (
        f"  {'Method':<34} | {'10,042 val':>10} | {'vs original':>11} | {'p-value':>10} "
        f"| {'val gap':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {ORIGINAL_LABEL:<34} | {100 * original_row_acc:>9.2f}% | {'baseline':>11} "
        f"| {'-':>10} | {models[ORIGINAL_LABEL]['val_gap']:>8.4f}"
    )
    for label in structured_labels:
        tag = models[label]["tag"]
        test = mcnemar.get(f"{tag}_vs_original", {})
        print(
            f"  {label:<34} | {100 * models[label]['accuracy']:>9.2f}% "
            f"| {100 * (models[label]['accuracy'] - original_row_acc):>+10.2f}% "
            f"| {test.get('mcnemar_p_value', float('nan')):>10.4f} "
            f"| {models[label]['val_gap']:>8.4f}"
        )
    print(
        f"  {'GPT-2 small (cached)':<34} | {100 * GPT2_10042:>9.2f}% "
        f"| {100 * (GPT2_10042 - original_row_acc):>+10.2f}% | {'cached':>10} | {'-':>8}"
    )
    print("  " + "-" * (len(header) - 2))

    print("\n" + "=" * 92)
    print("McNEMAR EXACT TESTS")
    print("=" * 92)
    for name, stats in mcnemar.items():
        a = f"{stats['first']}_correct_{stats['second']}_wrong"
        b = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(
            f"  {name:<24} | {a} {stats[a]:>5} | {b} {stats[b]:>5} "
            f"| discordant {stats['discordant_pairs']:>5} "
            f"| p = {stats['mcnemar_p_value']:.4f} "
            f"| significant {stats['significant_at_0.05']}"
        )

    print("\n" + "=" * 92)
    print("VAL GAP (does the training gap generalize to held-out data?)")
    print("=" * 92)
    print(
        f"  {'Method':<34} | {'cos_correct':>11} | {'cos_wrong':>9} | {'val gap':>8} "
        f"| {'gap std':>7}"
    )
    print("  " + "-" * 78)
    for label in [ORIGINAL_LABEL] + structured_labels:
        entry = models[label]
        print(
            f"  {label:<34} | {entry['mean_cos_correct']:>11.4f} "
            f"| {entry['mean_cos_wrong']:>9.4f} | {entry['val_gap']:>8.4f} "
            f"| {entry['gap_std']:>7.4f}"
        )
    print("  " + "-" * 78)

    # ------------------------------------------------------------ summary ----
    accuracies = {tag: models[tag_to_label[tag]]["accuracy"] for tag in structured_tags}
    gaps = {tag: models[tag_to_label[tag]]["val_gap"] for tag in structured_tags}
    significant = {
        tag: bool(mcnemar[f"{tag}_vs_original"]["significant_at_0.05"])
        for tag in structured_tags
    }
    wins = [tag for tag in structured_tags if accuracies[tag] > original_row_acc]
    if wins:
        best_tag = max(wins, key=lambda tag: (accuracies[tag], gaps[tag]))
    elif structured_tags:
        best_tag = max(structured_tags, key=lambda tag: accuracies[tag])
    else:
        best_tag = None
    best_margin = models[tag_to_label[best_tag]]["margin"] if best_tag else None

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    for tag in structured_tags:
        margin = models[tag_to_label[tag]]["margin"]
        print(
            f"  {tag} (margin={margin:g}): accuracy {100 * accuracies[tag]:.2f}% "
            f"| delta vs original {100 * (accuracies[tag] - original_row_acc):+.2f} points "
            f"| val gap {gaps[tag]:.4f} | p = "
            f"{mcnemar[f'{tag}_vs_original']['mcnemar_p_value']:.4f} "
            f"| significant {significant[tag]}"
        )
    if best_tag:
        print(f"\n  higher-accuracy margin : {best_tag} (margin={best_margin:g})")
    print(f"  either beats original significantly: {any(significant.values())}")

    payload = {
        "task": "TASK 14 EXPERIMENT A/B - full 10,042 no_navigation comparison",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": total,
        "config": {
            "scoring": "argmax_k cos(mu(ending_k), mu(ctx))  [geometric no_navigation, lambda=0]",
            "encoder_batching": EMBED_BATCH_SIZE,
            "batching_equivalence_check": (
                "max |delta| 3.4e-07 vs checkpoints/val_full_vec_chunks, 500/500 identical argmax"
            ),
        },
        "baselines_cached": {
            ORIGINAL_LABEL: original_acc,
            "gpt2_small": GPT2_10042,
            "original_val_gap_from_cached_scores": original_geometry["val_gap"],
            "cached_scores_argmax_matches_cached_predictions": bool(predictions_agree),
        },
        "models": models,
        "comparison": {
            "original_accuracy": original_row_acc,
            "gpt2_accuracy": GPT2_10042,
            "structured": {
                tag: {
                    "accuracy": accuracies[tag],
                    "delta_vs_original": accuracies[tag] - original_row_acc,
                    "val_gap": gaps[tag],
                    "margin": models[tag_to_label[tag]]["margin"],
                    "mcnemar_p_value_vs_original": mcnemar[f"{tag}_vs_original"][
                        "mcnemar_p_value"
                    ],
                    "significant_vs_original": significant[tag],
                }
                for tag in structured_tags
            },
            "best_by_accuracy": best_tag,
            "best_margin": best_margin,
            "any_significant_vs_original": bool(any(significant.values())),
        },
        "mcnemar": mcnemar,
        "notes": {
            "training_probe_gaps": {
                tag: models[tag_to_label[tag]]["final_probe_gap"] for tag in structured_tags
            },
            "cached_val_vectors": str(VEC_CACHE_DIR.relative_to(PROJECT_ROOT)),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    size_kb = RESULTS_PATH.stat().st_size / 1024
    print(f"\nSaved {RESULTS_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

