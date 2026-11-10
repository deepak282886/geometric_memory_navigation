"""STEP 0 verification on Paper 1's own metric: the HellaSwag ctx-only score gap.

The task quotes the original VAE's gap as **0.0195**. That number is not a WikiText
next-sentence gap - it is Paper 1's ``score_gap_ctx`` on HellaSwag validation
(``mean cos(ctx, correct ending)`` minus ``mean cos(ctx, wrong endings)`` with
``lambda_nav = 0``), recorded in ``experiments/navigation_diagnosis.json`` and
``experiments/results_continuation.json`` as 0.019494308728662313. This script
measures exactly that quantity for the original and the continuation head, so the
"is the gap wider?" question is answered in the units the task used.

It is cheap because ``checkpoints/val_full_base_embeddings.npz`` already holds the
frozen MiniLM embeddings of all 10,042 HellaSwag val ctxs and their 4 endings
(ctx ``(10042, 384)``, endings ``(10042, 4, 384)``, labels ``(10042,)``) - only
the VAE head has to be run.

Self-check: the original checkpoint must reproduce both published numbers
(score gap 0.0194943 and no-navigation accuracy 31.308%), otherwise the harness
disagrees with the pipeline that produced the paper and nothing here can be
trusted.

Note on fairness: this is the one metric where the ORIGINAL head has a domain
advantage (it was trained on HellaSwag *train* text) while the continuation head
was fine-tuned on WikiText pairs only and has never seen HellaSwag. A wider gap
here is therefore cross-domain transfer, and a narrower one means the WikiText
geometry does not transfer to HellaSwag ending discrimination.

Artifacts
---------
    experiments/hellaswag_encoder_gap.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402

BASE_CACHE = PROJECT_ROOT / "checkpoints" / "val_full_base_embeddings.npz"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "hellaswag_encoder_gap.json"
DEFAULT_CHECKPOINTS = [
    f"Original (vae.pt)={PROJECT_ROOT / 'checkpoints' / 'vae.pt'}",
    f"Continuation (vae_continuation.pt)="
    f"{PROJECT_ROOT / 'checkpoints' / 'vae_continuation.pt'}",
]

#: published Paper 1 numbers this harness must reproduce for the original VAE
PUBLISHED_ORIGINAL_SCORE_GAP = 0.019494308728662313
PUBLISHED_ORIGINAL_NO_NAV_ACCURACY = 0.31308504282015537
NUM_ENDINGS = 4
CHUNK = 2048
SEED = 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HellaSwag ctx-only score gap for each encoder checkpoint."
    )
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--base-cache", type=Path, default=BASE_CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=0, help="first N examples (0 = all)")
    parser.add_argument(
        "--indices-from",
        type=Path,
        default=None,
        help=(
            "restrict to the example indices stored in this JSON (e.g. "
            "experiments/navigation_diagnosis.json's sampled_val_indices) so the "
            "number is comparable with a published diagnosis slice"
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
    """Rebuild a VAE from a checkpoint (identical recipe to the other VAEs)."""
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


def parse_checkpoints(entries: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for entry in entries:
        if "=" in entry:
            label, raw = entry.split("=", 1)
        else:
            raw, label = entry, Path(entry).stem
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        parsed.append((label.strip(), path))
    return parsed


@torch.no_grad()
def scores_for(
    vae: VAE, base_ctx: torch.Tensor, base_endings: torch.Tensor
) -> np.ndarray:
    """``(N, 4)`` cosine scores ``cos(mu(ctx), mu(ending_k))``, chunked, no grad.

    ``mu`` (not sampled ``z``) and L2 normalisation, exactly like
    ``src/eval/evaluate.py``'s ``no_navigation`` scorer and Paper 1's diagnostics.
    """
    total = int(base_ctx.shape[0])
    endings_per_example = int(base_endings.shape[1])
    scores = np.zeros((total, endings_per_example), dtype=np.float64)
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        ctx_mu, _ = vae.head(base_ctx[start:stop])
        flat_endings = base_endings[start:stop].reshape(-1, base_endings.shape[-1])
        end_mu, _ = vae.head(flat_endings)
        ctx_unit = F.normalize(ctx_mu, dim=-1)
        end_unit = F.normalize(end_mu, dim=-1).reshape(stop - start, endings_per_example, -1)
        scores[start:stop] = torch.einsum("nd,nkd->nk", ctx_unit, end_unit).cpu().numpy()
    return scores


def paired_stats(better: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    """Paired per-example deltas ``better - reference`` with wins and a t-statistic."""
    delta = np.asarray(better, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    total = int(delta.size)
    std = float(delta.std(ddof=1)) if total > 1 else 0.0
    standard_error = std / float(np.sqrt(total)) if total else 0.0
    return {
        "mean_better": float(np.mean(better)),
        "mean_reference": float(np.mean(reference)),
        "mean_delta": float(delta.mean()),
        "delta_std": std,
        "paired_t_statistic": float(delta.mean() / standard_error) if standard_error > 0 else 0.0,
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "share_wins": float((delta > 0).mean()) if total else 0.0,
    }



def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    set_seed(SEED)

    print("=" * 84)
    print("STEP 0 VERIFICATION - HellaSwag ctx-only score gap (Paper 1's metric)")
    print("=" * 84)

    specs = parse_checkpoints(args.checkpoints)
    for label, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path} does not exist")

    with np.load(args.base_cache, allow_pickle=False) as cached:
        base_ctx = cached["ctx"].astype(np.float32)
        base_endings = cached["endings"].astype(np.float32)
        labels = cached["labels"].astype(np.int64)
    subset_indices: Optional[np.ndarray] = None
    if args.indices_from is not None:
        with args.indices_from.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        for key in ("sampled_val_indices", "val_indices", "pair_indices", "indices"):
            if key in source:
                subset_indices = np.asarray(source[key], dtype=np.int64)
                break
        if subset_indices is None:
            raise KeyError(
                f"{args.indices_from} has none of the keys "
                "sampled_val_indices/val_indices/pair_indices/indices"
            )
        base_ctx = base_ctx[subset_indices]
        base_endings = base_endings[subset_indices]
        labels = labels[subset_indices]
        print(f"restricted to {subset_indices.size} indices from {args.indices_from}")
    if args.limit:
        base_ctx = base_ctx[: args.limit]
        base_endings = base_endings[: args.limit]
        labels = labels[: args.limit]
    print(
        f"cached base embeddings: ctx {base_ctx.shape} | endings {base_endings.shape} "
        f"| labels {labels.shape}"
    )
    ctx_tensor = torch.from_numpy(np.ascontiguousarray(base_ctx))
    endings_tensor = torch.from_numpy(np.ascontiguousarray(base_endings))

    rows = np.arange(labels.shape[0])
    results: Dict[str, Any] = {}
    per_example_gaps: Dict[str, np.ndarray] = {}
    for label, path in specs:
        vae = load_vae(path, device)
        scores = scores_for(vae, ctx_tensor, endings_tensor)
        cos_correct = scores[rows, labels]
        wrong_mask = np.ones(scores.shape, dtype=bool)
        wrong_mask[rows, labels] = False
        cos_wrong = scores[wrong_mask].reshape(labels.shape[0], NUM_ENDINGS - 1)
        per_example = cos_correct - cos_wrong.mean(axis=1)
        predictions = scores.argmax(axis=1)
        results[label] = {
            "mean_cos_correct": float(cos_correct.mean()),
            "mean_cos_wrong": float(cos_wrong.mean()),
            "score_gap_ctx": float(per_example.mean()),
            "score_gap_ctx_std": float(per_example.std()),
            "no_navigation_accuracy": float((predictions == labels).mean()),
            "correct": int((predictions == labels).sum()),
            "num_examples": int(labels.shape[0]),
        }
        per_example_gaps[label] = per_example

    reference_label = specs[0][0]
    comparison: Dict[str, Any] = {}
    for label, _ in specs[1:]:
        comparison[label] = {
            "vs": reference_label,
            "score_gap_ctx": paired_stats(
                per_example_gaps[label], per_example_gaps[reference_label]
            ),
            "no_navigation_accuracy": results[label]["no_navigation_accuracy"],
            "accuracy_delta": (
                results[label]["no_navigation_accuracy"]
                - results[reference_label]["no_navigation_accuracy"]
            ),
        }

    print("\n  model                          | cos(correct) | cos(wrong) |  score gap | no-nav acc")
    print("  " + "-" * 92)
    for label, metrics in results.items():
        print(
            f"  {label:<30} | {metrics['mean_cos_correct']:>12.4f} | "
            f"{metrics['mean_cos_wrong']:>10.4f} | "
            f"{metrics['score_gap_ctx']:>10.6f} | {100 * metrics['no_navigation_accuracy']:>8.2f}%"
        )
    print("  " + "-" * 92)

    original = results[reference_label]
    # the published gap (0.019494308728662313) is the 200-example diagnosis slice
    # (seed 42, experiments/navigation_diagnosis.json), the published accuracy
    # (0.31308504282015537) is the full 10,042-example split - so each is only
    # expected to match on its own set. The tolerance covers float32 head outputs
    # being scored in float64.
    full_set = subset_indices is None and not args.limit
    reproduces_gap = (
        abs(original["score_gap_ctx"] - PUBLISHED_ORIGINAL_SCORE_GAP) <= 1e-6
    )
    reproduces_acc = full_set and (
        abs(original["no_navigation_accuracy"] - PUBLISHED_ORIGINAL_NO_NAV_ACCURACY) <= 1e-9
    )
    print("\n  SELF-CHECK (the original VAE must reproduce Paper 1)")
    print(
        f"    score gap  this harness {original['score_gap_ctx']:.9f} | published "
        f"{PUBLISHED_ORIGINAL_SCORE_GAP:.9f} (200-example diagnosis slice) "
        f"-> {reproduces_gap}"
    )
    print(
        f"    no-nav acc this harness {original['no_navigation_accuracy']:.9f} | published "
        f"{PUBLISHED_ORIGINAL_NO_NAV_ACCURACY:.9f} (full 10,042) -> {reproduces_acc}"
        + ("" if full_set else "   [not applicable: this run uses a subset]")
    )

    print("\n  paired deltas vs the original (positive = wider gap)")
    for label, stats in comparison.items():
        gap_stats = stats["score_gap_ctx"]
        print(
            f"    {label:<30} gap {gap_stats['mean_reference']:.6f} -> "
            f"{gap_stats['mean_better']:.6f} ({gap_stats['mean_delta']:+.6f}, "
            f"{100 * gap_stats['share_wins']:.1f}% of examples up, "
            f"t {gap_stats['paired_t_statistic']:+.2f}) | no-nav acc "
            f"{100 * stats['no_navigation_accuracy']:.2f}% ({100 * stats['accuracy_delta']:+.2f})"
        )

    payload = {
        "task": "STEP 0 verification - HellaSwag ctx-only score gap",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "metric": (
            "mean cos(ctx, correct ending) - mean cos(ctx, wrong endings), lambda_nav = 0 "
            "(same quantity as Paper 1's score_gap_ctx)"
        ),
        "base_cache": str(args.base_cache),
        "num_examples": int(labels.shape[0]),
        "slice": {
            "indices_from": str(args.indices_from) if args.indices_from is not None else None,
            "num_indices": int(subset_indices.size) if subset_indices is not None else None,
            "indices": (
                [int(value) for value in subset_indices]
                if subset_indices is not None
                else None
            ),
        },
        "published_original": {
            "score_gap_ctx": PUBLISHED_ORIGINAL_SCORE_GAP,
            "no_navigation_accuracy": PUBLISHED_ORIGINAL_NO_NAV_ACCURACY,
        },
        "self_check": {
            "reproduces_score_gap": bool(reproduces_gap),
            "reproduces_no_navigation_accuracy": bool(reproduces_acc),
            "note": (
                "published score_gap_ctx is the 200-example diagnosis slice; published "
                "no_navigation_accuracy is the full 10,042-example val split"
            ),
        },
        "models": results,
        "paired_vs_original": comparison,
        "fairness_note": (
            "the original head was trained on HellaSwag train text, the continuation "
            "head on WikiText pairs only; a wider gap here is cross-domain transfer"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved {args.output}")
    print("=" * 84)
    return payload


if __name__ == "__main__":
    main()

