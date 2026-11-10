"""STEP 0 verification: did the continuation head actually improve the geometry?

The claim to check (exactly the check in the task):

    for each sampled WikiText pair (S_n, S_n+1)
        cos(encode(S_n), encode(S_n+1))   should be HIGHER
        cos(encode(S_n), random sentence) should be LOWER
    so the gap between them should be WIDER than the original VAE's

Measured for every checkpoint passed in, on the SAME sampled pairs and with the
SAME random negatives, so the comparison is paired per example.

Metrics per model
-----------------
    cos_positive        mean cos(anchor, its own next sentence)
    cos_negative        mean cos(anchor, random sentence), averaged over
                        ``--negatives`` draws per pair (10 by default, so the
                        number is not one lucky draw)
    gap                 cos_positive - cos_negative          <- the headline
    cos_random_single   the task's literal "random sentence": one draw per pair
    gap_random_single   cos_positive - cos_random_single
    retrieval_top1_pool is the true next sentence the best of the 200 sampled
                        positives for each anchor? (a negative pool 6x larger
                        than the batch-32 pool used during training)

Held-out control
----------------
``--heldout-from N`` samples pairs with index >= N, i.e. pairs a control arm
trained on the first N pairs has never seen. The same script then answers
"is the geometry better, or is it memorised?" - the confound that matters,
because the standard arm trains on exactly the 7,666 pairs that TASK 27
subsequently predicts over.

Artifacts
---------
    experiments/continuation_geometry.json            (or --output)
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

PAIRS_PATH = PROJECT_ROOT / "environment" / "data" / "wikitext" / "val_pairs.json"
BASE_CACHE_PATH = PROJECT_ROOT / "checkpoints" / "wikitext_val_base_embeddings.npz"
DEFAULT_CHECKPOINTS = [
    f"Original (vae.pt)={PROJECT_ROOT / 'checkpoints' / 'vae.pt'}",
    f"Continuation (vae_continuation.pt)="
    f"{PROJECT_ROOT / 'checkpoints' / 'vae_continuation.pt'}",
]
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "continuation_geometry.json"

SAMPLE_SIZE = 200
NEGATIVES_PER_PAIR = 10
SEED = 0
#: gap quoted for the original VAE before this step (recorded for traceability;
#: every number in the JSON is re-measured on the same sample, never quoted)
REFERENCE_ORIGINAL_GAP = 0.0195
EMBED_BATCH_SIZE = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired geometry check: does the continuation head widen the gap?"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="LABEL=path entries; paths may be relative to the repo root",
    )
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--base-cache", type=Path, default=BASE_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--negatives", type=int, default=NEGATIVES_PER_PAIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--heldout-from",
        type=int,
        default=0,
        help="sample pairs with index >= this value (0 = any pair)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force-reencode", action="store_true")
    return parser.parse_args(argv)


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
    """``LABEL=path`` -> ``[(label, absolute path)]`` (bare paths get the stem)."""
    parsed: List[Tuple[str, Path]] = []
    for entry in entries:
        if "=" in entry:
            label, raw = entry.split("=", 1)
        else:
            raw = entry
            label = Path(entry).stem
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        parsed.append((label.strip(), path))
    return parsed


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [(str(first), str(second)) for first, second in raw]


def load_or_build_base_embeddings(
    vae: VAE,
    sentences: Sequence[str],
    cache_path: Path,
    force_reencode: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Shared frozen-base cache, identical to the one the STEP 0 trainer writes."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_reencode:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_sentences = [str(value) for value in cached["sentences"]]
            if cached_sentences == list(sentences):
                return cached["embeddings"].astype(np.float32)
        print(f"cache {cache_path} does not match the pair file - re-embedding")
    print(f"embedding {len(sentences)} unique sentences with the frozen base encoder ...")
    with torch.no_grad():
        embeddings = vae.embed(list(sentences), batch_size=batch_size).cpu().numpy()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, sentences=np.asarray(list(sentences)), embeddings=embeddings)
    print(f"  cached {embeddings.shape} base embeddings to {cache_path}")
    return embeddings


@torch.no_grad()
def unit_mu(vae: VAE, base_embeddings: torch.Tensor) -> np.ndarray:
    """``mu`` of the VAE head for base embeddings -> ``(N, 256)`` unit-norm rows."""
    mu, _ = vae.head(base_embeddings)
    return F.normalize(mu, dim=-1).cpu().numpy()


def paired_stats(better: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    """Paired per-example deltas ``better - reference`` with wins and a t-statistic.

    Same caveat as elsewhere in this project: WikiText pairs are ordered by
    document, so adjacent pairs share sentences and topics; treat the t-statistic
    as an effect-size guide rather than an exact p-value.
    """
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
        "ties": int((delta == 0).sum()),
        "share_wins": float((delta > 0).mean()) if total else 0.0,
    }



# ---------------------------------------------------------------------------
# Geometry measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def model_geometry(
    vae: VAE,
    base_embeddings: np.ndarray,
    anchor_rows: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Per-model geometry for the sampled pairs -> ``(metrics, per-pair arrays)``.

    All vectors are ``mu``, L2-normalised - exactly what TASK 27's predictor and
    Paper 1's scorer consume.
    """
    latent = unit_mu(vae, torch.from_numpy(np.ascontiguousarray(base_embeddings)))
    anchors = latent[anchor_rows].astype(np.float64)
    positives = latent[positive_rows].astype(np.float64)
    negatives = latent[negative_rows].astype(np.float64)  # (P, K, 256)

    cos_positive = np.einsum("pd,pd->p", anchors, positives)
    cos_negative_all = np.einsum("pd,pkd->pk", anchors, negatives)  # (P, K)
    cos_negative = cos_negative_all.mean(axis=1)
    cos_random_single = cos_negative_all[:, 0]
    gap = cos_positive - cos_negative
    gap_random_single = cos_positive - cos_random_single

    scores = anchors @ positives.T
    retrieval_top1 = float((scores.argmax(axis=1) == np.arange(scores.shape[0])).mean())

    metrics = {
        "cos_positive": float(cos_positive.mean()),
        "cos_positive_std": float(cos_positive.std()),
        "cos_negative": float(cos_negative_all.mean()),
        "cos_negative_std": float(cos_negative_all.std()),
        "gap": float(gap.mean()),
        "gap_std": float(gap.std()),
        "cos_random_single": float(cos_random_single.mean()),
        "gap_random_single": float(gap_random_single.mean()),
        "retrieval_top1_pool": retrieval_top1,
        "retrieval_chance_level": 1.0 / scores.shape[0],
        "num_pairs": int(scores.shape[0]),
        "negatives_per_pair": int(negative_rows.shape[1]),
    }
    per_pair = {
        "cos_positive": cos_positive,
        "cos_negative": cos_negative,
        "cos_random_single": cos_random_single,
        "gap": gap,
        "gap_random_single": gap_random_single,
    }
    return metrics, per_pair



def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    set_seed(args.seed)

    print("=" * 84)
    print("STEP 0 VERIFICATION - geometry of the continuation head vs the original")
    print("=" * 84)

    specs = parse_checkpoints(args.checkpoints)
    for label, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path} does not exist")
    print("models:")
    for label, path in specs:
        print(f"  {label:<38} {path}")

    pairs = load_pairs(args.pairs)
    sentences: List[str] = []
    sentence_index: Dict[str, int] = {}
    for first, second in pairs:
        for sentence in (first, second):
            if sentence not in sentence_index:
                sentence_index[sentence] = len(sentences)
                sentences.append(sentence)
    anchor_rows_all = np.fromiter(
        (sentence_index[first] for first, _ in pairs), dtype=np.int64, count=len(pairs)
    )
    positive_rows_all = np.fromiter(
        (sentence_index[second] for _, second in pairs), dtype=np.int64, count=len(pairs)
    )

    # the frozen base encoder is identical in every checkpoint here, so one model
    # is enough to (re)build the shared base-embedding cache
    reference_vae = load_vae(specs[0][1], device)
    base_embeddings = load_or_build_base_embeddings(
        reference_vae, sentences, args.base_cache, force_reencode=args.force_reencode
    )

    rng = np.random.default_rng(args.seed)
    candidates = np.arange(int(args.heldout_from), len(pairs))
    if candidates.size == 0:
        raise ValueError(f"--heldout-from {args.heldout_from} leaves no pairs")
    sample_size = min(int(args.sample_size), candidates.size)
    sample = np.sort(rng.choice(candidates, size=sample_size, replace=False))
    anchor_rows = anchor_rows_all[sample]
    positive_rows = positive_rows_all[sample]

    # random negatives: one draw per pair, shared by every model (paired design)
    pool = np.arange(len(sentences))
    negative_rows = np.zeros((sample_size, int(args.negatives)), dtype=np.int64)
    for position in range(sample_size):
        excluded = {int(anchor_rows[position]), int(positive_rows[position])}
        allowed = np.array([row for row in pool.tolist() if row not in excluded], dtype=np.int64)
        negative_rows[position] = rng.choice(allowed, size=int(args.negatives), replace=False)

    heldout = int(args.heldout_from) > 0
    print(
        f"\nsampled {sample_size} pairs (seed {args.seed})"
        + (
            f" with index >= {args.heldout_from} (HELD OUT for a head trained on the "
            f"first {args.heldout_from} pairs)"
            if heldout
            else " from the whole file"
        )
        + f" | {args.negatives} random negatives per pair, shared across models"
    )

    results: Dict[str, Any] = {}
    per_pair_store: Dict[str, Dict[str, np.ndarray]] = {}
    for label, path in specs:
        vae = reference_vae if path == specs[0][1] else load_vae(path, device)
        metrics, per_pair = model_geometry(
            vae, base_embeddings, anchor_rows, positive_rows, negative_rows
        )
        results[label] = metrics
        per_pair_store[label] = per_pair

    reference_label = specs[0][0]
    comparison: Dict[str, Any] = {}
    for label, _ in specs[1:]:
        comparison[label] = {
            "vs": reference_label,
            "cos_positive": paired_stats(
                per_pair_store[label]["cos_positive"],
                per_pair_store[reference_label]["cos_positive"],
            ),
            "cos_negative": paired_stats(
                per_pair_store[label]["cos_negative"],
                per_pair_store[reference_label]["cos_negative"],
            ),
            "cos_random_single": paired_stats(
                per_pair_store[label]["cos_random_single"],
                per_pair_store[reference_label]["cos_random_single"],
            ),
            "gap": paired_stats(
                per_pair_store[label]["gap"], per_pair_store[reference_label]["gap"]
            ),
            "gap_random_single": paired_stats(
                per_pair_store[label]["gap_random_single"],
                per_pair_store[reference_label]["gap_random_single"],
            ),
        }
    return _report(args, results, comparison, sample, heldout, sample_size)



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(
    args: argparse.Namespace,
    results: Dict[str, Any],
    comparison: Dict[str, Any],
    sample: np.ndarray,
    heldout: bool,
    sample_size: int,
) -> Dict[str, Any]:
    """Print the geometry table + verdicts and write the JSON artifact."""
    print("\n" + "=" * 84)
    print("GEOMETRY (same pairs, same random negatives)")
    print("=" * 84)
    print(
        "  model                          | cos(S_n,S_n+1) | cos(S_n,rand) | "
        "   gap | gap(1 draw) | top1(pool)"
    )
    print("  " + "-" * 96)
    for label, metrics in results.items():
        print(
            f"  {label:<30} | {metrics['cos_positive']:>14.4f} | "
            f"{metrics['cos_negative']:>13.4f} | {metrics['gap']:>6.4f} | "
            f"{metrics['gap_random_single']:>11.4f} | "
            f"{100 * metrics['retrieval_top1_pool']:>9.1f}%"
        )
    print("  " + "-" * 96)
    chance = next(iter(results.values()))["retrieval_chance_level"]
    print(
        f"  retrieval chance level = {100 * chance:.1f}% ({sample_size}-way) | "
        f"negatives per pair = {args.negatives}"
    )

    print("\n  paired deltas vs the first model (positive = better geometry)")
    for label, stats in comparison.items():
        print(f"    {label}")
        print(
            f"      cos(S_n,S_n+1) : {stats['cos_positive']['mean_delta']:+.4f} "
            f"({100 * stats['cos_positive']['share_wins']:.1f}% of pairs up, "
            f"t {stats['cos_positive']['paired_t_statistic']:+.2f})"
        )
        print(
            f"      cos(S_n,rand)  : {stats['cos_negative']['mean_delta']:+.4f} "
            f"(lower is better; {100 * stats['cos_negative']['share_wins']:.1f}% of "
            f"pairs up, t {stats['cos_negative']['paired_t_statistic']:+.2f})"
        )
        print(
            f"      gap (K draws)  : {stats['gap']['mean_delta']:+.4f} "
            f"({100 * stats['gap']['share_wins']:.1f}% of pairs up, "
            f"t {stats['gap']['paired_t_statistic']:+.2f})"
        )
        print(
            f"      gap (1 draw)   : {stats['gap_random_single']['mean_delta']:+.4f} "
            f"(t {stats['gap_random_single']['paired_t_statistic']:+.2f})"
        )

    print("\n  VERDICT")
    for label, stats in comparison.items():
        wider = stats["gap"]["mean_delta"] > 0
        higher = stats["cos_positive"]["mean_delta"] > 0
        lower_neg = stats["cos_negative"]["mean_delta"] < 0
        print(
            f"    {label}: cos(S_n,S_n+1) higher = {higher} | "
            f"cos(S_n,random) lower = {lower_neg} | gap wider = {wider}"
        )

    payload = {
        "task": "STEP 0 verification - continuation encoder geometry",
        "config": {
            "pairs": str(args.pairs),
            "base_cache": str(args.base_cache),
            "sample_size": sample_size,
            "negatives_per_pair": args.negatives,
            "seed": args.seed,
            "heldout_from": args.heldout_from,
            "device": args.device,
        },
        "sample": {
            "num_pairs": sample_size,
            "pair_indices": [int(value) for value in sample],
            "held_out": heldout,
        },
        "reference_original_gap_quoted": REFERENCE_ORIGINAL_GAP,
        "models": results,
        "paired_vs_first_model": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved {args.output}")
    print("=" * 84)
    return payload


if __name__ == "__main__":
    main()

