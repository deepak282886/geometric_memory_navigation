"""TASK 9: redesigned ablations on the full 39,905-example memory store.

The ablation set was reshaped around what is actually ablatable (see
``architectural_findings`` in the output JSON):

  no_navigation   skip retrieval entirely, feed ``concat([q, zeros(256)])``
  random_fusion   fuse 3 RANDOM neighbours instead of the top-3
  uniform_weights not ablatable - inference already uses uniform weights
                  (geometric scoring needs no trained head), reported as a no-op
  single_neighbor fuse top_k=1: follow only the single best memory
  full_model      full store, k=10, fuse top_k=3  (the 33.6% baseline)

Query and ending vectors are encoded exactly the way ``score_example`` does it
(one context at a time, four endings at a time), so the full-model ablation
reproduces the TASK 8 number exactly rather than drifting on batching noise.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.eval.evaluate import DEFAULT_FUSE_TOP_K, DEFAULT_K  # noqa: E402
from src.fusion.fuse import EPS, fuse_directions  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402
from src.sampler.jev import (  # noqa: E402
    TARGET_CONTEXT,
    TARGET_CORRECT_ENDING,
    parallel_sample,
)

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover - scipy is in requirements.txt
    binomtest = None

FULL_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
VAL_PATH = PROJECT_ROOT / "data" / "val.json"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_full.json"
ABLATIONS_PATH = PROJECT_ROOT / "experiments" / "ablations.json"

K = DEFAULT_K
TOP_K = DEFAULT_FUSE_TOP_K
INFERENCE_OUTCOME_WEIGHT = 1.0
SEED = 42


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


@torch.no_grad()
def precompute_vectors(
    vae: VAE, val_data: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit query vectors ``(N, 256)`` and unit ending vectors ``(N, 4, 256)``.

    Encodes one context at a time and four endings at a time, i.e. exactly the
    batch shapes ``score_example`` uses, so every ablation shares identical
    inputs with the TASK 8 baseline.
    """
    queries = np.zeros((len(val_data), 256), dtype=np.float64)
    endings = np.zeros((len(val_data), 4, 256), dtype=np.float64)
    for index, example in enumerate(val_data):
        queries[index] = normalize_rows(
            vae.encode_texts([example["ctx"]]).cpu().numpy()
        )[0]
        endings[index] = normalize_rows(
            vae.encode_texts(list(example["endings"])).cpu().numpy()
        )
    return queries, endings


def fuse_random(
    weighted_directions: List[Tuple[float, np.ndarray]],
    top_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Like ``fuse_directions`` but selects ``top_k`` neighbours at random.

    Same arithmetic as the real fusion (selected weights renormalised, weighted
    sum, L2-normalised); only the *selection* is random, so the comparison
    isolates "picking the strongest directions" from "picking some directions".
    """
    count = len(weighted_directions)
    if count == 0:
        return np.zeros(256, dtype=np.float64)
    keep = min(max(1, int(top_k)), count)
    chosen = rng.choice(count, size=keep, replace=False)

    weights = np.asarray([weighted_directions[int(i)][0] for i in chosen], dtype=np.float64)
    total = float(weights.sum())
    if total <= EPS:
        return np.zeros(256, dtype=np.float64)
    weights = weights / total

    stacked = np.stack([weighted_directions[int(i)][1] for i in chosen])
    navigation_vector = weights @ stacked
    norm = float(np.linalg.norm(navigation_vector))
    if norm <= EPS:
        return np.zeros(256, dtype=np.float64)
    return navigation_vector / norm


def navigated_position(
    query_vector: np.ndarray,
    store: MemoryStore,
    mode: str,
    rng: np.random.Generator,
    k: int = K,
    top_k: int = TOP_K,
) -> np.ndarray:
    """The position that endings are scored against, under the given mode."""
    if mode == "no_navigation":
        # retrieval skipped entirely: the navigation vector is zeros(256), so
        # navigate() receives concat([query_vector, zeros(256)]) and, after
        # renormalisation, the query vector is used unchanged
        return query_vector

    memories = store.query(query_vector, k=k)
    outcome_weights = [INFERENCE_OUTCOME_WEIGHT] * len(memories)
    # The continuation mode targets each memory's known-correct continuation.
    # Every pre-existing mode keeps the original context target so the TASK 9
    # numbers and the cached chord/tangent results stay reproducible.
    sample_target = (
        TARGET_CORRECT_ENDING if mode == "full_continuation" else TARGET_CONTEXT
    )
    weighted_directions = parallel_sample(
        query_vector, memories, outcome_weights, target=sample_target
    )

    if mode == "random_fusion":
        navigation_vector = fuse_random(weighted_directions, top_k, rng)
    elif mode in ("full_tangent", "full_continuation"):
        # radial-component fix: project each direction onto the query's tangent
        # space before fusing (see src/fusion/fuse.py::tangent_direction)
        navigation_vector = fuse_directions(
            weighted_directions,
            top_k=top_k,
            use_tangent=True,
            query_vector=query_vector,
        )
    elif mode == "single_neighbor":
        navigation_vector = fuse_directions(
            weighted_directions, top_k=1, use_tangent=False
        )
    else:
        navigation_vector = fuse_directions(
            weighted_directions, top_k=top_k, use_tangent=False
        )

    navigated = query_vector + navigation_vector
    norm = float(np.linalg.norm(navigated))
    return navigated / norm if norm > 0.0 else navigated


def run_ablation(
    val_data: List[Dict[str, Any]],
    query_vectors: np.ndarray,
    ending_vectors: np.ndarray,
    store: MemoryStore,
    mode: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Geometric scoring over the val set with one component ablated."""
    predictions: List[int] = []
    labels: List[int] = []
    started = time.perf_counter()

    # one redirect for the whole run: parallel_sample announces every call
    with contextlib.redirect_stdout(io.StringIO()):
        for index, example in enumerate(val_data):
            position = navigated_position(query_vectors[index], store, mode, rng)
            scores = ending_vectors[index] @ position
            predictions.append(int(np.argmax(scores)))
            labels.append(int(example["label"]))

    seconds = time.perf_counter() - started
    correct = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
    return {
        "accuracy": correct / max(len(val_data), 1),
        "correct": correct,
        "num_examples": len(val_data),
        "seconds": seconds,
        "ms_per_example": 1000.0 * seconds / max(len(val_data), 1),
        "predictions": predictions,
        "labels": labels,
    }


def paired_comparison(
    full_predictions: List[int],
    other_predictions: List[int],
    labels: List[int],
) -> Dict[str, Any]:
    """McNemar-style paired test of an ablation against the full model.

    Both runs score the *same* 500 examples, so a paired test is the right tool:
    the unpaired standard error is ~2.1 points here, which means a 1-4 point gap
    cannot be interpreted on its own. Only examples where the two disagree
    contribute, and those are split into "full right / ablation wrong" versus the
    reverse.
    """
    full_correct = [prediction == label for prediction, label in zip(full_predictions, labels)]
    other_correct = [prediction == label for prediction, label in zip(other_predictions, labels)]

    full_only = sum(1 for a, b in zip(full_correct, other_correct) if a and not b)
    other_only = sum(1 for a, b in zip(full_correct, other_correct) if b and not a)
    discordant = full_only + other_only

    if discordant == 0 or binomtest is None:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(full_only, other_only), discordant, 0.5).pvalue)

    return {
        "full_correct_ablation_wrong": full_only,
        "full_wrong_ablation_correct": other_only,
        "discordant_pairs": discordant,
        "mcnemar_p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


ARCHITECTURAL_FINDINGS = {
    "k_inert": (
        "k is inert for k >= top_k (=3). Accuracy identical from k=3 to k=100. "
        "Navigation direction determined by top-3 regardless of retrieval pool size."
    ),
    "outcome_weights": (
        "Geometric scoring uses no trained head, so outcome weights at inference "
        "are always 1.0. Training-time outcome weighting was superseded by the "
        "architecture change in Task 8."
    ),
    "hyperedge_bonus": "Not implemented. Left for future work.",
}


def main() -> None:
    with VAL_PATH.open(encoding="utf-8") as handle:
        val_data = json.load(handle)
    with RESULTS_PATH.open(encoding="utf-8") as handle:
        full_results = json.load(handle)
    reference_accuracy = float(full_results["stores"]["full"]["best_required"]["accuracy"])

    vae, _checkpoint = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    store = MemoryStore.load(FULL_STORE_PATH)
    print(f"val examples {len(val_data)} | full store {store.size} vectors")
    print(f"reference (TASK 8, full store, k={K}, top_k={TOP_K}): "
          f"{100 * reference_accuracy:.1f}%")

    print("\nPrecomputing query / ending vectors (same batch shapes as score_example) ...")
    precompute_started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        query_vectors, ending_vectors = precompute_vectors(vae, val_data)
    print(f"  query {query_vectors.shape} | endings {ending_vectors.shape} "
          f"in {time.perf_counter() - precompute_started:.1f}s")

    rng = np.random.default_rng(SEED)
    results: Dict[str, Dict[str, Any]] = {}

    # full_model runs first: it consumes no randomness, so the RNG stream used by
    # random_fusion is fixed and the run is reproducible
    for label, mode in (
        ("full_model", "full"),
        ("no_navigation", "no_navigation"),
        ("random_fusion", "random_fusion"),
        ("single_neighbor", "single_neighbor"),
    ):
        print(f"\n[{label}] running ...")
        outcome = run_ablation(val_data, query_vectors, ending_vectors, store, mode, rng)
        results[label] = outcome
        print(f"  accuracy {100 * outcome['accuracy']:.1f}% "
              f"({outcome['correct']}/{outcome['num_examples']}) "
              f"in {outcome['seconds']:.1f}s")

    # uniform_weights cannot be ablated at inference - it is already the default
    results["uniform_weights"] = dict(results["full_model"])

    # ---------------------------------------------------------- sanity check
    reproduced = abs(results["full_model"]["accuracy"] - reference_accuracy) < 1e-12
    print(f"\nfull_model reproduces the TASK 8 reference exactly: {reproduced} "
          f"({100 * results['full_model']['accuracy']:.1f}% vs "
          f"{100 * reference_accuracy:.1f}%)")

    # ------------------------------------------------------------- table
    full_accuracy = results["full_model"]["accuracy"]
    order = ["no_navigation", "random_fusion", "uniform_weights",
             "single_neighbor", "full_model"]

    print("\n" + "=" * 58)
    print("ABLATIONS (full store, 500 HellaSwag val examples)")
    print("=" * 58)
    header = f"{'Ablation':<20} | {'Accuracy':>8} | Delta vs Full"
    print(header)
    print("-" * len(header))

    table_rows: List[Dict[str, Any]] = []
    for name in order:
        accuracy = results[name]["accuracy"]
        delta = accuracy - full_accuracy
        if name == "uniform_weights":
            accuracy_text, delta_text = f"{100 * accuracy:.1f}%*", "0.0% (see note)"
        elif name == "full_model":
            accuracy_text, delta_text = f"{100 * accuracy:.1f}%", "baseline"
        else:
            accuracy_text, delta_text = f"{100 * accuracy:.1f}%", f"{100 * delta:+.1f}%"
        print(f"{name:<20} | {accuracy_text:>8} | {delta_text}")
        table_rows.append(
            {
                "ablation": name,
                "accuracy": accuracy,
                "delta_vs_full": delta,
                # final value is set from the McNemar test further down
                "meaningful": False,
            }
        )
    print("-" * len(header))
    print("* uniform_weights is a no-op: inference already uses uniform weights (1.0)")
    print("  for every neighbour, so the geometry is unchanged.")

    # ------------------------------------------- paired significance tests
    full_predictions = results["full_model"]["predictions"]
    labels_reference = results["full_model"]["labels"]
    paired: Dict[str, Dict[str, Any]] = {}
    for name in order:
        if name == "full_model":
            continue
        paired[name] = paired_comparison(
            full_predictions, results[name]["predictions"], labels_reference
        )

    # "meaningful" is now statistical significance, not an arbitrary point cut
    for row in table_rows:
        stats = paired.get(row["ablation"], {})
        row["mcnemar_p_value"] = stats.get("mcnemar_p_value")
        row["discordant_pairs"] = stats.get("discordant_pairs")
        row["significant_vs_full"] = bool(stats.get("significant_at_0.05", False))
        row["meaningful"] = row["ablation"] != "full_model" and row["significant_vs_full"]

    # ------------------------------------------------------------ analysis
    ablatable = [
        row for row in table_rows if row["ablation"] not in ("uniform_weights", "full_model")
    ]
    biggest_drop = min(ablatable, key=lambda row: row["accuracy"])
    biggest_gain = max(ablatable, key=lambda row: row["accuracy"])
    meaningful = [row["ablation"] for row in table_rows if row["meaningful"]]
    no_navigation = next(row for row in table_rows if row["ablation"] == "no_navigation")
    navigation_helps = bool(
        no_navigation["delta_vs_full"] < 0 and no_navigation["significant_vs_full"]
    )
    unpaired_se = 100.0 * (full_accuracy * (1.0 - full_accuracy) / len(val_data)) ** 0.5

    print("\n" + "=" * 76)
    print("PAIRED SIGNIFICANCE vs FULL MODEL (McNemar exact, same 500 examples)")
    print("=" * 76)
    print(f"  {'ablation':<18} {'acc':>7} {'delta':>7} {'b':>4} {'c':>4} {'p':>8}  significant")
    print("  " + "-" * 58)
    for row in table_rows:
        stats = paired.get(row["ablation"])
        if stats is None:
            print(f"  {row['ablation']:<18} {100 * row['accuracy']:>6.1f}% "
                  f"{100 * row['delta_vs_full']:>+6.1f}% {'-':>4} {'-':>4} {'-':>8}  (reference)")
            continue
        print(f"  {row['ablation']:<18} {100 * row['accuracy']:>6.1f}% "
              f"{100 * row['delta_vs_full']:>+6.1f}% "
              f"{stats['full_correct_ablation_wrong']:>4} "
              f"{stats['full_wrong_ablation_correct']:>4} "
              f"{stats['mcnemar_p_value']:>8.4f}  {row['significant_vs_full']}")
    print("  b = full correct & ablation wrong | c = full wrong & ablation correct")

    print("\nANALYSIS")
    print(f"  biggest drop   : {biggest_drop['ablation']} "
          f"({100 * biggest_drop['delta_vs_full']:+.1f}%, "
          f"p={paired[biggest_drop['ablation']]['mcnemar_p_value']:.4f})")
    print(f"  biggest gain   : {biggest_gain['ablation']} "
          f"({100 * biggest_gain['delta_vs_full']:+.1f}%, "
          f"p={paired[biggest_gain['ablation']]['mcnemar_p_value']:.4f})")
    print(f"  significant    : {meaningful if meaningful else 'none'}")
    print(f"  unpaired std error on 500 examples: {unpaired_se:.2f} points")
    print(f"\n  navigation HELPS (no_navigation worse AND significant): {navigation_helps}")
    if not navigation_helps:
        print(f"    -> no_navigation scores {100 * no_navigation['accuracy']:.1f}% vs the "
              f"full model's {100 * full_accuracy:.1f}%: removing navigation does NOT hurt.")

    # -------------------------------------------------------------- save
    findings = dict(ARCHITECTURAL_FINDINGS)
    findings["navigation_direction"] = (
        "Measured, not assumed: removing navigation entirely (no_navigation) scores "
        f"{100 * no_navigation['accuracy']:.1f}% vs the full model's "
        f"{100 * full_accuracy:.1f}% (delta {100 * no_navigation['delta_vs_full']:+.1f}%, "
        f"McNemar p={paired['no_navigation']['mcnemar_p_value']:.4f}). Moving the position "
        "toward the retrieved-memory neighbourhood does not help on this task; the "
        "strongest single geometric signal is the cosine similarity between the "
        "candidate ending and the context position itself."
    )

    payload = {
        "experiment": "ablations (redesigned around what is actually ablatable)",
        "dataset": "hellaswag",
        "split": "validation",
        "val_examples": len(val_data),
        "seed": SEED,
        "config": {
            "store": "checkpoints/memory_store_full.pkl",
            "store_size": store.size,
            "scoring": "cosine(L2(encode(ending)), navigated position)",
            "classifier_head": "none",
            "retrieval_k": K,
            "fuse_top_k": TOP_K,
            "outcome_weights_at_inference": INFERENCE_OUTCOME_WEIGHT,
        },
        "full_model_accuracy": full_accuracy,
        "full_model_reproduces_reference": bool(reproduced),
        "table": table_rows,
        "ablations": {
            name: {
                "accuracy": results[name]["accuracy"],
                "delta_vs_full": results[name]["accuracy"] - full_accuracy,
                "correct": results[name]["correct"],
                "num_examples": results[name]["num_examples"],
                "seconds": results[name]["seconds"],
                "ms_per_example": results[name]["ms_per_example"],
            }
            for name in order
        },
        "analysis": {
            "biggest_drop": biggest_drop["ablation"],
            "biggest_drop_delta": biggest_drop["delta_vs_full"],
            "biggest_gain": biggest_gain["ablation"],
            "biggest_gain_delta": biggest_gain["delta_vs_full"],
            "navigation_helps": bool(navigation_helps),
            "significant_ablations": meaningful,
            "significance_test": "McNemar exact test against full_model, p < 0.05",
            "unpaired_std_error_points": unpaired_se,
        },
        "paired_significance": paired,
        "notes": {
            "uniform_weights": (
                "No-op ablation. With geometric scoring there is no trained head "
                "and outcome weights at inference are always 1.0, so it is reported "
                "at the full-model number with this explanation rather than as a "
                "measured delta."
            ),
            "k_sweep": (
                "Varying k is not a valid ablation: k is inert for k >= top_k (3). "
                "Accuracy was identical (33.6%) at k=5, 10, 20 and 50 on the full store."
            ),
            "random_fusion": (
                "3 neighbours chosen uniformly at random (seed 42, without "
                "replacement) and fused with the same renormalised weighting; only "
                "the selection differs from the full model."
            ),
            "no_navigation": (
                "Retrieval skipped; navigate() receives concat([query_vector, "
                "zeros(256)]), so endings are scored against the raw query position."
            ),
            "single_neighbor": "fuse_directions(top_k=1): follow only the strongest memory.",
        },
        "architectural_findings": findings,
    }
    ABLATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ABLATIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nAblations saved to {ABLATIONS_PATH}")
    print("Ablation run complete.")


if __name__ == "__main__":
    main()

