"""STEP 2: why does navigation hurt? Per-example rank bookkeeping.

For 200 randomly chosen validation examples this compares two positions:

    ctx_vector   L2-normalised context embedding        ("no navigation")
    navigated    L2(ctx_vector + fused navigation)      ("with navigation")

and asks, for the CORRECT ending only, the score and rank under each position,
and whether navigation improved, hurt or left the rank unchanged.

It also records the activity labels of the top-3 neighbours fusion actually
selects and splits the rank changes by whether those neighbours share the query's
activity label - that is the mechanism check.

Writes ``experiments/navigation_diagnosis.json``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.fusion.fuse import fuse_directions  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402
from src.sampler.jev import parallel_sample  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
FULL_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full.pkl"
#: store that also carries each memory's known-correct continuation
STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full_v2.pkl"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "navigation_diagnosis.json"
TANGENT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "navigation_diagnosis_tangent.json"
CONTINUATION_OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / "navigation_diagnosis_continuation.json"
)

NUM_EXAMPLES = 200
SEED = 42
K = 10
TOP_K = 3
INFERENCE_OUTCOME_WEIGHT = 1.0


def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
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


def rank_of(scores: np.ndarray, label: int) -> int:
    """Rank of ``label`` among the scores, 1 = highest."""
    order = np.argsort(-np.asarray(scores))
    return int(np.where(order == label)[0][0]) + 1


def run_diagnosis(use_tangent: bool = False, target: str = "context") -> Dict[str, Any]:
    """Run the per-example rank bookkeeping; returns the payload (no file I/O).

    ``use_tangent=True`` applies the radial-component fix in fusion, so
    ``cos(ctx, navigation_vector)`` should drop from ~-0.45 to ~0.0.
    ``target="correct_ending"`` points navigation at each memory's known-correct
    continuation instead of the neighbouring context.
    """
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)

    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(len(val_data), size=NUM_EXAMPLES, replace=False))
    print(f"val split {len(val_data)} | sampling {NUM_EXAMPLES} examples (seed {SEED})")

    vae = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    store = MemoryStore.load(STORE_PATH)
    print(f"store {store.size} vectors | k={K} | fuse top_k={TOP_K} | "
          f"use_tangent={use_tangent} | target={target}")

    records: List[Dict[str, Any]] = []
    started = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        for position, raw_index in enumerate(indices):
            example = val_data[int(raw_index)]
            label = int(example["label"])

            ctx_vector = normalize_rows(
                vae.encode_texts([example["ctx"]]).cpu().numpy()
            )[0].astype(np.float64)
            ending_vectors = normalize_rows(
                vae.encode_texts(list(example["endings"])).cpu().numpy()
            ).astype(np.float64)

            memories = store.query(ctx_vector, k=K)
            weights = [INFERENCE_OUTCOME_WEIGHT] * len(memories)
            weighted_directions = parallel_sample(
                ctx_vector, memories, weights, target=target
            )
            navigation_vector = fuse_directions(
                weighted_directions,
                top_k=TOP_K,
                use_tangent=use_tangent,
                query_vector=ctx_vector,
            )

            navigated = ctx_vector + navigation_vector
            norm = float(np.linalg.norm(navigated))
            navigated = navigated / norm if norm > 0.0 else navigated

            scores_ctx = ending_vectors @ ctx_vector
            scores_nav = ending_vectors @ navigated

            # the neighbours fusion actually used: top-TOP_K by weight
            chosen = np.argsort([-w for w, _ in weighted_directions])[:TOP_K]
            neighbour_activities = [memories[int(i)]["activity_label"] for i in chosen]
            same_activity = [a == example["activity_label"] for a in neighbour_activities]
            neighbour_similarities = [float(m["distance"]) for m in memories]

            rank_ctx = rank_of(scores_ctx, label)
            rank_nav = rank_of(scores_nav, label)
            change = rank_ctx - rank_nav  # >0 improved, <0 hurt, 0 unchanged

            incorrect_ctx = [float(s) for k, s in enumerate(scores_ctx) if k != label]
            incorrect_nav = [float(s) for k, s in enumerate(scores_nav) if k != label]

            records.append(
                {
                    "val_index": int(raw_index),
                    "activity_label": example["activity_label"],
                    "label": label,
                    "score_ctx_correct": float(scores_ctx[label]),
                    "score_nav_correct": float(scores_nav[label]),
                    "rank_ctx": rank_ctx,
                    "rank_nav": rank_nav,
                    "rank_change": int(change),
                    "improved": bool(change > 0),
                    "hurt": bool(change < 0),
                    "unchanged": bool(change == 0),
                    "pred_ctx": int(np.argmax(scores_ctx)),
                    "pred_nav": int(np.argmax(scores_nav)),
                    "neighbour_activities": neighbour_activities,
                    "neighbour_same_activity": same_activity,
                    "any_neighbour_same_activity": bool(any(same_activity)),
                    "all_neighbour_same_activity": bool(all(same_activity)),
                    "mean_neighbour_similarity": float(np.mean(neighbour_similarities)),
                    "mean_incorrect_ctx": float(np.mean(incorrect_ctx)),
                    "mean_incorrect_nav": float(np.mean(incorrect_nav)),
                    "nav_norm": float(np.linalg.norm(navigation_vector)),
                    "cos_ctx_nav": float(ctx_vector @ navigation_vector),
                }
            )
            if (position + 1) % 50 == 0:
                print(f"  {position + 1}/{NUM_EXAMPLES}", flush=True)

    elapsed = time.perf_counter() - started
    n = len(records)

    improved = [r for r in records if r["improved"]]
    hurt = [r for r in records if r["hurt"]]
    unchanged = [r for r in records if r["unchanged"]]
    mean_score_ctx = float(np.mean([r["score_ctx_correct"] for r in records]))
    mean_score_nav = float(np.mean([r["score_nav_correct"] for r in records]))
    accuracy_ctx = float(np.mean([r["rank_ctx"] == 1 for r in records]))
    accuracy_nav = float(np.mean([r["rank_nav"] == 1 for r in records]))

    slots = [flag for r in records for flag in r["neighbour_same_activity"]]
    same_slots = int(sum(slots))
    same_pct = 100.0 * same_slots / max(len(slots), 1)
    any_match = sum(1 for r in records if r["any_neighbour_same_activity"])

    # ------------------------------------------------------------- report
    print("\n" + "=" * 66)
    print(f"NAVIGATION DIAGNOSIS ({n} random val examples, full store, "
          f"{'TANGENT' if use_tangent else 'CHORD'})")
    print("=" * 66)
    print(f"  Mean score of correct ending from ctx       : {mean_score_ctx:+.4f}")
    print(f"  Mean score of correct ending from navigated : {mean_score_nav:+.4f}")
    print(f"  change                                      : "
          f"{mean_score_nav - mean_score_ctx:+.4f}")
    print(f"\n  Cases where navigation IMPROVED rank : {len(improved):>3d}/{n}")
    print(f"  Cases where navigation HURT rank     : {len(hurt):>3d}/{n}")
    print(f"  Cases where rank unchanged           : {len(unchanged):>3d}/{n}")
    print(f"  net change in correct predictions    : {len(improved) - len(hurt):+d}")
    print(f"  accuracy without navigation (rank 1) : {100 * accuracy_ctx:.1f}%")
    print(f"  accuracy with navigation    (rank 1) : {100 * accuracy_nav:.1f}%")

    print(f"\n  Top-{TOP_K} neighbour activity labels vs query activity label:")
    print(f"    same activity     : {same_pct:.1f}% of {len(slots)} neighbour slots")
    print(f"    different activity: {100 - same_pct:.1f}%")
    print(f"    examples with >=1 matching top-{TOP_K} neighbour: {any_match}/{n} "
          f"({100 * any_match / n:.1f}%)")

    mean_incorrect_ctx = float(np.mean([r["mean_incorrect_ctx"] for r in records]))
    mean_incorrect_nav = float(np.mean([r["mean_incorrect_nav"] for r in records]))
    mean_neighbour_similarity = float(np.mean([r["mean_neighbour_similarity"] for r in records]))
    mean_cos_ctx_nav = float(np.mean([r["cos_ctx_nav"] for r in records]))
    mean_nav_norm = float(np.mean([r["nav_norm"] for r in records]))
    # for unit vectors q, m with cos(q,m) = mu, the chord direction m - q has
    # cos(q, m - q) = (mu - 1) / sqrt(2 - 2 mu); strictly negative for mu < 1
    chord_prediction = (
        (mean_neighbour_similarity - 1.0) / np.sqrt(2.0 - 2.0 * mean_neighbour_similarity)
        if mean_neighbour_similarity < 1.0
        else 0.0
    )

    print("\n  WHY: the correct ending is MORE similar to the context than the others")
    print(f"    mean cos(correct ending, ctx)      : {mean_score_ctx:+.4f}")
    print(f"    mean cos(incorrect endings, ctx)   : {mean_incorrect_ctx:+.4f}")
    print(f"    gap                                : "
          f"{mean_score_ctx - mean_incorrect_ctx:+.4f}")
    print(f"    mean cos(correct ending, navigated): {mean_score_nav:+.4f}")
    print(f"    mean cos(incorrect endings, navigated): {mean_incorrect_nav:+.4f}")
    print(f"    gap                                : "
          f"{mean_score_nav - mean_incorrect_nav:+.4f}")
    print("\n  WHY: the navigation direction points AWAY from the context")
    print(f"    mean cos(ctx, neighbour memory)    : {mean_neighbour_similarity:+.4f}  (mu)")
    print(f"    mean ||navigation_vector||         : {mean_nav_norm:.4f}")
    print(f"    mean cos(ctx, nav) measured        : {mean_cos_ctx_nav:+.4f}")
    if mean_nav_norm > 0:
        print(f"    chord formula (mu-1)/sqrt(2-2mu)   : {chord_prediction:+.4f}  "
              f"<- predicts the measured angle")
    print("    => because the memory vectors are UNIT vectors, the step from ctx")
    print("       toward any neighbour is nearly perpendicular to ctx (and its cosine")
    print("       with ctx is negative for mu < 1), so 'navigating' rotates the")
    print("       position away from the very signal the scoring relies on.")

    print("\n  how far navigation moves the position:")
    print(f"    mean ||navigation_vector|| : "
          f"{np.mean([r['nav_norm'] for r in records]):.4f}")
    print(f"    mean cos(ctx, nav)         : "
          f"{np.mean([r['cos_ctx_nav'] for r in records]):+.4f}")

    # ------------------------------------------------- mechanism split
    def summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "improved": sum(1 for r in group if r["improved"]),
            "hurt": sum(1 for r in group if r["hurt"]),
            "unchanged": sum(1 for r in group if r["unchanged"]),
            "accuracy_without_navigation": float(np.mean([r["rank_ctx"] == 1 for r in group])),
            "accuracy_with_navigation": float(np.mean([r["rank_nav"] == 1 for r in group])),
            "mean_rank_change": float(np.mean([r["rank_change"] for r in group])),
        }

    by_activity_match = {
        "at_least_one_top_k_neighbour_same_activity": summarize(
            [r for r in records if r["any_neighbour_same_activity"]]
        ),
        "no_top_k_neighbour_same_activity": summarize(
            [r for r in records if not r["any_neighbour_same_activity"]]
        ),
    }

    print("\n  MECHANISM: rank change split by neighbour/query activity agreement")
    for name, stats in by_activity_match.items():
        if stats["n"] == 0:
            continue
        print(f"    {name}: n={stats['n']}")
        print(f"      improved {stats['improved']} | hurt {stats['hurt']} | "
              f"unchanged {stats['unchanged']}")
        print(f"      accuracy: without nav "
              f"{100 * stats['accuracy_without_navigation']:.1f}% -> with nav "
              f"{100 * stats['accuracy_with_navigation']:.1f}%")

    # -------------------------------------------------------------- save
    payload = {
        "experiment": "navigation diagnosis",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "num_examples": n,
        "seed": SEED,
        "sampled_val_indices": [int(i) for i in indices],
        "config": {
            "store": "checkpoints/memory_store_full.pkl",
            "store_size": store.size,
            "retrieval_k": K,
            "fuse_top_k": TOP_K,
            "use_tangent": use_tangent,
            "target": target,
            "outcome_weights_at_inference": INFERENCE_OUTCOME_WEIGHT,
        },
        "seconds": elapsed,
        "summary": {
            "mean_score_correct_ctx": mean_score_ctx,
            "mean_score_correct_navigated": mean_score_nav,
            "mean_score_change": mean_score_nav - mean_score_ctx,
            "improved": len(improved),
            "hurt": len(hurt),
            "unchanged": len(unchanged),
            "net_correct_change": len(improved) - len(hurt),
            "accuracy_without_navigation": accuracy_ctx,
            "accuracy_with_navigation": accuracy_nav,
            "neighbour_slots": len(slots),
            "neighbour_same_activity": same_slots,
            "neighbour_same_activity_pct": same_pct,
            "examples_with_any_matching_neighbour": any_match,
            "mean_nav_norm": mean_nav_norm,
            "mean_cos_ctx_nav": mean_cos_ctx_nav,
            "mean_neighbour_similarity_mu": mean_neighbour_similarity,
            "chord_prediction_cos_ctx_nav": float(chord_prediction),
            "mean_incorrect_score_ctx": mean_incorrect_ctx,
            "mean_incorrect_score_navigated": mean_incorrect_nav,
            "score_gap_ctx": mean_score_ctx - mean_incorrect_ctx,
            "score_gap_navigated": mean_score_nav - mean_incorrect_nav,
        },
        "by_neighbour_activity_match": by_activity_match,
        "records": records,
    }
    return payload


def main() -> None:
    """Chord/context diagnosis by default.

    ``--tangent``       radial-component fix, context target
    ``--continuation``  radial-component fix, correct-continuation target
    """
    flags = set(sys.argv[1:])
    if "--continuation" in flags:
        use_tangent, target = True, "correct_ending"
        output_path = CONTINUATION_OUTPUT_PATH
    elif "--tangent" in flags:
        use_tangent, target, output_path = True, "context", TANGENT_OUTPUT_PATH
    else:
        use_tangent, target, output_path = False, "context", OUTPUT_PATH

    payload = run_diagnosis(use_tangent=use_tangent, target=target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nDiagnosis saved to {output_path}")
    print(f"({payload['seconds']:.1f}s for {payload['num_examples']} examples = "
          f"{1000 * payload['seconds'] / payload['num_examples']:.1f} ms/example)")
    print("Navigation diagnosis complete.")


if __name__ == "__main__":
    main()


