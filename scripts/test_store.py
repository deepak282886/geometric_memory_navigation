"""TASK 4 check: build/load the memory store and inspect retrieved neighbours.

Builds the store from ``data/train.json`` (or reloads ``checkpoints/memory_store.pkl``),
then queries it with three fixed validation examples and prints their top-3
neighbourhoods so retrieval quality can be judged by eye.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.memory.store import INFERENCE_OUTCOME_WEIGHT, MemoryStore  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "train.json"
VAL_PATH = PROJECT_ROOT / "data" / "val.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store.pkl"

VAL_INDICES: Tuple[int, ...] = (42, 137, 299)
TOP_K = 10
DISPLAY_K = 3
QUERY_CTX_CHARS = 80
NEIGHBOUR_CTX_CHARS = 60
BENCHMARK_REPEATS = 200
SELF_CHECK_SAMPLES = 25
SEED = 42


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config and load head/decoder weights.

    Same loading pattern as scripts/visualize_latent.py (TASK 3).
    """
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


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def encode(vae: VAE, texts: Sequence[str]) -> np.ndarray:
    return vae.encode_texts(list(texts)).cpu().numpy().astype(np.float32)


def main() -> None:
    device = torch.device("cpu")
    with TRAIN_PATH.open(encoding="utf-8") as handle:
        train_data = json.load(handle)
    with VAL_PATH.open(encoding="utf-8") as handle:
        val_data = json.load(handle)

    print(f"Train examples: {len(train_data)} | Val examples: {len(val_data)}")
    vae, checkpoint = load_vae(CHECKPOINT_PATH, device)
    print(f"VAE loaded (latent_dim={checkpoint['config']['latent_dim']}, "
          f"frozen encoder={vae.encoder.is_frozen})")

    # ----------------------------------------------------------- build/load
    print()
    if STORE_PATH.exists():
        store = MemoryStore.load(STORE_PATH)
        print("(reused the cached store; delete checkpoints/memory_store.pkl to rebuild)")
    else:
        store = MemoryStore(vae, train_data)
        store.save(STORE_PATH)

    info = store.summary()
    print("\nStore summary:")
    print(f"  size                  : {info['size']}")
    print(f"  vectors shape         : {info['vector_shape']}")
    print(f"  index type            : {info['index_type']}")
    print(f"  normalized (IP=cosine): {info['normalized']}")
    print(f"  unique activity labels: {info['unique_activity_labels']}")
    print(f"  label distribution    : {info['label_counts']}")
    print(f"  distinct ctxs         : {info['distinct_ctxs']}")

    # ------------------------------------------------------- self-retrieval
    rng = np.random.default_rng(SEED)
    probe = rng.choice(len(train_data), size=SELF_CHECK_SAMPLES, replace=False)
    hits = 0
    top1_similarities: List[float] = []
    for position in probe:
        neighbours = store.query(store.vectors[int(position)], k=1)
        top1_similarities.append(float(neighbours[0]["distance"]))
        hits += int(neighbours[0]["idx"] == int(position))
    print(f"\nIndex sanity: querying a stored vector returns itself as top-1 in "
          f"{hits}/{SELF_CHECK_SAMPLES} cases "
          f"(mean similarity {np.mean(top1_similarities):.4f})")

    # ------------------------------------------------------- full result dump
    sample_query = val_data[VAL_INDICES[0]]
    sample_vector = encode(vae, [sample_query["ctx"]])[0]
    full = store.query(sample_vector, k=TOP_K)
    print(f"\nOne full query result (val index {VAL_INDICES[0]}, k={TOP_K}):")
    for key in ("idx", "label", "activity_label", "distance", "outcome_weight"):
        print(f"  {key:<14s}: {full[0][key]}")
    print(f"  ctx           : {clip(full[0]['ctx'], 60)}")
    print(f"  endings       : {[clip(e, 35) for e in full[0]['endings']]}")
    print(f"  vector        : shape {full[0]['vector'].shape}, dtype {full[0]['vector'].dtype}")

    # -------------------------------------------------------- the 3 queries
    print("\n" + "=" * 100)
    print("TOP-3 NEIGHBOURS FOR 3 VALIDATION EXAMPLES")
    print("=" * 100)

    assessments: List[Tuple[int, int, int]] = []
    for val_index in VAL_INDICES:
        example = val_data[val_index]
        query_vector = encode(vae, [example["ctx"]])[0]
        neighbours = store.query(query_vector, k=TOP_K)
        labels = [n["activity_label"] for n in neighbours]
        matches = sum(1 for label in labels if label == example["activity_label"])

        print(f"\n[val {val_index}] query activity_label: {example['activity_label']!r}")
        print(f"  query ctx        : {clip(example['ctx'], QUERY_CTX_CHARS)!r}")
        print(f"  query true label : {example['label']} "
              f"(inference outcome_weight for every neighbour is {INFERENCE_OUTCOME_WEIGHT})")
        print(f"  neighbours returned: {len(neighbours)} (asked k={TOP_K})")
        print(f"  top-{DISPLAY_K}:")
        for rank, neighbour in enumerate(neighbours[:DISPLAY_K], start=1):
            same = "SAME" if neighbour["activity_label"] == example["activity_label"] else "diff"
            print(
                f"    {rank}. sim={neighbour['distance']:.4f} | "
                f"w={neighbour['outcome_weight']:.1f} | [{same}] "
                f"activity={neighbour['activity_label']!r}"
            )
            print(f"       ctx: {clip(neighbour['ctx'], NEIGHBOUR_CTX_CHARS)!r}")
        print(f"  activity_label matches in top-{TOP_K}: {matches}/{len(neighbours)} "
              f"({100.0 * matches / max(len(neighbours), 1):.0f}%)")
        print("  training-only outcome weights vs query label "
              f"{example['label']}: "
              f"{[store.get_training_outcome_weight(n['label'], example['label']) for n in neighbours]}")
        assessments.append((val_index, matches, len(neighbours)))

    # -------------------------------------------------------------- latency
    bench = store.benchmark_query(store.vectors[:50], repeats=BENCHMARK_REPEATS, k=TOP_K)
    print("\n" + "=" * 100)
    print(f"QUERY LATENCY ({BENCHMARK_REPEATS} single-vector queries, k={TOP_K}, "
          f"{store.size} vectors)")
    print("=" * 100)
    print(f"  mean   : {bench['mean_ms']:.4f} ms")
    print(f"  median : {bench['median_ms']:.4f} ms")
    print(f"  min    : {bench['min_ms']:.4f} ms")
    print(f"  max    : {bench['max_ms']:.4f} ms")

    print("\nSummary of the 3 test queries (activity-label matches in top-10):")
    for val_index, matches, returned in assessments:
        print(f"  val {val_index:>3d}: {matches}/{returned}")
    print("Memory store test complete.")


if __name__ == "__main__":
    main()
