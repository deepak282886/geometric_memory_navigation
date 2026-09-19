"""TASK 5 check: drive the parallel Jev sampler against the memory store.

Loads ``checkpoints/memory_store.pkl`` (no VAE required), queries it with a
seeded random unit vector, and prints the weight / direction stats that the
sampler produces. Also cross-checks the parallel path against a sequential
reference and exercises the non-positive-weight guard.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.store import MemoryStore  # noqa: E402
from src.sampler.jev import (  # noqa: E402
    MIN_WEIGHT,
    cosine_similarity,
    parallel_sample,
    sample_memory,
    sample_sequentially,
)

STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store.pkl"
VAL_PATH = PROJECT_ROOT / "data" / "val.json"

VAL_INDEX = 42
TOP_K = 10
SEED = 0
CTX_CHARS = 80


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    store = MemoryStore.load(STORE_PATH)
    with VAL_PATH.open(encoding="utf-8") as handle:
        val_data = json.load(handle)
    example = val_data[VAL_INDEX]

    print(f"\nVal example {VAL_INDEX}: activity_label={example['activity_label']!r}, "
          f"label={example['label']}")
    print(f"  ctx: {clip(example['ctx'], CTX_CHARS)!r}")
    print("  (the random dummy query below is intentionally unrelated to this example)")

    # ------------------------------------------------------- dummy query vector
    rng = np.random.default_rng(SEED)
    query_vector = rng.standard_normal(store.dim)
    query_vector /= np.linalg.norm(query_vector)
    query_vector = query_vector.astype(np.float32)
    print(f"\nDummy query vector: shape {query_vector.shape}, "
          f"norm {np.linalg.norm(query_vector):.6f}, dtype {query_vector.dtype}")

    memories = store.query(query_vector, k=TOP_K)
    print(f"store.query -> {len(memories)} memories (k={TOP_K})")
    for rank, memory in enumerate(memories, start=1):
        print(f"  retrieved {rank:>2d}: sim={memory['distance']:.4f} "
              f"{memory['activity_label']!r}")

    outcome_weights = [1.0] * len(memories)
    print(f"outcome_weights: {len(outcome_weights)} entries, all 1.0 (inference setting)")

    # ------------------------------------------------------- parallel sampling
    started = time.perf_counter()
    sampled = parallel_sample(
        query_vector, memories, outcome_weights, target="context"
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    print(f"parallel_sample wall time: {wall_ms:.3f} ms for {len(sampled)} memories")

    print("\nrank |     weight | direction norm | direction mean | sim    | activity")
    print("-" * 86)
    weights: List[float] = []
    norms: List[float] = []
    for rank, ((weight, direction), memory) in enumerate(zip(sampled, memories), start=1):
        norm = float(np.linalg.norm(direction))
        weights.append(weight)
        norms.append(norm)
        print(f"{rank:>4d} | {weight:>10.6f} | {norm:>14.6f} | "
              f"{float(direction.mean()):>+14.6f} | {memory['distance']:.4f} | "
              f"{memory['activity_label']!r}")

    weights_array = np.asarray(weights, dtype=np.float64)
    norms_array = np.asarray(norms, dtype=np.float64)

    # ----------------------------------------------------------- verification
    print("\n" + "=" * 86)
    print("VERIFICATION")
    print("=" * 86)
    all_positive = bool((weights_array > 0.0).all())
    all_float = all(isinstance(weight, float) for weight, _ in sampled)
    all_dim = all(np.asarray(direction).shape == (store.dim,) for _, direction in sampled)
    print(f"  all weights positive      : {all_positive}")
    print(f"  weights are python floats : {all_float}")
    print(f"  weight range              : [{weights_array.min():.6f}, "
          f"{weights_array.max():.6f}]")
    print(f"  direction norm range      : [{norms_array.min():.6f}, "
          f"{norms_array.max():.6f}]")
    print(f"  direction mean range      : [{min(d.mean() for _, d in sampled):+.6f}, "
          f"{max(d.mean() for _, d in sampled):+.6f}]")
    print(f"  every direction shape == ({store.dim},): {all_dim}")
    print(f"  direction dtype           : {sampled[0][1].dtype}")

    # cosine_similarity must agree with np.dot and with the FAISS distances
    sims = [cosine_similarity(query_vector, memory["vector"]) for memory in memories]
    dots = [
        float(np.dot(query_vector.astype(np.float64), memory["vector"].astype(np.float64)))
        for memory in memories
    ]
    faiss_distances = [memory["distance"] for memory in memories]
    print(f"  cosine_similarity == np.dot        : {bool(np.allclose(sims, dots))}")
    print(f"  cosine_similarity == FAISS distance: "
          f"{bool(np.allclose(sims, faiss_distances, atol=1e-6))}")

    # weight must equal similarity * outcome_weight (both here are 1.0)
    expected = [max(sim * 1.0, MIN_WEIGHT) for sim in sims]
    print(f"  weight == max(sim * w, 1e-6)       : "
          f"{bool(np.allclose(weights_array, expected, atol=1e-12))}")

    # parallel path must match the sequential reference exactly
    sequential = sample_sequentially(
        query_vector, memories, outcome_weights, target="context"
    )
    same_weights = all(
        abs(a[0] - b[0]) < 1e-12 for a, b in zip(sampled, sequential)
    )
    same_directions = all(
        np.allclose(a[1], b[1]) for a, b in zip(sampled, sequential)
    )
    print(f"  parallel matches sequential (weight)   : {same_weights}")
    print(f"  parallel matches sequential (direction) : {same_directions}")

    # ------------------------------------------------------------ guard tests
    negative_weight, negative_direction = sample_memory(
        query_vector, {"vector": (-query_vector).astype(np.float32)}, 1.0, target="context"
    )
    zero_weight, zero_direction = sample_memory(
        query_vector, {"vector": np.zeros(store.dim, dtype=np.float32)}, 1.0, target="context"
    )
    print(f"\n  guard - negated memory: dot={cosine_similarity(query_vector, -query_vector):+.6f} "
          f"-> weight {negative_weight:.3e} (clamped: {negative_weight == MIN_WEIGHT})")
    print(f"  guard - zero memory   : dot=+0.000000 -> weight {zero_weight:.3e} "
          f"(clamped: {zero_weight == MIN_WEIGHT})")
    print(f"  guard - direction still computed: "
          f"{np.asarray(negative_direction).shape}, {np.asarray(zero_direction).shape}")

    try:
        parallel_sample(query_vector, memories, [1.0] * 3, target="context")
        print("  length mismatch detected: NO (bug)")
    except ValueError:
        print("  length mismatch raises ValueError: True")

    print(f"\n  empty input handled: "
          f"{parallel_sample(query_vector, [], [], target='context') == []}")
    print("\nSampler test complete.")


if __name__ == "__main__":
    main()
