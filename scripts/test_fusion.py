"""TASK 6 check: run the full geometric pipeline and inspect the fused output.

For val examples 42 / 137 / 299:

    encode ctx -> unit query_vector
    store.query(k=10)
    parallel_sample (all outcome_weights = 1.0)
    fuse_directions(top_k=3)
    navigate -> 512-d classifier input

The pipeline timing reported here excludes the (frozen) VAE encoding step, so it
measures only: query + sample + fuse + navigate.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.fusion.fuse import DEFAULT_TOP_K, fuse_directions, navigate  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402
from src.sampler.jev import parallel_sample  # noqa: E402

STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
VAL_PATH = PROJECT_ROOT / "data" / "val.json"

VAL_INDICES: Tuple[int, ...] = (42, 137, 299)
RETRIEVAL_K = 10
FUSE_TOP_K = DEFAULT_TOP_K
TIMING_REPEATS = 200


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config (same pattern as TASK 3/4)."""
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


@torch.no_grad()
def encode_query(vae: VAE, text: str) -> Tuple[np.ndarray, float]:
    """Encode one string and L2-normalise it -> ``(unit_vector, raw_norm)``.

    Normalising matters: the memory store holds unit-norm vectors and
    ``parallel_sample`` computes ``dot(query, memory)`` as the cosine similarity,
    so the query must be unit length too for the weights to be true cosines.
    """
    raw = vae.encode_texts([text]).cpu().numpy()[0].astype(np.float64)
    norm = float(np.linalg.norm(raw))
    return normalize_rows(raw.reshape(1, -1))[0].astype(np.float64), norm


def run_pipeline(
    store: MemoryStore,
    query_vector: np.ndarray,
    retrieval_k: int = RETRIEVAL_K,
    fuse_top_k: int = FUSE_TOP_K,
) -> Dict[str, Any]:
    """query -> sample -> fuse -> navigate (no VAE involved)."""
    memories = store.query(query_vector, k=retrieval_k)
    sampled = parallel_sample(
        query_vector, memories, [1.0] * len(memories), target="context"
    )
    navigation_vector = fuse_directions(sampled, top_k=fuse_top_k)
    classifier_input = navigate(query_vector, navigation_vector)
    return {
        "memories": memories,
        "sampled": sampled,
        "navigation_vector": navigation_vector,
        "classifier_input": classifier_input,
    }


def main() -> None:
    device = torch.device("cpu")
    store = MemoryStore.load(STORE_PATH)
    vae, checkpoint = load_vae(CHECKPOINT_PATH, device)
    with VAL_PATH.open(encoding="utf-8") as handle:
        val_data = json.load(handle)

    print(f"VAE latent_dim={checkpoint['config']['latent_dim']} | store size={store.size}")
    print(f"pipeline: store.query(k={RETRIEVAL_K}) -> parallel_sample -> "
          f"fuse_directions(top_k={FUSE_TOP_K}) -> navigate")

    nav_norms: List[float] = []
    out_shapes: List[Tuple[int, ...]] = []
    any_nan = False

    for val_index in VAL_INDICES:
        example = val_data[val_index]
        query_vector, raw_norm = encode_query(vae, example["ctx"])
        result = run_pipeline(store, query_vector)

        navigation_vector = result["navigation_vector"]
        classifier_input = result["classifier_input"]
        sampled = result["sampled"]
        memories = result["memories"]

        order = np.argsort([-weight for weight, _ in sampled])[:FUSE_TOP_K]
        nav_norm = float(np.linalg.norm(navigation_vector))
        nav_bad = bool(np.isnan(navigation_vector).any() or np.isinf(navigation_vector).any())
        out_bad = bool(np.isnan(classifier_input).any() or np.isinf(classifier_input).any())

        print("\n" + "=" * 88)
        print(f"[val {val_index}] activity_label = {example['activity_label']!r} "
              f"(true label {example['label']})")
        print("=" * 88)
        print(f"  query vector      : shape {query_vector.shape}, raw norm {raw_norm:.4f} "
              f"-> normalised norm {np.linalg.norm(query_vector):.6f}")
        print(f"  retrieved         : {len(memories)} memories (k={RETRIEVAL_K})")
        print(f"  top-{FUSE_TOP_K} fused contributions (of {len(sampled)} sampled):")
        for position, index in enumerate(order, start=1):
            weight, direction = sampled[int(index)]
            print(f"    {position}. weight={weight:.6f} "
                  f"| dir_norm={np.linalg.norm(direction):.6f} "
                  f"| {memories[int(index)]['activity_label']!r}")
        fused_weights = np.asarray([sampled[int(i)][0] for i in order], dtype=np.float64)
        fused_weights = fused_weights / fused_weights.sum()
        print(f"    normalised fused weights: {[round(float(w), 6) for w in fused_weights]} "
              f"(sum {fused_weights.sum():.6f})")
        print(f"  navigation_vector : shape {navigation_vector.shape}, norm {nav_norm:.6f}, "
              f"any NaN/inf {nav_bad}")
        print(f"  navigate output   : shape {classifier_input.shape}, any NaN/inf {out_bad}")

        nav_norms.append(nav_norm)
        out_shapes.append(tuple(classifier_input.shape))
        any_nan = any_nan or nav_bad or out_bad

    # ------------------------------------------------------------- timing
    print("\n" + "=" * 88)
    print(f"PIPELINE TIMING (query + sample + fuse + navigate, VAE encode excluded)")
    print("=" * 88)
    timed_vector = encode_query(vae, val_data[VAL_INDICES[0]]["ctx"])[0]
    run_pipeline(store, timed_vector)  # warm-up

    quiet = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        for _ in range(TIMING_REPEATS):
            run_pipeline(store, timed_vector)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    single_started = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        run_pipeline(store, timed_vector)
    single_ms = (time.perf_counter() - single_started) * 1000.0

    print(f"  single call : {single_ms:.4f} ms")
    print(f"  mean over {TIMING_REPEATS} calls : {elapsed_ms / TIMING_REPEATS:.4f} ms")

    # ------------------------------------------------------- verification
    print("\n" + "=" * 88)
    print("VERIFICATION")
    print("=" * 88)
    print(f"  navigation norms (expect ~1.0) : {[round(n, 6) for n in nav_norms]}")
    print(f"  all norms within 1e-6 of 1.0   : "
          f"{all(abs(n - 1.0) < 1e-6 for n in nav_norms)}")
    print(f"  navigate output shapes         : {out_shapes}")
    print(f"  all shapes == (512,)           : {all(s == (512,) for s in out_shapes)}")
    print(f"  any NaN or inf anywhere        : {any_nan}")
    print("\nFusion test complete.")


if __name__ == "__main__":
    main()
