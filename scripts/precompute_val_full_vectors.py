"""Precompute unit query/ending vectors for the full 10,042-example val split.

Encoding is the expensive part (~18 ms/example), so results are cached as chunk
files and existing chunks are skipped - an interrupted run resumes instead of
starting over.

Each chunk holds
    queries  (B, 256)      unit-normalised ``mu`` of the context
    endings  (B, 4, 256)   unit-normalised ``mu`` of the four endings

Encoding is done one context at a time and four endings at a time, i.e. exactly
the batch shapes ``score_example`` uses, so the first 500 examples reproduce the
TASK 8/9 numbers bit-for-bit rather than drifting on batching noise.
"""

from __future__ import annotations

import json
import math
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
from src.memory.store import normalize_rows  # noqa: E402

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
CHUNKS_DIR = PROJECT_ROOT / "checkpoints" / "val_full_vec_chunks"

CHUNK_SIZE = 1000
LATENT_DIM = 256
NUM_ENDINGS = 4


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


def chunk_path(index: int) -> Path:
    return CHUNKS_DIR / f"chunk_{index:04d}.npz"


def load_all_chunks(num_examples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the cached chunks back into ``(N, 256)`` and ``(N, 4, 256)``."""
    num_chunks = math.ceil(num_examples / CHUNK_SIZE)
    queries = np.zeros((num_examples, LATENT_DIM), dtype=np.float32)
    endings = np.zeros((num_examples, NUM_ENDINGS, LATENT_DIM), dtype=np.float32)
    for index in range(num_chunks):
        path = chunk_path(index)
        if not path.exists():
            raise FileNotFoundError(f"missing vector chunk {path}")
        with np.load(path) as data:
            start = index * CHUNK_SIZE
            stop = start + data["queries"].shape[0]
            queries[start:stop] = data["queries"]
            endings[start:stop] = data["endings"]
    return queries, endings


def main() -> None:
    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)
    print(f"{len(val_data)} validation examples to encode")

    vae = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    num_chunks = math.ceil(len(val_data) / CHUNK_SIZE)
    overall_started = time.perf_counter()

    for index in range(num_chunks):
        path = chunk_path(index)
        start = index * CHUNK_SIZE
        stop = min(start + CHUNK_SIZE, len(val_data))
        if path.exists():
            print(f"  chunk {index + 1}/{num_chunks}: cached, skipping", flush=True)
            continue

        chunk_started = time.perf_counter()
        queries = np.zeros((stop - start, LATENT_DIM), dtype=np.float32)
        endings = np.zeros((stop - start, NUM_ENDINGS, LATENT_DIM), dtype=np.float32)
        with torch.no_grad():
            for offset, example in enumerate(val_data[start:stop]):
                queries[offset] = normalize_rows(
                    vae.encode_texts([example["ctx"]]).cpu().numpy()
                )[0]
                endings[offset] = normalize_rows(
                    vae.encode_texts(list(example["endings"])).cpu().numpy()
                )
        np.savez(path, queries=queries, endings=endings)
        elapsed = time.perf_counter() - chunk_started
        print(
            f"  chunk {index + 1}/{num_chunks}: {stop - start} examples in "
            f"{elapsed:.1f}s ({1000 * elapsed / (stop - start):.1f} ms/example)",
            flush=True,
        )

    print(f"\ntotal encode wall time: {time.perf_counter() - overall_started:.1f}s")
    queries, endings = load_all_chunks(len(val_data))
    print(f"assembled queries {queries.shape} | endings {endings.shape}")
    print(f"finite: {bool(np.isfinite(queries).all() and np.isfinite(endings).all())}")
    print(f"query norms  min {np.linalg.norm(queries, axis=1).min():.6f} "
          f"max {np.linalg.norm(queries, axis=1).max():.6f}")
    print(f"ending norms min {np.linalg.norm(endings, axis=2).min():.6f} "
          f"max {np.linalg.norm(endings, axis=2).max():.6f}")
    print("Vector cache ready.")


if __name__ == "__main__":
    main()
