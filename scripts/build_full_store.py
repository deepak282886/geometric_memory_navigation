"""EXPERIMENT: build a memory store over the full 39,905-example train split.

No new parameters and no retraining - just a bigger FAISS index, so the sampler
has far more directions to choose from.

    data/train_full.json  --(frozen VAE, L2-normalised)-->  (39905, 256)
                          --(IndexFlatIP)-->  checkpoints/memory_store_full.pkl

v2 additionally encodes each memory's KNOWN-CORRECT continuation
(``endings[label]``), because navigation should point at correct continuations
rather than at neighbouring contexts:

    + correct_ending_vectors (39905, 256)  ->  checkpoints/memory_store_full_v2.pkl

When the v1 store already exists its context vectors are reused, so the only
difference between v1 and v2 is the navigation target.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402

TRAIN_FULL_PATH = PROJECT_ROOT / "data" / "train_full.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full.pkl"
#: v2 additionally stores each memory's KNOWN-CORRECT continuation, which is what
#: navigation targets (see src/sampler/jev.py)
V2_STORE_PATH = PROJECT_ROOT / "checkpoints" / "memory_store_full_v2.pkl"
ENCODE_BATCH_SIZE = 128  # larger than MemoryStore's default 64 for throughput


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild the VAE from the checkpoint config (same pattern as TASK 3/4/6/7/8)."""
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


def main() -> None:
    device = torch.device("cpu")
    with TRAIN_FULL_PATH.open(encoding="utf-8") as handle:
        train_data = json.load(handle)
    print(f"Loaded {len(train_data)} examples from {TRAIN_FULL_PATH.name}")

    vae, checkpoint = load_vae(CHECKPOINT_PATH, device)
    print(f"VAE loaded (latent_dim={checkpoint['config']['latent_dim']}, "
          f"encoder frozen={vae.encoder.is_frozen})")

    contexts = [str(example["ctx"]) for example in train_data]

    # Reuse the existing context vectors when available: the v1 store holds the
    # same rows in the same order, so reusing them means the ONLY thing that
    # changes in v2 is the navigation target (no retrieval drift at all).
    encode_seconds = 0.0
    if STORE_PATH.exists():
        previous = MemoryStore.load(STORE_PATH)
        if previous.size != len(train_data) or previous.ctxs != contexts:
            raise RuntimeError(
                f"{STORE_PATH.name} does not match {TRAIN_FULL_PATH.name}; "
                "delete it to force a full re-encode"
            )
        vectors = previous.vectors.copy()
        print(f"\nReusing {vectors.shape} context vectors from {STORE_PATH.name} "
              f"(identical rows and order -> isolates the target change)")
    else:
        print(f"\nEncoding {len(contexts)} ctx strings "
              f"(batch {ENCODE_BATCH_SIZE}) - this takes a few minutes ...")
        encode_started = time.perf_counter()
        with torch.no_grad():
            raw = vae.encode_texts(contexts, batch_size=ENCODE_BATCH_SIZE).cpu().numpy()
        encode_seconds = time.perf_counter() - encode_started
        vectors = normalize_rows(raw).astype(np.float32)
        print(f"  encoded: {vectors.shape} in {encode_seconds:.1f}s "
              f"({len(contexts) / encode_seconds:.0f} texts/s)")

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  finite {bool(np.isfinite(vectors).all())} | row norms "
          f"min {norms.min():.6f} max {norms.max():.6f} "
          f"(unit: {bool(np.allclose(norms, 1.0, atol=1e-5))})")

    # ---------------- the new work for v2: correct-continuation latents -------
    correct_endings = [
        str(example["endings"][int(example["label"])]) for example in train_data
    ]
    print(f"\nEncoding {len(correct_endings)} CORRECT ENDINGS "
          f"(batch {ENCODE_BATCH_SIZE}) - this is the new work for v2 ...")
    ending_started = time.perf_counter()
    with torch.no_grad():
        raw_endings = vae.encode_texts(
            correct_endings, batch_size=ENCODE_BATCH_SIZE
        ).cpu().numpy()
    ending_seconds = time.perf_counter() - ending_started
    correct_ending_vectors = normalize_rows(raw_endings).astype(np.float32)
    ending_norms = np.linalg.norm(correct_ending_vectors, axis=1)
    print(f"  encoded: {correct_ending_vectors.shape} in {ending_seconds:.1f}s "
          f"({len(correct_endings) / ending_seconds:.0f} texts/s)")
    print(f"  finite {bool(np.isfinite(correct_ending_vectors).all())} | row norms "
          f"min {ending_norms.min():.6f} max {ending_norms.max():.6f} "
          f"(unit: {bool(np.allclose(ending_norms, 1.0, atol=1e-5))})")

    print("\nBuilding the FAISS index + v2 store ...")
    build_started = time.perf_counter()
    store = MemoryStore.from_vectors(
        vectors=vectors,
        labels=[int(example["label"]) for example in train_data],
        activity_labels=[str(example["activity_label"]) for example in train_data],
        endings=[list(example["endings"]) for example in train_data],
        ctxs=contexts,
        source=f"train_full.json ({len(train_data)}) + correct endings",
        correct_ending_vectors=correct_ending_vectors,
    )
    build_seconds = time.perf_counter() - build_started

    info = store.summary()
    print("\nStore summary:")
    print(f"  size / vectors shape : {info['size']} / {info['vector_shape']}")
    print(f"  index type           : {info['index_type']}")
    print(f"  normalised           : {info['normalized']}")
    print(f"  unique activities    : {info['unique_activity_labels']}")
    print(f"  label distribution   : {info['label_counts']}")
    print(f"  index build time     : {build_seconds:.2f}s")
    print(f"  has correct endings  : {store.has_correct_ending_vectors} "
          f"(shape {None if store.correct_ending_vectors is None else store.correct_ending_vectors.shape})")

    save_started = time.perf_counter()
    store.save(V2_STORE_PATH)
    save_seconds = time.perf_counter() - save_started
    size_mb = V2_STORE_PATH.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 64)
    if encode_seconds:
        print(f"CONTEXT ENCODING TIME ({len(contexts)} examples)  : {encode_seconds:.1f}s")
    else:
        print(f"CONTEXT ENCODING TIME ({len(contexts)} examples)  : reused from v1, skipped")
    print(f"CORRECT-ENDING ENCODING TIME ({len(correct_endings)} endings) : "
          f"{ending_seconds:.1f}s ({ending_seconds / 60:.1f} min)")
    print(f"INDEX SIZE                                        : {info['size']} vectors")
    print(f"PICKLE                                            : {size_mb:.1f} MB "
          f"(saved in {save_seconds:.1f}s)")
    print(f"SAVED TO                                          : {V2_STORE_PATH}")
    print("=" * 64)
    print("Full store v2 build complete.")


if __name__ == "__main__":
    main()
