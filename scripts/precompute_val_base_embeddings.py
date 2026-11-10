"""TASK 16 (modified): cache frozen base embeddings for the full val split.

The CLIP-style head is trained/fine-tuned on top of the FROZEN
``all-MiniLM-L6-v2`` base, so the 384-d base embeddings are a constant. Caching
them once turns every subsequent head experiment into a pure matrix operation.

Encodes the *scorer's* text forms, matching ``src/eval/evaluate.py``:
    ctx      -> ``ctx`` alone                (N, 384)
    endings  -> each ending alone            (N, 4, 384)

Batched encoding was verified equivalent to the per-example pipeline used by
``scripts/precompute_val_full_vectors.py`` (max |delta| 3.4e-07, 500/500 identical
argmax), which is what makes the 10x speedup safe.

Artifact: ``checkpoints/val_full_base_embeddings.npz`` (~76 MB).
"""

from __future__ import annotations

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

VAL_FULL_PATH = PROJECT_ROOT / "data" / "val_full.json"
CACHE_PATH = PROJECT_ROOT / "checkpoints" / "val_full_base_embeddings.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
BASE_EMBED_DIM = 384
NUM_ENDINGS = 4
EMBED_BATCH_SIZE = 128


def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
    """Rebuild a VAE from a checkpoint (same recipe as TASK 2/8/13/14)."""
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


def build_text_lists(
    val_data: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """``(ctx_texts, ending_texts)`` where endings are flattened ``(N * 4)``."""
    ctx_texts = [example["ctx"].strip() for example in val_data]
    ending_texts = [
        example["endings"][index].strip()
        for example in val_data
        for index in range(NUM_ENDINGS)
    ]
    return ctx_texts, ending_texts


def main() -> None:
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as cached:
            print(
                f"{CACHE_PATH} already exists: ctx {cached['ctx'].shape} | "
                f"endings {cached['endings'].shape} - nothing to do"
            )
        return

    with VAL_FULL_PATH.open(encoding="utf-8") as handle:
        val_data: List[Dict[str, Any]] = json.load(handle)
    ctx_texts, ending_texts = build_text_lists(val_data)
    print(
        f"Encoding {len(ctx_texts)} ctx + {len(ending_texts)} ending texts "
        f"({len(val_data)} val examples) with the FROZEN base encoder"
    )

    vae = load_vae(CHECKPOINT_PATH, torch.device("cpu"))
    started = time.perf_counter()
    with torch.no_grad():
        ctx_vectors = vae.embed(ctx_texts, batch_size=EMBED_BATCH_SIZE)
        ending_vectors = vae.embed(ending_texts, batch_size=EMBED_BATCH_SIZE)
    elapsed = time.perf_counter() - started
    ending_vectors = ending_vectors.reshape(
        len(val_data), NUM_ENDINGS, BASE_EMBED_DIM
    )
    total = len(ctx_texts) + len(ending_texts)
    print(
        f"Encoded in {elapsed:.1f}s ({1000 * elapsed / total:.2f} ms/text) | "
        f"ctx {tuple(ctx_vectors.shape)} | endings {tuple(ending_vectors.shape)}"
    )

    labels = np.asarray([int(e["label"]) for e in val_data], dtype=np.int64)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        ctx=ctx_vectors.numpy(),
        endings=ending_vectors.numpy(),
        labels=labels,
    )
    size_mb = CACHE_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved {CACHE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
