"""Train the GMN VAE head + decoder on HellaSwag texts (TASK 2).

Training texts = ``ctx`` alone plus ``ctx + " " + ending`` for each of the 4
endings  ->  2000 * 5 = 10,000 strings.

The sentence-transformers base encoder is FROZEN, so its 384-d embeddings are
precomputed once and only ``VAEHead`` / ``VAEDecoder`` are optimised.

To avoid posterior collapse (a plain ``torch.sum`` KL, or a logvar bias of 0,
lets the decoder ignore ``z`` entirely) the run uses a negative ``logvar`` bias
initialisation plus a short KL warm-up that ramps ``beta`` 0 -> 0.001.

Artifacts:
    checkpoints/vae.pt                  trained weights + config + losses
    checkpoints/vae_latent_samples.npz  mu / logvar / z for 5 sample texts
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import (  # noqa: E402
    BASE_EMBED_DIM,
    BASE_MODEL_NAME,
    DECODER_HIDDEN_DIM,
    HEAD_HIDDEN_DIM,
    KL_WEIGHT,
    LATENT_DIM,
    VAE,
    vae_loss,
)

DATA_PATH = PROJECT_ROOT / "data" / "train.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae.pt"
LATENT_SAMPLES_PATH = PROJECT_ROOT / "checkpoints" / "vae_latent_samples.npz"

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOG_EVERY = 100
EMBED_BATCH_SIZE = 64
SEED = 42
NUM_SAMPLE_TEXTS = 5

# Posterior-collapse mitigation (see the VAEHead / vae_loss docstrings):
#   * logvar bias starts at -6 so z carries signal before the noise grows
#   * KL warm-up ramps beta linearly 0 -> KL_WEIGHT over the first few epochs.
#     beta never exceeds KL_WEIGHT, so the required weight of 0.001 holds.
WARMUP_EPOCHS = 4
LOGVAR_BIAS = -6.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_train_data(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/load_data.py first")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_training_texts(train_data: List[Dict[str, Any]]) -> List[str]:
    """``ctx`` + 4 x ``ctx ending`` per example."""
    texts: List[str] = []
    for example in train_data:
        ctx = example["ctx"].strip()
        texts.append(ctx)  # ctx alone
        for ending in example["endings"]:
            texts.append(f"{ctx} {ending.strip()}")  # ctx + ending
    return texts


def tensor_stats(tensor: torch.Tensor) -> Tuple[float, float, float, float]:
    """``(min, max, mean, std)`` as plain floats."""
    return (
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.mean()),
        float(tensor.std()),
    )


def main() -> None:
    set_seed(SEED)

    train_data = load_train_data(DATA_PATH)
    texts = build_training_texts(train_data)
    print(f"Loaded {len(train_data)} train examples -> {len(texts)} training texts")

    device = torch.device("cpu")
    vae = VAE(device=device, logvar_bias=LOGVAR_BIAS).to(device)

    n_head = sum(p.numel() for p in vae.head.parameters())
    n_decoder = sum(p.numel() for p in vae.decoder.parameters())
    print(f"Base encoder  : {BASE_MODEL_NAME}")
    print(f"Encoder frozen: {vae.encoder.is_frozen}")
    print(f"Vanilla params: VAE head {n_head:,} | VAE decoder {n_decoder:,}")
    print(f"Latent dim    : {LATENT_DIM} | KL weight: {KL_WEIGHT}")
    print(f"Logvar bias   : {LOGVAR_BIAS} | KL warm-up: {WARMUP_EPOCHS} epochs (beta 0 -> {KL_WEIGHT})")

    # ---------------------------------------------------------------- embeddings
    print(f"\nPrecomputing frozen {BASE_EMBED_DIM}-d embeddings for {len(texts)} texts ...")
    started = time.time()
    base_embeddings = vae.embed(texts, batch_size=EMBED_BATCH_SIZE).to(device)
    print(
        f"Embeddings ready: {tuple(base_embeddings.shape)} in {time.time() - started:.1f}s "
        f"| requires_grad={base_embeddings.requires_grad}"
    )

    loader = DataLoader(
        TensorDataset(base_embeddings),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.Adam(vae.trainable_parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------ training
    print(f"\nTraining for {EPOCHS} epochs (batch={BATCH_SIZE}, lr={LEARNING_RATE}) ...\n")
    global_step = 0
    last_losses = (float("nan"), float("nan"), float("nan"))
    epoch_losses: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        # Linear KL warm-up, capped at KL_WEIGHT so the required 0.001 is respected.
        beta = KL_WEIGHT * min(1.0, epoch / WARMUP_EPOCHS) if WARMUP_EPOCHS else KL_WEIGHT
        vae.head.train()
        vae.decoder.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)

            mu, logvar = vae.head(batch)
            z = vae.head.reparameterize(mu, logvar)
            reconstructed = vae.decoder(z)

            total, reconstruction_loss, kl_loss = vae_loss(
                reconstructed, batch, mu, logvar, kl_weight=beta
            )
            total.backward()
            optimizer.step()

            global_step += 1
            last_losses = (
                float(total.detach()),
                float(reconstruction_loss.detach()),
                float(kl_loss.detach()),
            )
            epoch_total += last_losses[0]
            epoch_recon += last_losses[1]
            epoch_kl += last_losses[2]
            num_batches += 1

            if global_step % LOG_EVERY == 0:
                print(
                    f"  step {global_step:>5d} | total {last_losses[0]:.6f} "
                    f"| recon {last_losses[1]:.6f} | kl {last_losses[2]:.4f}"
                )

        mean_total = epoch_total / max(num_batches, 1)
        epoch_losses.append(mean_total)
        print(
            f"epoch {epoch:>2d}/{EPOCHS} | beta {beta:.5f} "
            f"| mean total {mean_total:.6f} "
            f"| recon {epoch_recon / max(num_batches, 1):.6f} "
            f"| kl {epoch_kl / max(num_batches, 1):.4f}"
        )

    # ----------------------------------------------------------------- summary
    final_total, final_recon, final_kl = last_losses
    print("\n" + "=" * 68)
    print("TRAINING SUMMARY")
    print("=" * 68)
    print(f"Final step loss    : total {final_total:.6f} | recon {final_recon:.6f} | kl {final_kl:.4f}")
    print(f"Last epoch mean    : total {epoch_losses[-1]:.6f}")
    print(f"First epoch mean   : total {epoch_losses[0]:.6f}")
    trend = epoch_losses[0] - epoch_losses[-1]
    print(f"Loss decreased     : {epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f} (delta {trend:+.6f})")
    print(f"Decreasing trend   : {'YES' if trend > 0 else 'NO'}")

    # ------------------------------------------------------------ latent checks
    vae.head.eval()
    vae.decoder.eval()
    sample_texts = texts[:NUM_SAMPLE_TEXTS]
    with torch.no_grad():
        z_sample, mu_sample, logvar_sample = vae.encode(sample_texts)
        recon_sample = vae.decoder(z_sample)

    print(f"\nEncoding {NUM_SAMPLE_TEXTS} sample texts after training:")
    print(f"  z batch shape      : {tuple(z_sample.shape)}   (expected ({NUM_SAMPLE_TEXTS}, {LATENT_DIM}))")
    print(f"  mu batch shape     : {tuple(mu_sample.shape)}")
    print(f"  logvar batch shape : {tuple(logvar_sample.shape)}")

    z_min, z_max, z_mean, z_std = tensor_stats(z_sample)
    mu_min, mu_max, _, _ = tensor_stats(mu_sample)
    lv_min, lv_max, _, _ = tensor_stats(logvar_sample)
    print(f"\n  z across batch : min {z_min:+.4f} | max {z_max:+.4f} | mean {z_mean:+.4f} | std {z_std:.4f}")
    print(f"  mu across batch: min {mu_min:+.4f} | max {mu_max:+.4f}")
    print(f"  logvar across batch: min {lv_min:+.4f} | max {lv_max:+.4f}")

    print(f"\n  Per-sample z vectors ({NUM_SAMPLE_TEXTS} texts):")
    all_finite = True
    any_all_zero = False
    for index, text in enumerate(sample_texts):
        z_i = z_sample[index]
        mn, mx, me, sd = tensor_stats(z_i)
        finite = bool(torch.isfinite(z_i).all())
        all_zero = bool(torch.allclose(z_i, torch.zeros_like(z_i)))
        all_finite = all_finite and finite
        any_all_zero = any_all_zero or all_zero
        print(
            f"    [{index}] z.shape={tuple(z_i.shape)} min={mn:+.4f} max={mx:+.4f} "
            f"mean={me:+.4f} std={sd:.4f} finite={finite} all_zero={all_zero} "
            f"| {text[:58]!r}"
        )
    recon_mse = float(torch.nn.functional.mse_loss(recon_sample, vae.embed(sample_texts)))
    print(f"\n  all z finite : {all_finite}")
    print(f"  any all-zero z vector : {any_all_zero}")
    print(f"  sample reconstruction MSE: {recon_mse:.6f}")

    # ------------------------------------------------- collapsed-posterior check
    # A healthy latent varies across inputs and reconstructs better than the
    # best constant predictor (the dataset mean embedding).
    with torch.no_grad():
        _, mu_all, logvar_all = vae.encode(texts)
        recon_all = vae.decoder(mu_all)
    reference = base_embeddings.detach()
    mean_baseline = float(((reference - reference.mean(dim=0)) ** 2).mean())
    full_recon = float(torch.nn.functional.mse_loss(recon_all, reference))
    improvement = 100.0 * (1.0 - full_recon / mean_baseline) if mean_baseline > 0 else 0.0
    print(f"\n  Posterior health over all {len(texts)} texts:")
    print(f"    mu     std across texts : {float(mu_all.std()):.5f}   (0 == collapsed)")
    print(f"    logvar mean             : {float(logvar_all.mean()):+.4f}")
    print(f"    mu row std (mean)       : {float(mu_all.std(dim=1).mean()):.5f}")
    print(f"    reconstruction MSE      : {full_recon:.6f}")
    print(f"    constant-predictor MSE  : {mean_baseline:.6f}")
    print(f"    improvement over constant: {improvement:.1f}%")
    print(f"    latent is informative   : {improvement > 10.0}")

    # -------------------------------------------------------------------- save
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_state = {
        key: value.detach().cpu()
        for key, value in vae.state_dict().items()
        if not key.startswith("encoder.")
    }
    checkpoint = {
        "config": {
            "base_model_name": BASE_MODEL_NAME,
            "base_embed_dim": BASE_EMBED_DIM,
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "decoder_hidden_dim": DECODER_HIDDEN_DIM,
            "kl_weight": KL_WEIGHT,
            "kl_warmup_epochs": WARMUP_EPOCHS,
            "logvar_bias": LOGVAR_BIAS,
        },
        # head.* and decoder.* only; mu_layer / logvar_layer kept separately
        "model_state_dict": model_state,
        "encoder_frozen": bool(vae.encoder.is_frozen),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "final_loss": final_total,
        "final_reconstruction_loss": final_recon,
        "final_kl_loss": final_kl,
        "epoch_losses": epoch_losses,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    checkpoint_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)

    # mu / logvar / z saved separately so they can be inspected independently
    np.savez(
        LATENT_SAMPLES_PATH,
        mu=mu_sample.numpy(),
        logvar=logvar_sample.numpy(),
        z=z_sample.numpy(),
        texts=np.array(sample_texts, dtype=object),
    )

    print("\n" + "=" * 68)
    print("ARTIFACTS")
    print("=" * 68)
    print(f"Checkpoint            : {CHECKPOINT_PATH}")
    print(f"Checkpoint size       : {checkpoint_mb:.3f} MB")
    print(f"State-dict tensors    : {len(model_state)}")
    print(f"Latent samples (mu/logvar/z): {LATENT_SAMPLES_PATH}")
    print(f"  mu shape={mu_sample.shape} | logvar shape={logvar_sample.shape} | z shape={z_sample.shape}")
    print("\nVAE training complete.")


if __name__ == "__main__":
    main()
