"""TASK 13: train a *structured* VAE with a margin ranking loss.

Motivation
----------
The TASK 1-12 VAE trains on reconstruction + KL only, so its latent geometry
carries no signal about which ending is correct. Measured honestly, navigation
therefore adds nothing: ``no_navigation`` scores 29.73% and the best navigated
model 28.58% (``experiments/results_honest.json``). Hypothesis: induce the
structure the scorer needs directly, then navigation becomes trivial.

The scorer (``src/eval/evaluate.py::score_example``) ranks endings by

    context_score_k = cos( unit_encode(ending_k), unit_encode(ctx) )

and the 29.73% baseline is ``argmax_k context_score_k``. So this script adds

    L_structure = mean( max(0, cos(ctx, wrong) - cos(ctx, correct) + margin) )

which is satisfied only once ``cos(ctx, correct) > cos(ctx, wrong) + margin``.

Encoding convention (important)
-------------------------------
The scorer encodes each ending **on its own** (``encode_texts(list(endings))``),
whereas the VAE reconstructs ``ctx + " " + ending`` (TASK 2 behaviour). The margin
loss must therefore be computed on the *ending-alone* encoding - that is the
geometry ``no_navigation`` actually reads. Reconstruction stays byte-for-byte
identical to ``scripts/train_vae.py`` (same 5 texts per example, same KL weight,
same warm-up, same logvar bias) so the only difference from the original model is
the added structure term: a clean ablation.

Recipe (TASK 13)
----------------
* fresh VAE, ``all-MiniLM-L6-v2`` frozen base, Adam lr=1e-3
* data/train_full.json (39,905 examples); base embeddings precomputed once and
  cached to ``checkpoints/train_full_base_embeddings.npz``
* total = recon + 0.001 * kl + alpha * structure
* alpha: epochs 1-2 -> 0.0, 3-5 -> 0.1, 6-10 -> 0.5
* 10 epochs, log every 200 steps, save best-by-structure-loss + final model

TASK 14 update (v2): all wrong endings
--------------------------------------
TASK 13 trained against ONE randomly sampled wrong ending per example. That left
the other two unconstrained, which showed up as a wider *mean* gap but a much
larger per-example gap variance and no ranking gain on the 500-example check
(probe gap +0.057 -> +0.325, yet rank-1 fell 37.6% -> 36.0%).

``--wrong-mode all`` (the default here) instead uses ALL THREE wrong endings::

    L_i = mean_k max(0, cos(ctx_i, wrong_ik) - cos(ctx_i, correct_i) + margin)

so every wrong ending must fall below the correct one (3x the gradient signal).
``--wrong-mode one`` reproduces TASK 13 exactly.

Artifacts (default tag ``v2``)
------------------------------
    checkpoints/vae_structured_v2.pt        best epoch (lowest probe structure loss)
    checkpoints/vae_structured_v2_final.pt  final epoch
    experiments/structured_training_v2.json training summary
    checkpoints/train_full_base_embeddings.npz  reusable frozen-encoder cache
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import (  # noqa: E402
    BASE_EMBED_DIM,
    BASE_MODEL_NAME,
    DECODER_HIDDEN_DIM,
    DEFAULT_STRUCTURE_MARGIN,
    HEAD_HIDDEN_DIM,
    KL_WEIGHT,
    LATENT_DIM,
    VAE,
    structure_loss,
    structure_loss_all_wrong,
    vae_loss,
)

DATA_PATH = PROJECT_ROOT / "data" / "train_full.json"
EMBED_CACHE_PATH = PROJECT_ROOT / "checkpoints" / "train_full_base_embeddings.npz"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
# TASK 13 reference run (one wrong ending) for the final gap comparison.
V1_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vae_structured.pt"
V1_FINAL_PROBE_GAP_FALLBACK = 0.324668

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOG_EVERY = 200
EMBED_BATCH_SIZE = 128
SEED = 42
MARGIN = DEFAULT_STRUCTURE_MARGIN
NUM_ENDINGS = 4
# TASK 14: "all" = every wrong ending (3x signal), "one" = TASK 13 behaviour.
WRONG_MODE = "all"
# Artifact tag: ``vae_structured_<TAG>.pt`` / ``structured_training_<TAG>.json``.
TAG = "v2"
# Same posterior-collapse mitigations as scripts/train_vae.py (unchanged ablation).
WARMUP_EPOCHS = 4
LOGVAR_BIAS = -6.0
# End-of-epoch probe: deterministic so "epoch 1" vs "epoch 10" is comparable.
PROBE_SIZE = 1024

# (last_epoch, alpha) - the schedule from TASK 13.
ALPHA_SCHEDULE: Tuple[Tuple[int, float], ...] = ((2, 0.0), (5, 0.1), (EPOCHS, 0.5))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def alpha_for_epoch(epoch: int) -> float:
    """Structure-loss weight for a 1-based epoch: 0.0 / 0.1 / 0.5."""
    for last_epoch, alpha in ALPHA_SCHEDULE:
        if epoch <= last_epoch:
            return alpha
    return ALPHA_SCHEDULE[-1][1]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI overrides. Defaults reproduce the TASK 14 ``all``-wrong run (tag ``v2``).

    ``--wrong-mode one`` reproduces TASK 13 exactly (tag would then be ``v1``).
    """
    parser = argparse.ArgumentParser(
        description="Train a structured (margin-ranking) VAE on HellaSwag."
    )
    parser.add_argument("--tag", default=TAG, help="artifact tag -> vae_structured_<tag>.pt")
    parser.add_argument(
        "--wrong-mode",
        choices=("one", "all"),
        default=WRONG_MODE,
        help="'one' = TASK 13 single sampled wrong ending, 'all' = all 3 wrong endings",
    )
    parser.add_argument("--margin", type=float, default=MARGIN, help="ranking margin")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of epochs")
    return parser.parse_args(argv)


def load_train_data(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/load_full_data.py first")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_text_lists(
    train_data: List[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """``(ctx_texts, concat_texts, ending_texts)``; last two are ``(N, 4)`` lists.

    * ``ctx_texts``      - ``ctx`` alone (reconstruction, TASK 2)
    * ``concat_texts``   - ``ctx + " " + ending`` (reconstruction, TASK 2)
    * ``ending_texts``   - ``ending`` alone (the scorer's convention, TASK 13)
    """
    ctx_texts: List[str] = []
    concat_texts: List[List[str]] = []
    ending_texts: List[List[str]] = []
    for example in train_data:
        ctx = example["ctx"].strip()
        ctx_texts.append(ctx)
        concat_texts.append([f"{ctx} {ending.strip()}" for ending in example["endings"]])
        ending_texts.append([ending.strip() for ending in example["endings"]])
    return ctx_texts, concat_texts, ending_texts


def flatten_ending_texts(text: List[List[str]]) -> List[str]:
    return [item for row in text for item in row]


def load_or_build_embeddings(
    vae: VAE, train_data: List[Dict[str, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen base embeddings ``(ctx (N,384), concat (N,4,384), ending (N,4,384), labels)``.

    Encoding is the only expensive step (~1 ms/text, ~360k texts), so the result
    is cached to ``EMBED_CACHE_PATH``; a re-run resumes instantly.
    """
    num_examples = len(train_data)
    if EMBED_CACHE_PATH.exists():
        with np.load(EMBED_CACHE_PATH) as cached:
            if cached["ctx"].shape[0] == num_examples:
                print(f"Loaded cached base embeddings from {EMBED_CACHE_PATH}")
                ctx = torch.from_numpy(cached["ctx"])
                concat = torch.from_numpy(cached["concat"])
                ending = torch.from_numpy(cached["ending"])
                labels = torch.from_numpy(cached["labels"]).long()
                return ctx, concat, ending, labels
        print(f"Cached embeddings at {EMBED_CACHE_PATH} are stale - rebuilding")

    ctx_texts, concat_texts, ending_texts = build_text_lists(train_data)
    print(
        f"Precomputing frozen {BASE_EMBED_DIM}-d embeddings: "
        f"{num_examples} ctx + {num_examples * NUM_ENDINGS} concat "
        f"+ {num_examples * NUM_ENDINGS} ending texts"
    )
    ctx_vectors = vae.embed(ctx_texts, batch_size=EMBED_BATCH_SIZE)
    concat_vectors = vae.embed(
        flatten_ending_texts(concat_texts), batch_size=EMBED_BATCH_SIZE
    ).reshape(num_examples, NUM_ENDINGS, BASE_EMBED_DIM)
    ending_vectors = vae.embed(
        flatten_ending_texts(ending_texts), batch_size=EMBED_BATCH_SIZE
    ).reshape(num_examples, NUM_ENDINGS, BASE_EMBED_DIM)

    labels = torch.tensor([int(example["label"]) for example in train_data], dtype=torch.long)
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBED_CACHE_PATH,
        ctx=ctx_vectors.numpy(),
        concat=concat_vectors.numpy(),
        ending=ending_vectors.numpy(),
        labels=labels.numpy(),
    )
    size_mb = EMBED_CACHE_PATH.stat().st_size / (1024 * 1024)
    print(f"Cached embeddings to {EMBED_CACHE_PATH} ({size_mb:.1f} MB)")
    return ctx_vectors, concat_vectors, ending_vectors, labels


def sample_wrong_indices(labels: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """One wrong-ending index per label, guaranteed ``!= label``.

    ``wrong = (label + offset) % 4`` with ``offset in {1, 2, 3}`` can never equal
    ``label``, so no rejection loop is needed.
    """
    offset = torch.randint(1, NUM_ENDINGS, labels.shape, generator=generator)
    return (labels + offset) % NUM_ENDINGS


def all_wrong_indices(labels: torch.Tensor) -> torch.Tensor:
    """``(N, 3)`` indices of every ending except the correct one.

    Column order is stable (ascending, skipping ``label``), so the probe is fully
    deterministic.
    """
    rows = torch.arange(NUM_ENDINGS).unsqueeze(0).expand(labels.shape[0], NUM_ENDINGS)
    return rows[rows != labels.unsqueeze(1)].reshape(labels.shape[0], NUM_ENDINGS - 1)


def structure_stats(
    vae: VAE,
    ctx_vectors: torch.Tensor,
    ending_vectors: torch.Tensor,
    labels: torch.Tensor,
    wrong_indices: torch.Tensor,
    margin: float,
) -> Dict[str, float]:
    """Detached probe of the induced geometry (deterministic ``mu`` vectors).

    ``wrong_indices`` is ``(N,)`` for the one-wrong (TASK 13) probe or ``(N, 3)``
    for the all-wrong (TASK 14) probe. ``gap`` is the mean over whichever wrong
    endings were supplied, and ``per_example_gap_std`` is the spread of the
    per-example gap ``cos+ - mean_k cos_wrong_k`` - the quantity that TASK 13's
    single-negative loss inflated (0.3107 on the 500-example geometry check).
    """
    with torch.no_grad():
        ctx_mu, _ = vae.head(ctx_vectors)
        rows = torch.arange(ctx_vectors.shape[0])
        correct_mu, _ = vae.head(ending_vectors[rows, labels])
        cos_correct = F.cosine_similarity(ctx_mu, correct_mu, dim=-1)  # (N,)

        if wrong_indices.dim() == 1:
            wrong_mu, _ = vae.head(ending_vectors[rows, wrong_indices])
            cos_wrong = F.cosine_similarity(ctx_mu, wrong_mu, dim=-1).unsqueeze(1)  # (N,1)
        else:
            # (N, K, 384) via broadcast advanced indexing
            wrong_mu, _ = vae.head(ending_vectors[rows[:, None], wrong_indices])
            cos_wrong = F.cosine_similarity(ctx_mu.unsqueeze(1), wrong_mu, dim=-1)  # (N,K)

        cos_correct_2d = cos_correct.unsqueeze(1)
        hinge = cos_wrong - cos_correct_2d + margin
        loss = F.relu(hinge).mean()  # identical to structure_loss*'s hinge
        per_example_gap = cos_correct - cos_wrong.mean(dim=1)

    return {
        "structure_loss": float(loss),
        "mean_cos_correct": float(cos_correct.mean()),
        "mean_cos_wrong": float(cos_wrong.mean()),
        "gap": float(cos_correct.mean() - cos_wrong.mean()),
        "per_example_gap_std": float(per_example_gap.std()),
        "margin_violation_rate": float((hinge > 0.0).float().mean()),
    }


def save_checkpoint(
    vae: VAE,
    path: Path,
    extras: Dict[str, Any],
    margin: float,
    wrong_mode: str,
    tag: str,
) -> float:
    """Mirror ``checkpoints/vae.pt``'s layout so TASK 14 can load this verbatim."""
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
        "model_state_dict": model_state,
        "encoder_frozen": bool(vae.encoder.is_frozen),
        "structured": True,
        "structure_margin": margin,
        "wrong_mode": wrong_mode,
        "tag": tag,
        "alpha_schedule": [[last, alpha] for last, alpha in ALPHA_SCHEDULE],
        **extras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    args = parse_args()
    tag = args.tag
    wrong_mode = args.wrong_mode
    margin = args.margin
    epochs = args.epochs
    checkpoint_path = CHECKPOINTS_DIR / f"vae_structured_{tag}.pt"
    final_checkpoint_path = CHECKPOINTS_DIR / f"vae_structured_{tag}_final.pt"
    summary_path = EXPERIMENTS_DIR / f"structured_training_{tag}.json"

    set_seed(SEED)

    train_data = load_train_data(DATA_PATH)
    print(f"Loaded {len(train_data)} train examples from {DATA_PATH}")

    device = torch.device("cpu")
    vae = VAE(device=device, logvar_bias=LOGVAR_BIAS).to(device)
    n_head = sum(p.numel() for p in vae.head.parameters())
    n_decoder = sum(p.numel() for p in vae.decoder.parameters())
    print(f"Base encoder  : {BASE_MODEL_NAME} (frozen={vae.encoder.is_frozen})")
    print(f"Fresh params  : VAE head {n_head:,} | VAE decoder {n_decoder:,}")
    print(f"Latent dim    : {LATENT_DIM} | KL weight {KL_WEIGHT}")
    print(
        f"Structure     : margin={margin} | wrong_mode={wrong_mode} "
        f"({'all 3 wrong endings' if wrong_mode == 'all' else '1 sampled wrong ending'})"
    )
    print(
        f"Artifacts     : {checkpoint_path.name} | {final_checkpoint_path.name} | "
        f"{summary_path.name}"
    )

    started = time.perf_counter()
    ctx_emb, concat_emb, ending_emb, labels = load_or_build_embeddings(vae, train_data)
    print(
        f"Embeddings ready in {time.perf_counter() - started:.1f}s | "
        f"ctx {tuple(ctx_emb.shape)} | concat {tuple(concat_emb.shape)} | "
        f"ending {tuple(ending_emb.shape)} | labels {tuple(labels.shape)}"
    )

    num_examples = ctx_emb.shape[0]
    steps_per_epoch = (num_examples + BATCH_SIZE - 1) // BATCH_SIZE
    optimizer = torch.optim.Adam(vae.trainable_parameters(), lr=LEARNING_RATE)

    # Deterministic end-of-epoch probes (no sampling, so epochs stay comparable):
    #   probe_wrong_primary - matches the training mode (all 3 wrong, or 1)
    #   probe_wrong_one     - always the single ending (label + 1) % 4, i.e. the
    #                         TASK 13 probe, so v1's +0.324668 gap stays comparable
    probe_idx = torch.arange(min(PROBE_SIZE, num_examples))
    probe_labels = labels[probe_idx]
    probe_wrong_one = (probe_labels + 1) % NUM_ENDINGS
    probe_wrong_primary = (
        all_wrong_indices(probe_labels) if wrong_mode == "all" else probe_wrong_one
    )
    row_ids = torch.arange(BATCH_SIZE)

    print(
        f"\nTraining {epochs} epochs | {num_examples} examples | "
        f"{steps_per_epoch} steps/epoch (batch={BATCH_SIZE}, lr={LEARNING_RATE})\n"
    )
    global_step = 0
    epoch_metrics: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    train_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        alpha = alpha_for_epoch(epoch)
        beta = KL_WEIGHT * min(1.0, epoch / WARMUP_EPOCHS) if WARMUP_EPOCHS else KL_WEIGHT
        epoch_started = time.perf_counter()
        vae.head.train()
        vae.decoder.train()

        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = torch.randperm(num_examples, generator=generator)
        sums = {"total": 0.0, "recon": 0.0, "kl": 0.0, "struct": 0.0,
                "cos_correct": 0.0, "cos_wrong": 0.0, "violations": 0.0}
        num_batches = 0

        for start in range(0, num_examples, BATCH_SIZE):
            idx = permutation[start : start + BATCH_SIZE]
            size = idx.shape[0]
            ctx_batch = ctx_emb[idx]                    # (B, 384)
            concat_batch = concat_emb[idx]              # (B, 4, 384)
            ending_batch = ending_emb[idx]              # (B, 4, 384)
            label_batch = labels[idx]                   # (B,)

            # reconstruction: exactly the TASK 2 texts (ctx + 4 x ctx+ending)
            recon_input = torch.cat(
                [ctx_batch, concat_batch.reshape(size * NUM_ENDINGS, BASE_EMBED_DIM)],
                dim=0,
            )                                           # (5B, 384)
            mu_all, logvar_all = vae.head(recon_input)
            z_all = vae.head.reparameterize(mu_all, logvar_all)
            reconstructed = vae.decoder(z_all)
            total, recon_loss, kl_loss = vae_loss(
                reconstructed, recon_input, mu_all, logvar_all, kl_weight=beta
            )

            # structure: scorer convention (ending encoded alone), mu vectors.
            # "all"  -> all 3 wrong endings (TASK 14 fix, 3x gradient signal)
            # "one"  -> 1 sampled wrong ending (TASK 13 behaviour)
            ctx_mu = mu_all[:size]
            correct_mu, _ = vae.head(ending_batch[row_ids[:size], label_batch])
            if wrong_mode == "all":
                wrong_idx = all_wrong_indices(label_batch)                      # (B, 3)
                wrong_mu, _ = vae.head(ending_batch[row_ids[:size, None], wrong_idx])
                struct_loss, cos_correct, cos_wrong = structure_loss_all_wrong(
                    ctx_mu, correct_mu, wrong_mu, margin=margin
                )
                with torch.no_grad():
                    cos_c = F.cosine_similarity(ctx_mu, correct_mu, dim=-1).unsqueeze(1)
                    cos_w = F.cosine_similarity(ctx_mu.unsqueeze(1), wrong_mu, dim=-1)
                    violation_rate = float(((cos_w - cos_c + margin) > 0).float().mean())
            else:
                wrong_idx = sample_wrong_indices(label_batch, generator)
                wrong_mu, _ = vae.head(ending_batch[row_ids[:size], wrong_idx])
                struct_loss, cos_correct, cos_wrong = structure_loss(
                    ctx_mu, correct_mu, wrong_mu, margin=margin
                )
                with torch.no_grad():
                    cos_c = F.cosine_similarity(ctx_mu, correct_mu, dim=-1)
                    cos_w = F.cosine_similarity(ctx_mu, wrong_mu, dim=-1)
                    violation_rate = float(((cos_w - cos_c + margin) > 0).float().mean())

            loss = total + alpha * struct_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            num_batches += 1
            sums["total"] += float(total.detach())
            sums["recon"] += float(recon_loss.detach())
            sums["kl"] += float(kl_loss.detach())
            sums["struct"] += float(struct_loss.detach())
            sums["cos_correct"] += float(cos_correct.detach())
            sums["cos_wrong"] += float(cos_wrong.detach())
            sums["violations"] += violation_rate

            if global_step % LOG_EVERY == 0:
                print(
                    f"  step {global_step:>6d} | total {float(total.detach()):.6f} "
                    f"| recon {float(recon_loss.detach()):.6f} "
                    f"| kl {float(kl_loss.detach()):.4f} "
                    f"| struct {float(struct_loss.detach()):.6f} "
                    f"| cos+ {float(cos_correct.detach()):+.4f} "
                    f"| cos- {float(cos_wrong.detach()):+.4f} "
                    f"| viol {100 * violation_rate:.1f}%",
                    flush=True,
                )

        batches = max(num_batches, 1)
        probe = structure_stats(
            vae,
            ctx_emb[probe_idx],
            ending_emb[probe_idx],
            probe_labels,
            probe_wrong_primary,
            margin,
        )
        probe_one = structure_stats(
            vae,
            ctx_emb[probe_idx],
            ending_emb[probe_idx],
            probe_labels,
            probe_wrong_one,
            margin,
        )
        metrics: Dict[str, float] = {
            "epoch": epoch,
            "alpha": alpha,
            "beta": beta,
            "total": sums["total"] / batches,
            "recon": sums["recon"] / batches,
            "kl": sums["kl"] / batches,
            "struct": sums["struct"] / batches,
            "cos_correct": sums["cos_correct"] / batches,
            "cos_wrong": sums["cos_wrong"] / batches,
            "violations": sums["violations"] / batches,
        }
        metrics.update({f"probe_{key}": value for key, value in probe.items()})
        metrics.update({f"probe1_{key}": value for key, value in probe_one.items()})
        epoch_metrics.append(metrics)

        if probe["structure_loss"] < best_loss:
            best_loss = probe["structure_loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in vae.state_dict().items()
                if not key.startswith("encoder.")
            }

        print(
            f"epoch {epoch:>2d}/{epochs} | alpha {alpha:.2f} | beta {beta:.5f} "
            f"| total {metrics['total']:.6f} | recon {metrics['recon']:.6f} "
            f"| kl {metrics['kl']:.4f} | struct {metrics['struct']:.6f} "
            f"| cos+ {metrics['cos_correct']:+.4f} | cos- {metrics['cos_wrong']:+.4f} "
            f"| probe gap {probe['gap']:+.4f} (std {probe['per_example_gap_std']:.4f}) "
            f"| gap1 {probe_one['gap']:+.4f} "
            f"| viol {100 * probe['margin_violation_rate']:.1f}% "
            f"| {time.perf_counter() - epoch_started:.0f}s",
            flush=True,
        )

    print(f"\nTraining wall time: {time.perf_counter() - train_started:.1f}s")

    # ------------------------------------------------------------- summaries
    first = epoch_metrics[0]
    last = epoch_metrics[-1]
    print("\n" + "=" * 86)
    print(f"STRUCTURE SUMMARY (deterministic probe of {probe_idx.shape[0]} train examples)")
    print("=" * 86)
    print(
        f"  {'epoch':>5} | {'cos+':>9} | {'cos-':>9} | {'gap*':>9} | {'gap std':>8} "
        f"| {'gap1':>9} | {'struct':>8} | {'viol':>7}"
    )
    print("  " + "-" * 82)
    for metrics in epoch_metrics:
        print(
            f"  {metrics['epoch']:>5} | {metrics['probe_mean_cos_correct']:+.6f} "
            f"| {metrics['probe_mean_cos_wrong']:+.6f} | {metrics['probe_gap']:+.6f} "
            f"| {metrics['probe_per_example_gap_std']:>8.4f} "
            f"| {metrics['probe1_gap']:+.6f} "
            f"| {metrics['probe_structure_loss']:.6f} "
            f"| {100 * metrics['probe_margin_violation_rate']:>6.1f}%"
        )
    print("  " + "-" * 82)
    print("  gap*  = mean cos+ - cos- over the TRAINING mode's wrong endings")
    print("  gap1  = same, but against the single ending (label+1)%4 (the TASK 13 probe)")
    print(f"\n  Initial (epoch  1) : cos+ {first['probe_mean_cos_correct']:+.6f} "
          f"vs cos- {first['probe_mean_cos_wrong']:+.6f} -> gap {first['probe_gap']:+.6f} "
          f"(std {first['probe_per_example_gap_std']:.4f})")
    print(f"  Final   (epoch {last['epoch']:>2d}) : cos+ {last['probe_mean_cos_correct']:+.6f} "
          f"vs cos- {last['probe_mean_cos_wrong']:+.6f} -> gap {last['probe_gap']:+.6f} "
          f"(std {last['probe_per_example_gap_std']:.4f})")
    print(f"  Gap growth         : {last['probe_gap'] - first['probe_gap']:+.6f}")
    print(f"  Per-example gap std: {first['probe_per_example_gap_std']:.4f} -> "
          f"{last['probe_per_example_gap_std']:.4f} "
          f"({'LOWER' if last['probe_per_example_gap_std'] < first['probe_per_example_gap_std'] else 'higher'} "
          f"than epoch 1)")
    print(f"  Margin violations  : {100 * last['probe_margin_violation_rate']:.1f}% of pairs "
          f"still violate cos+ > cos- + {margin}")

    # --------------------------------- v1 (one wrong ending) comparison -----
    v1_gap = V1_FINAL_PROBE_GAP_FALLBACK
    v1_source = "fallback constant"
    if V1_CHECKPOINT_PATH.exists():
        v1_checkpoint = torch.load(V1_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if v1_checkpoint.get("final_probe_gap") is not None:
            v1_gap = float(v1_checkpoint["final_probe_gap"])
            v1_source = f"read from {V1_CHECKPOINT_PATH.name}"
    print("\n" + "=" * 86)
    print("vs TASK 13 (one wrong ending)")
    print("=" * 86)
    print(f"  v1 final probe gap        : {v1_gap:+.6f}   ({v1_source})")
    print(f"  this run final gap*       : {last['probe_gap']:+.6f}   (wrong_mode={wrong_mode})")
    print(f"  this run final gap1       : {last['probe1_gap']:+.6f}   "
          f"(same (label+1)%4 probe as v1 -> directly comparable)")
    print(f"  gap*  vs v1               : {last['probe_gap'] - v1_gap:+.6f}")
    print(f"  gap1  vs v1               : {last['probe1_gap'] - v1_gap:+.6f}")
    print(f"  Requested check (gap* > v1): {last['probe_gap'] > v1_gap}")

    # ---------------------------------------------------------------- saving
    try:
        data_ref = str(DATA_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        data_ref = str(DATA_PATH)
    extras: Dict[str, Any] = {
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "data": data_ref,
        "num_examples": num_examples,
        "wrong_mode": wrong_mode,
        "tag": tag,
        "margin": margin,
        "best_epoch": best_epoch,
        "best_probe_structure_loss": best_loss,
        "epoch_metrics": epoch_metrics,
        "initial_probe_mean_cos_correct": first["probe_mean_cos_correct"],
        "initial_probe_mean_cos_wrong": first["probe_mean_cos_wrong"],
        "initial_probe_gap": first["probe_gap"],
        "initial_probe_per_example_gap_std": first["probe_per_example_gap_std"],
        "final_probe_mean_cos_correct": last["probe_mean_cos_correct"],
        "final_probe_mean_cos_wrong": last["probe_mean_cos_wrong"],
        "final_probe_gap": last["probe_gap"],
        "final_probe_per_example_gap_std": last["probe_per_example_gap_std"],
        "final_probe_margin_violation_rate": last["probe_margin_violation_rate"],
        "final_probe1_gap": last["probe1_gap"],
        "v1_final_probe_gap": v1_gap,
    }
    final_state = {
        key: value.detach().cpu()
        for key, value in vae.state_dict().items()
        if not key.startswith("encoder.")
    }
    if best_state is not None:
        vae.load_state_dict(best_state, strict=False)
    best_mb = save_checkpoint(
        vae, checkpoint_path, dict(extras, saved="best"), margin, wrong_mode, tag
    )
    vae.load_state_dict(final_state, strict=False)
    final_mb = save_checkpoint(
        vae, final_checkpoint_path, dict(extras, saved="final"), margin, wrong_mode, tag
    )
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "TASK 14 STEP 1 - structured VAE retrain",
                "tag": tag,
                "wrong_mode": wrong_mode,
                "margin": margin,
                "epochs": epochs,
                "num_examples": num_examples,
                "best_epoch": best_epoch,
                "best_probe_structure_loss": best_loss,
                "initial_probe_gap": first["probe_gap"],
                "final_probe_gap": last["probe_gap"],
                "final_probe1_gap": last["probe1_gap"],
                "final_probe_mean_cos_correct": last["probe_mean_cos_correct"],
                "final_probe_mean_cos_wrong": last["probe_mean_cos_wrong"],
                "final_probe_per_example_gap_std": last["probe_per_example_gap_std"],
                "final_probe_margin_violation_rate": last["probe_margin_violation_rate"],
                "v1_final_probe_gap": v1_gap,
                "gap_vs_v1": last["probe_gap"] - v1_gap,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "epoch_metrics": epoch_metrics,
            },
            handle,
            indent=2,
        )

    print("\n" + "=" * 86)
    print("ARTIFACTS")
    print("=" * 86)
    print(f"  best  (epoch {best_epoch:>2d}) : {checkpoint_path} ({best_mb:.2f} MB) "
          f"| probe struct {best_loss:.6f}")
    print(f"  final (epoch {last['epoch']:>2d}) : {final_checkpoint_path} ({final_mb:.2f} MB)")
    print(f"  training summary : {summary_path}")
    print(f"  embedding cache  : {EMBED_CACHE_PATH}")
    print("\nStructured VAE training complete.")


if __name__ == "__main__":
    main()
