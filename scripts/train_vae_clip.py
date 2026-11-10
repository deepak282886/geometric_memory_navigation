"""TASK 16 (modified): CLIP-style end-to-end training of the VAE head.

Why
---
The margin-ranking losses (TASK 13/14) widened the correct-vs-wrong cosine gap but
did not improve accuracy (+0.06 / +0.57 points, both non-significant). Diagnosis:
the margin term fights the reconstruction term, so the head *rescales* the latent
space instead of reorganising it. This script removes the compromise and trains the
head on the actual scoring objective.

Architecture / loss (the scorer, made differentiable)
-----------------------------------------------------
    base encoder   : FROZEN all-MiniLM-L6-v2 (384-d)      [precomputed and cached]
    VAE head       : FINE-TUNED (fc, mu_layer, logvar_layer)
    VAE decoder    : ignored

    ctx_vector      = unit(mu_head(base_encode(ctx)))
    ending_vectors  = unit_rows(mu_head(base_encode(ending_k))),  k = 1..4
    scores          = ending_vectors @ ctx_vector
    loss            = CrossEntropy(scores / temperature, label),  temperature = 0.07

CLIP analogy: image encoder vs text encoder -> here ctx encoder vs ending encoder.

Notes
-----
* L2 normalisation makes the loss scale-invariant, so the head cannot cheat by
  inflating ``mu`` magnitudes - it must reorganise *directions*. (This is exactly
  the failure mode the margin loss fell into.)
* ``logvar_layer`` is unfrozen as specified but receives **no gradient**: this loss
  has no KL term and only ``mu`` is used, so it stays at its ``vae.pt`` init.
* Inputs are cached frozen-base embeddings, so a training step is pure head
  forward/backward - 5 epochs over 39,905 examples takes minutes.
* Before training, the untouched ``vae.pt`` head is scored through this harness as
  a self-check: it must reproduce the published 31.31% on 10,042 val.

Artifacts
---------
    checkpoints/vae_clip.pt        best epoch by 500-example rank_1
    checkpoints/vae_clip_final.pt  epoch 5
    experiments/results_clip.json
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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.autoencoder.vae import (  # noqa: E402
    BASE_EMBED_DIM,
    BASE_MODEL_NAME,
    DECODER_HIDDEN_DIM,
    HEAD_HIDDEN_DIM,
    LATENT_DIM,
    VAE,
)
from run_ablations import paired_comparison  # noqa: E402

LOAD_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vae.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae_clip.pt"
FINAL_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae_clip_final.pt"
TRAIN_CACHE = PROJECT_ROOT / "checkpoints" / "train_full_base_embeddings.npz"
VAL_CACHE = PROJECT_ROOT / "checkpoints" / "val_full_base_embeddings.npz"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results_clip.json"
GMN_PRED_PATH = PROJECT_ROOT / "experiments" / "gmn_full_val_predictions.json"
GPT2_PRED_PATH = PROJECT_ROOT / "experiments" / "gpt2_val_full_predictions.json"
STRUCT_COMPARISON_PATH = PROJECT_ROOT / "experiments" / "results_structure_comparison.json"

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TEMPERATURE = 0.07
SEED = 42
NUM_ENDINGS = 4
EVAL_SIZE = 500
LOG_EVERY = 400
ORIGINAL_NO_NAV_10042 = 0.31308504282015537
GPT2_10042 = 0.28579964150567616


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP-style end-to-end task training of the VAE head."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    return parser.parse_args(argv)


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
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
    return vae, checkpoint


def save_checkpoint(
    vae: VAE,
    path: Path,
    extras: Dict[str, Any],
    temperature: float,
    training: Dict[str, Any],
) -> float:
    """Same layout as ``vae.pt`` so existing loaders keep working."""
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
            "kl_weight": 0.001,
            "logvar_bias": -6.0,
        },
        "model_state_dict": model_state,
        "encoder_frozen": bool(vae.encoder.is_frozen),
        "training_objective": "clip_style_task_ce",
        "temperature": temperature,
        "head_finetuned": True,
        "training": training,
        **extras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path.stat().st_size / (1024 * 1024)


def scores_from_mu(ctx_mu: torch.Tensor, end_mu: torch.Tensor) -> torch.Tensor:
    """``cos(ending, ctx)`` from L2-normalised head means -> ``(N, 4)``."""
    ctx_unit = F.normalize(ctx_mu, dim=-1)
    end_unit = F.normalize(end_mu, dim=-1)
    return torch.einsum("nd,nkd->nk", ctx_unit, end_unit)


@torch.no_grad()
def head_mu(
    vae: VAE, base_ctx: torch.Tensor, base_end: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``mu`` for ctx ``(N,384)`` and endings ``(N,4,384)`` -> ``(N,256)``/``(N,4,256)``."""
    vae.head.eval()
    count = base_ctx.shape[0]
    ctx_mu, _ = vae.head(base_ctx)
    end_mu, _ = vae.head(base_end.reshape(count * NUM_ENDINGS, BASE_EMBED_DIM))
    return ctx_mu, end_mu.reshape(count, NUM_ENDINGS, LATENT_DIM)


@torch.no_grad()
def evaluate(
    vae: VAE,
    base_ctx: torch.Tensor,
    base_end: torch.Tensor,
    labels: torch.Tensor,
    num_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Rank-by-cosine accuracy + geometry (``rank_1`` IS the no_navigation accuracy)."""
    if num_examples is not None:
        base_ctx = base_ctx[:num_examples]
        base_end = base_end[:num_examples]
        labels = labels[:num_examples]
    ctx_mu, end_mu = head_mu(vae, base_ctx, base_end)
    scores = scores_from_mu(ctx_mu, end_mu).numpy()
    gold = labels.numpy()
    total = scores.shape[0]
    rows = np.arange(total)
    cos_correct = scores[rows, gold]
    wrong_mask = np.ones(scores.shape, dtype=bool)
    wrong_mask[rows, gold] = False
    cos_wrong = scores[wrong_mask].reshape(total, NUM_ENDINGS - 1)
    predictions = np.argmax(scores, axis=1)
    ranks = 1 + (scores > cos_correct[:, None]).sum(axis=1)
    return {
        "num_examples": total,
        "accuracy": float((predictions == gold).mean()),
        "correct": int((predictions == gold).sum()),
        "mean_cos_correct": float(cos_correct.mean()),
        "mean_cos_wrong": float(cos_wrong.mean()),
        "val_gap": float(cos_correct.mean() - cos_wrong.mean()),
        "gap_std": float((cos_correct - cos_wrong.mean(axis=1)).std()),
        "rank_counts": {
            int(rank): int((ranks == rank).sum()) for rank in range(1, NUM_ENDINGS + 1)
        },
        "predictions": [int(value) for value in predictions],
    }


def rename_pairs(stats: Dict[str, Any], first: str, second: str) -> Dict[str, Any]:
    """Re-key ``paired_comparison`` output so ``first``/``second`` are explicit."""
    return {
        "first": first,
        "second": second,
        f"{first}_correct_{second}_wrong": stats["full_correct_ablation_wrong"],
        f"{first}_wrong_{second}_correct": stats["full_wrong_ablation_correct"],
        "discordant_pairs": stats["discordant_pairs"],
        "mcnemar_p_value": stats["mcnemar_p_value"],
        "significant_at_0.05": stats["significant_at_0.05"],
    }


def train_head(
    vae: VAE,
    train_ctx: torch.Tensor,
    train_end: torch.Tensor,
    train_labels: torch.Tensor,
    val_ctx: torch.Tensor,
    val_end: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    eval_size: int = EVAL_SIZE,
    log_every: int = LOG_EVERY,
    seed: int = SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fine-tune ``vae.head`` on the CLIP-style task loss.

    Unfreezes ONLY the head, runs ``epochs`` passes over the cached base embeddings,
    and records the ``eval_size``-example val ``rank_1`` accuracy after every epoch.
    The returned ``best_state`` is the head weights of the highest-scoring epoch.

    Returns ``{history, best_state, best_epoch, best_accuracy, seconds}``.
    """
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    for parameter in vae.head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [p for p in vae.head.parameters() if p.requires_grad], lr=learning_rate
    )

    num_examples = train_ctx.shape[0]
    steps_per_epoch = (num_examples + batch_size - 1) // batch_size
    if verbose:
        print(
            f"\nTraining {epochs} epochs | {num_examples} examples | "
            f"{steps_per_epoch} steps/epoch (batch={batch_size}, T={temperature}, "
            f"lr={learning_rate})\n"
        )

    history: List[Dict[str, Any]] = []
    best_accuracy = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        vae.head.train()
        epoch_started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(num_examples, generator=generator)
        running_loss = 0.0
        batch_correct = 0
        seen = 0

        for start in range(0, num_examples, batch_size):
            index = permutation[start : start + batch_size]
            size = index.shape[0]
            gold = train_labels[index]
            ctx_mu, _ = vae.head(train_ctx[index])
            end_mu, _ = vae.head(
                train_end[index].reshape(size * NUM_ENDINGS, BASE_EMBED_DIM)
            )
            scores = scores_from_mu(ctx_mu, end_mu.reshape(size, NUM_ENDINGS, LATENT_DIM))
            loss = F.cross_entropy(scores / temperature, gold)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += float(loss.detach()) * size
            batch_correct += int((scores.detach().argmax(dim=1) == gold).sum())
            seen += size

            if verbose and log_every and global_step % log_every == 0:
                batch_acc = float((scores.detach().argmax(dim=1) == gold).float().mean())
                print(
                    f"  step {global_step:>5d} | loss {float(loss.detach()):.4f} "
                    f"| batch acc {100 * batch_acc:.1f}%",
                    flush=True,
                )

        eval_result = evaluate(vae, val_ctx, val_end, val_labels, eval_size)
        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_batch_accuracy": batch_correct / seen,
            "val_500_accuracy": eval_result["accuracy"],
            "val_500_correct": eval_result["correct"],
            "val_500_val_gap": eval_result["val_gap"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)

        if eval_result["accuracy"] > best_accuracy:
            best_accuracy = eval_result["accuracy"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in vae.state_dict().items()
                if not key.startswith("encoder.")
            }

        if verbose:
            print(
                f"epoch {epoch}/{epochs} | loss {record['train_loss']:.4f} "
                f"| train acc {100 * record['train_batch_accuracy']:.2f}% "
                f"| val{eval_size} rank_1 {100 * eval_result['accuracy']:.2f}% "
                f"({eval_result['correct']}/{eval_result['num_examples']}) "
                f"| val{eval_size} gap {eval_result['val_gap']:.4f} "
                f"| {record['seconds']:.0f}s",
                flush=True,
            )

    return {
        "history": history,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cpu")

    if not VAL_CACHE.exists():
        raise FileNotFoundError(
            f"{VAL_CACHE} missing - run scripts/precompute_val_base_embeddings.py first"
        )
    with np.load(TRAIN_CACHE) as data:
        train_ctx = torch.from_numpy(data["ctx"])
        train_end = torch.from_numpy(data["ending"])
        train_labels = torch.from_numpy(data["labels"]).long()
    with np.load(VAL_CACHE) as data:
        val_ctx = torch.from_numpy(data["ctx"])
        val_end = torch.from_numpy(data["endings"])
        val_labels = torch.from_numpy(data["labels"]).long()

    with (PROJECT_ROOT / "data" / "val_full.json").open(encoding="utf-8") as handle:
        val_data = json.load(handle)
    if [int(e["label"]) for e in val_data] != val_labels.tolist():
        raise RuntimeError("val base-embedding cache labels disagree with data/val_full.json")

    print(
        f"Train cache: ctx {tuple(train_ctx.shape)} | endings {tuple(train_end.shape)} "
        f"| labels {tuple(train_labels.shape)}"
    )
    print(
        f"Val cache  : ctx {tuple(val_ctx.shape)} | endings {tuple(val_end.shape)} "
        f"| labels {tuple(val_labels.shape)}"
    )

    vae, source = load_vae(LOAD_CHECKPOINT, device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    for parameter in vae.head.parameters():
        parameter.requires_grad_(True)
    trainable = [p for p in vae.head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"\nStart from   : {LOAD_CHECKPOINT.name} (epochs={source.get('epochs')})")
    print(f"Frozen       : base encoder ({BASE_MODEL_NAME}) + decoder (ignored)")
    print(f"Fine-tuned   : VAE head fc/mu_layer/logvar_layer = {n_trainable:,} params")
    del trainable
    print(
        f"Objective    : CrossEntropy(cos(ending, ctx) / {args.temperature}, label) "
        f"| epochs={args.epochs} batch={args.batch_size} lr={args.lr}"
    )
    if source.get("structure_margin") is not None:
        print("NOTE: starting checkpoint was NOT a margin-trained model")

    # ------------------------------------------------- pre-training self-check
    pre_eval = evaluate(vae, val_ctx, val_end, val_labels, args.eval_size)
    pre_full = evaluate(vae, val_ctx, val_end, val_labels)
    print("\n" + "=" * 84)
    print("SELF-CHECK: untouched vae.pt head scored through this harness")
    print("=" * 84)
    print(
        f"  first {args.eval_size} val : {100 * pre_eval['accuracy']:.1f}% "
        f"({pre_eval['correct']}/{pre_eval['num_examples']})"
    )
    print(
        f"  all {pre_full['num_examples']} val : {100 * pre_full['accuracy']:.4f}% "
        f"({pre_full['correct']}/{pre_full['num_examples']}) | published cached "
        f"{100 * ORIGINAL_NO_NAV_10042:.4f}% | delta "
        f"{100 * (pre_full['accuracy'] - ORIGINAL_NO_NAV_10042):+.4f} points"
    )
    print(
        f"  reproduces published baseline: "
        f"{abs(pre_full['accuracy'] - ORIGINAL_NO_NAV_10042) < 1e-9}"
    )

    # ------------------------------------------------------------- training
    training = train_head(
        vae,
        train_ctx,
        train_end,
        train_labels,
        val_ctx,
        val_end,
        val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        eval_size=args.eval_size,
    )
    history = training["history"]
    best_state = training["best_state"]
    best_epoch = training["best_epoch"]
    best_accuracy = training["best_accuracy"]
    print(f"\nTraining wall time: {training['seconds']:.1f}s")

    # ------------------------------------------------------------- saving
    final_state = {
        key: value.detach().cpu()
        for key, value in vae.state_dict().items()
        if not key.startswith("encoder.")
    }
    training_meta: Dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_500_accuracy": best_accuracy,
        "pretrain_val_500_accuracy": pre_eval["accuracy"],
        "pretrain_val_10042_accuracy": pre_full["accuracy"],
    }
    if best_state is not None:
        vae.load_state_dict(best_state, strict=False)
    best_mb = save_checkpoint(
        vae, CHECKPOINT_PATH, {"saved": "best"}, args.temperature, training_meta
    )
    vae.load_state_dict(final_state, strict=False)
    final_mb = save_checkpoint(
        vae, FINAL_CHECKPOINT_PATH, {"saved": "final"}, args.temperature, training_meta
    )

    # --------------------------------------------- full 10,042 evaluation
    if best_state is not None:
        vae.load_state_dict(best_state, strict=False)
    clip_eval = evaluate(vae, val_ctx, val_end, val_labels)

    with GMN_PRED_PATH.open(encoding="utf-8") as handle:
        gmn = json.load(handle)
    with GPT2_PRED_PATH.open(encoding="utf-8") as handle:
        gpt2 = json.load(handle)
    original_pred = [int(value) for value in gmn["no_navigation"]["predictions"]]
    gpt2_pred = [int(value) for value in gpt2["predictions"]]
    original_acc = float(gmn["no_navigation"]["accuracy"])
    v3_acc = None
    v3_pred: List[int] = []
    if STRUCT_COMPARISON_PATH.exists():
        with STRUCT_COMPARISON_PATH.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        v3_entry = previous["models"].get("Structured v3 (margin=0.3)")
        if v3_entry is not None:
            v3_acc = float(v3_entry["accuracy"])
            v3_pred = [int(value) for value in v3_entry["predictions"]]

    print("\n" + "=" * 84)
    print(f"FULL 10,042-EXAMPLE EVALUATION - CLIP-style (best epoch {best_epoch})")
    print("=" * 84)
    print(
        f"  accuracy   : {100 * clip_eval['accuracy']:.4f}% "
        f"({clip_eval['correct']}/{clip_eval['num_examples']})"
    )
    print(
        f"  val gap    : {clip_eval['val_gap']:.6f} "
        f"(cos+ {clip_eval['mean_cos_correct']:.4f} | cos- {clip_eval['mean_cos_wrong']:.4f})"
    )
    print(f"  val gap std: {clip_eval['gap_std']:.4f}")
    print(
        f"  rank dist  : "
        + " | ".join(
            f"rank{r} {100 * c / clip_eval['num_examples']:.1f}%"
            for r, c in sorted(clip_eval["rank_counts"].items())
        )
    )

    label = "CLIP-style no_nav"
    print("\n" + "=" * 84)
    print("FULL VALIDATION no_navigation ACCURACY (all 10,042 val examples)")
    print("=" * 84)
    print(f"  {'Condition':<26} | {'Accuracy':>9} | {'vs original':>12} | {'p-value':>10}")
    print("  " + "-" * 68)
    print(
        f"  {'Original VAE no_nav':<26} | {100 * original_acc:>8.4f}% | {'baseline':>12} "
        f"| {'-':>10}"
    )
    if v3_acc is not None:
        print(
            f"  {'Margin v3 no_nav':<26} | {100 * v3_acc:>8.4f}% "
            f"| {100 * (v3_acc - original_acc):>+11.4f}% | {'0.3083':>10}"
        )
    print(
        f"  {'CLIP-style no_nav':<26} | {100 * clip_eval['accuracy']:>8.4f}% "
        f"| {100 * (clip_eval['accuracy'] - original_acc):>+11.4f}% | "
        f"{'see below':>10}"
    )
    print(
        f"  {'GPT-2 small (cached)':<26} | {100 * GPT2_10042:>8.4f}% "
        f"| {100 * (GPT2_10042 - original_acc):>+11.4f}% | {'cached':>10}"
    )
    print("  " + "-" * 68)

    # ---------------------------------------------------------- McNemar ----
    mcnemar: Dict[str, Dict[str, Any]] = {
        "clip_vs_original": rename_pairs(
            paired_comparison(
                clip_eval["predictions"], original_pred, val_labels.tolist()
            ),
            "clip",
            "original",
        ),
        "clip_vs_gpt2_small": rename_pairs(
            paired_comparison(clip_eval["predictions"], gpt2_pred, val_labels.tolist()),
            "clip",
            "gpt2_small",
        ),
    }
    if v3_pred:
        mcnemar["clip_vs_margin_v3"] = rename_pairs(
            paired_comparison(clip_eval["predictions"], v3_pred, val_labels.tolist()),
            "clip",
            "margin_v3",
        )
    print("\n" + "=" * 84)
    print("McNEMAR EXACT TESTS")
    print("=" * 84)
    for name, stats in mcnemar.items():
        first_key = f"{stats['first']}_correct_{stats['second']}_wrong"
        second_key = f"{stats['first']}_wrong_{stats['second']}_correct"
        print(
            f"  {name:<22} | {first_key} {stats[first_key]:>5} "
            f"| {second_key} {stats[second_key]:>5} "
            f"| discordant {stats['discordant_pairs']:>5} "
            f"| p = {stats['mcnemar_p_value']:.4f} "
            f"| significant {stats['significant_at_0.05']}"
        )

    primary = mcnemar["clip_vs_original"]
    beats = bool(primary["significant_at_0.05"] and clip_eval["accuracy"] > original_acc)
    print("\n" + "=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print(f"  per-epoch val_{args.eval_size} rank_1:")
    for record in history:
        print(
            f"    epoch {record['epoch']}: {100 * record['val_500_accuracy']:.2f}% "
            f"({record['val_500_correct']}/{args.eval_size}) | loss {record['train_loss']:.4f}"
        )
    print(
        f"\n  full 10,042 accuracy   : {100 * clip_eval['accuracy']:.4f}%"
    )
    print(f"  vs original (31.31%)   : {100 * (clip_eval['accuracy'] - original_acc):+.4f} points")
    print(f"  McNemar p vs original  : {primary['mcnemar_p_value']:.4f}")
    print(f"  val gap / std          : {clip_eval['val_gap']:.6f} / {clip_eval['gap_std']:.4f}")
    print(f"  beats original significantly: {beats}")

    payload = {
        "task": "TASK 16 (modified) - CLIP-style end-to-end task training",
        "dataset": "hellaswag",
        "split": "validation (full)",
        "val_examples": clip_eval["num_examples"],
        "method": {
            "objective": "CrossEntropy(cos(ending, ctx) / temperature, label)",
            "temperature": args.temperature,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "base_encoder": "frozen all-MiniLM-L6-v2",
            "fine_tuned": "VAE head (fc, mu_layer, logvar_layer)",
            "decoder": "ignored",
            "start_checkpoint": str(LOAD_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "note": (
                "logvar_layer is unfrozen but receives no gradient: this loss has no "
                "KL term and only mu is used."
            ),
        },
        "self_check": {
            "pretrain_val_10042_accuracy_this_harness": pre_full["accuracy"],
            "published_cached_original_10042": ORIGINAL_NO_NAV_10042,
            "reproduces_published_baseline": bool(
                abs(pre_full["accuracy"] - ORIGINAL_NO_NAV_10042) < 1e-9
            ),
            "pretrain_val_500_accuracy": pre_eval["accuracy"],
        },
        "training_history": history,
        "results": {
            label: {
                "accuracy": clip_eval["accuracy"],
                "correct": clip_eval["correct"],
                "val_gap": clip_eval["val_gap"],
                "gap_std": clip_eval["gap_std"],
                "mean_cos_correct": clip_eval["mean_cos_correct"],
                "mean_cos_wrong": clip_eval["mean_cos_wrong"],
                "rank_counts": clip_eval["rank_counts"],
                "predictions": clip_eval["predictions"],
            },
            "original_no_nav_cached": original_acc,
            "margin_v3_no_nav": v3_acc,
            "gpt2_small_cached": GPT2_10042,
        },
        "comparison": {
            "clip_accuracy": clip_eval["accuracy"],
            "delta_vs_original": clip_eval["accuracy"] - original_acc,
            "delta_vs_gpt2": clip_eval["accuracy"] - GPT2_10042,
            "delta_vs_margin_v3": (clip_eval["accuracy"] - v3_acc) if v3_acc else None,
            "mcnemar_p_value_vs_original": primary["mcnemar_p_value"],
            "significant_vs_original": bool(primary["significant_at_0.05"]),
            "beats_original_significantly": beats,
            "val_gap": clip_eval["val_gap"],
            "val_gap_std": clip_eval["gap_std"],
        },
        "mcnemar": mcnemar,
        "artifacts": {
            "best": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "best_epoch": best_epoch,
            "best_size_mb": best_mb,
            "final": str(FINAL_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "final_size_mb": final_mb,
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {RESULTS_PATH} ({RESULTS_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

