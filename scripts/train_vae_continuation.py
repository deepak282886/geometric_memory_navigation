"""STEP 0: retrain the VAE encoder head with a CLIP-style next-sentence signal.

Why (the TASK 27 finding)
-------------------------
TASK 27 measured the latent memory with the ORIGINAL VAE (``checkpoints/vae.pt``,
trained on reconstruction + KL over HellaSwag text):

    sweep memory      0 : cos 0.5903   (memoryless reference)
    sweep memory    256 : cos 0.5837   (-0.0066, t -8.78)
    sweep memory  38330 : cos 0.5984   (+0.0081, t +10.23)

i.e. **relevant** memory helps and **irrelevant** memory hurts. The memory
mechanism was working; the geometry it was reading was the bottleneck: the
original encoder never learned that "S_n+1 follows S_n", so a neighbour's
direction only carried information when the neighbour happened to be topically
identical. This script fixes the geometry with the standard contrastive recipe.

Architecture / loss (the retrieval metric, made differentiable)
---------------------------------------------------------------
    base encoder : FROZEN all-MiniLM-L6-v2 (384-d)    [embeddings precomputed/cached]
    VAE head     : FINE-TUNED (fc, mu_layer, logvar_layer)
    VAE decoder  : frozen and unused

    anchor   = unit(mu_head(base(S_n)))        (256)
    positive = unit(mu_head(base(S_n+1)))      (256)

    scores = anchor @ positive^T               (B, B); [i, j] = cos(anchor_i, pos_j)
    loss   = CrossEntropy(scores / temperature, labels = arange(B)),  T = 0.07

In-batch negatives: every other pair's ``S_n+1`` in the batch (31 of them at
batch 32) is treated as a wrong continuation of pair ``i``, so minimising the
loss directly forces ``cos(S_n, S_n+1) > cos(S_n, S_m+1)`` - exactly the
geometry the latent predictor needs, and exactly the property Paper 1's scorer
reads off with ``cos(ending, ctx)``.

Notes
-----
* L2 normalisation makes the loss scale-invariant, so the head cannot cheat by
  rescaling ``mu``; it has to reorganise directions (the failure mode TASK 13/14
  margin ranking fell into, per ``scripts/train_vae_clip.py``).
* ``logvar_layer`` is unfrozen as specified but receives no gradient: this loss
  has no KL term and only ``mu`` is used, so it keeps its ``vae.pt`` values.
* The base embeddings of all 8,171 unique WikiText val sentences are computed
  once (frozen encoder) and cached, so an epoch is pure head forward/backward.
* Same dataset both arms read: ``environment/data/wikitext/val_pairs.json``.

Artifacts
---------
    checkpoints/wikitext_val_base_embeddings.npz   cached frozen-base embeddings
    checkpoints/vae_continuation.pt                best epoch by in-batch top-1
    checkpoints/vae_continuation_final.pt          epoch 5
    experiments/continuation_training.json         per-epoch metrics

``--max-pairs`` trains on the first N pairs only, which the geometry check uses
as a held-out control: pair indices >= N are then pairs the head never saw.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    HEAD_HIDDEN_DIM,
    LATENT_DIM,
    VAE,
)

LOAD_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vae.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae_continuation.pt"
FINAL_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae_continuation_final.pt"
PAIRS_PATH = PROJECT_ROOT / "environment" / "data" / "wikitext" / "val_pairs.json"
BASE_CACHE_PATH = PROJECT_ROOT / "checkpoints" / "wikitext_val_base_embeddings.npz"
HISTORY_PATH = PROJECT_ROOT / "experiments" / "continuation_training.json"

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # same fine-tuning rate as the TASK 16 CLIP-style head run
TEMPERATURE = 0.07
SEED = 42
EMBED_BATCH_SIZE = 128
LOG_EVERY = 100
#: negative batch floor: a batch of 1 has no in-batch negative to learn from
MIN_BATCH = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# CLI + checkpoint plumbing (same recipes as TASK 2/8/13/14/16)
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP-style next-sentence fine-tuning of the VAE head (STEP 0)."
    )
    parser.add_argument("--load", type=Path, default=LOAD_CHECKPOINT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--final-checkpoint", type=Path, default=FINAL_CHECKPOINT_PATH)
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH)
    parser.add_argument("--base-cache", type=Path, default=BASE_CACHE_PATH)
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--embed-batch-size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="train on the first N pairs only (held-out control arm); 0 = all",
    )
    parser.add_argument("--force-reencode", action="store_true")
    return parser.parse_args(argv)


def load_vae(checkpoint_path: Path, device: torch.device) -> Tuple[VAE, Dict[str, Any]]:
    """Rebuild a VAE from a checkpoint (identical recipe to the other VAEs)."""
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
    training: Dict[str, Any],
) -> float:
    """Same layout as ``vae.pt`` (so ``scripts/*`` loaders keep working).

    Only head/decoder keys are written, like ``scripts/train_vae_clip.py``: the
    base encoder is the published frozen ``all-MiniLM-L6-v2`` and is rebuilt from
    ``base_model_name`` on load, so storing 90 MB of unchanged weights again would
    be pointless.
    """
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
        "training_objective": "clip_style_next_sentence_infonce",
        "temperature": training.get("temperature"),
        "head_finetuned": True,
        "training": training,
        **extras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path.stat().st_size / (1024 * 1024)



# ---------------------------------------------------------------------------
# Data: cached frozen-base embeddings for the WikiText val pairs
# ---------------------------------------------------------------------------


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    """``[[S_n, S_n+1], ...]`` from ``val_pairs.json`` (utf-8)."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    pairs = [(str(first), str(second)) for first, second in raw]
    if not pairs:
        raise ValueError(f"{path} contains no sentence pairs")
    return pairs


def unique_sentences(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], Dict[str, int]]:
    """Sentence list + ``sentence -> row`` map, in first-seen order."""
    sentences: List[str] = []
    index: Dict[str, int] = {}
    for first, second in pairs:
        for sentence in (first, second):
            if sentence not in index:
                index[sentence] = len(sentences)
                sentences.append(sentence)
    return sentences, index


def load_or_build_base_embeddings(
    vae: VAE,
    sentences: Sequence[str],
    cache_path: Path,
    force_reencode: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """``(len(sentences), 384)`` frozen base embeddings, cached on disk.

    The cache stores the sentence list it was built for, so a cache from a
    different pair file is detected and rebuilt instead of being silently reused.
    This is the only place MiniLM runs; a training epoch is pure head
    forward/backward on these cached rows.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_reencode:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_sentences = [str(value) for value in cached["sentences"]]
            if cached_sentences == list(sentences):
                embeddings = cached["embeddings"].astype(np.float32)
                print(f"loaded {embeddings.shape} cached base embeddings from {cache_path}")
                return embeddings
        print(f"cache {cache_path} does not match the pair file - re-embedding")

    print(f"embedding {len(sentences)} unique sentences with the frozen base encoder ...")
    started = time.perf_counter()
    with torch.no_grad():
        embeddings = vae.embed(list(sentences), batch_size=batch_size).cpu().numpy()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, sentences=np.asarray(list(sentences)), embeddings=embeddings)
    print(
        f"  embedded in {time.perf_counter() - started:.1f}s -> {embeddings.shape}, "
        f"cached to {cache_path}"
    )
    return embeddings



# ---------------------------------------------------------------------------
# The CLIP-style next-sentence objective
# ---------------------------------------------------------------------------


def info_nce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-batch InfoNCE over ``(B, D)`` unit vectors -> ``(loss, scores)``.

    ``scores[i, j] = cos(anchor_i, positive_j)`` (a plain dot product, since both
    sides are L2-normalised). Row ``i`` must be highest on its own column ``i``,
    so ``CrossEntropy(scores / temperature, labels=arange(B))`` is the retrieval
    objective with every other pair in the batch as a negative - cross-entropy,
    not a margin: the head has to *rank* the true continuation highest.

    ``temperature`` (0.07) sharpens the softmax, so the loss keeps pushing even
    once the diagonal already wins; that is the setting that made TASK 16's
    CLIP-style head the strongest encoder in Paper 1.
    """
    if anchors.shape != positives.shape:
        raise ValueError(
            f"shape mismatch: anchors {tuple(anchors.shape)} vs "
            f"positives {tuple(positives.shape)}"
        )
    scores = anchors @ positives.t()
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    loss = F.cross_entropy(scores / float(temperature), labels)
    return loss, scores


@torch.no_grad()
def in_batch_metrics(
    vae: VAE,
    base_anchor: torch.Tensor,
    base_positive: torch.Tensor,
    temperature: float = TEMPERATURE,
    batch_size: int = BATCH_SIZE,
    permutation: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Epoch metrics measured on the same batches the head trains on (no gradient).

    Reported per epoch:

        loss       mean in-batch InfoNCE
        cos_pair   mean cos(anchor_i, positive_i)       - the diagonal
        cos_other  mean cos(anchor_i, positive_{j != i}) - the negatives
        gap        cos_pair - cos_other: the property the predictor reads off
        top1       share of anchors whose best positive IS their own true next
                   sentence (in-batch retrieval accuracy at batch 32)
    """
    vae.head.eval()
    total = int(base_anchor.shape[0])
    if permutation is None:
        permutation = torch.arange(total)
    losses: List[float] = []
    pair_similarities: List[float] = []
    other_similarities: List[float] = []
    correct = 0
    seen = 0

    for start in range(0, total, batch_size):
        index = permutation[start : start + batch_size]
        if index.numel() < MIN_BATCH:  # a single example has no negative
            continue
        anchor_mu, _ = vae.head(base_anchor[index])
        positive_mu, _ = vae.head(base_positive[index])
        anchors = F.normalize(anchor_mu, dim=-1)
        positives = F.normalize(positive_mu, dim=-1)

        loss, scores = info_nce_loss(anchors, positives, temperature)
        size = int(anchors.shape[0])
        diagonal = scores.diagonal()
        losses.append(float(loss))
        pair_similarities.extend(diagonal.tolist())
        if size > 1:
            other = (scores.sum() - diagonal.sum()) / (size * (size - 1))
            other_similarities.append(float(other))
        correct += int((scores.argmax(dim=1) == torch.arange(size)).sum())
        seen += size

    cos_pair = float(np.mean(pair_similarities)) if pair_similarities else 0.0
    cos_other = float(np.mean(other_similarities)) if other_similarities else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "cos_pair": cos_pair,
        "cos_other": cos_other,
        "gap": cos_pair - cos_other,
        "top1_in_batch": correct / seen if seen else 0.0,
        "num_examples": seen,
        "num_batches": len(losses),
    }



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_head(
    vae: VAE,
    base_anchor: torch.Tensor,
    base_positive: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    seed: int = SEED,
    log_every: int = LOG_EVERY,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fine-tune ``vae.head`` on the in-batch next-sentence InfoNCE loss.

    Unfreezes ONLY the head (``fc``, ``mu_layer``, ``logvar_layer``): the frozen
    base encoder and the decoder are never touched, so ``mu`` is the only thing
    that moves. Returns the per-epoch history plus the head state of the best
    epoch, chosen by in-batch top-1 retrieval (loss only as a tie-breaker) since
    top-1 is the property the predictor consumes.
    """
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    for parameter in vae.head.parameters():
        parameter.requires_grad_(True)

    trainable = [p for p in vae.head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    total = int(base_anchor.shape[0])
    steps_per_epoch = (total + batch_size - 1) // batch_size
    if verbose:
        print(
            f"\ntraining {epochs} epochs | {total} pairs | {steps_per_epoch} steps/epoch "
            f"| batch {batch_size} | T {temperature} | lr {learning_rate} "
            f"| head params {sum(p.numel() for p in trainable):,}"
        )

    # where we start: the untouched vae.pt head through the same harness, on the
    # SAME protocol as the epochs (a random batch permutation). Measuring it on
    # consecutive pairs instead would be unfair: the file is ordered by document,
    # so a consecutive batch is full of near-duplicate sentences and every model
    # scores far worse on it.
    initial = in_batch_metrics(
        vae,
        base_anchor,
        base_positive,
        temperature,
        batch_size,
        torch.randperm(total, generator=torch.Generator().manual_seed(seed)),
    )
    if verbose:
        print(
            f"  epoch 0 (vae.pt head) | loss {initial['loss']:.4f} "
            f"| cos_pair {initial['cos_pair']:.4f} | cos_other {initial['cos_other']:.4f} "
            f"| gap {initial['gap']:.4f} | top1 {100 * initial['top1_in_batch']:.1f}%"
        )

    history: List[Dict[str, Any]] = []
    best_key = (initial["top1_in_batch"], -initial["loss"])
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        vae.head.train()
        epoch_started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(total, generator=generator)
        running_loss = 0.0
        steps = 0

        for start in range(0, total, batch_size):
            index = permutation[start : start + batch_size]
            if index.numel() < MIN_BATCH:
                continue
            anchor_mu, _ = vae.head(base_anchor[index])
            positive_mu, _ = vae.head(base_positive[index])
            loss, _ = info_nce_loss(
                F.normalize(anchor_mu, dim=-1),
                F.normalize(positive_mu, dim=-1),
                temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            steps += 1
            if log_every and steps % log_every == 0:
                print(
                    f"    epoch {epoch} step {steps}/{steps_per_epoch} "
                    f"| loss {running_loss / steps:.4f}"
                )

        metrics = in_batch_metrics(
            vae, base_anchor, base_positive, temperature, batch_size, permutation
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / steps if steps else 0.0,
                "seconds": time.perf_counter() - epoch_started,
                **metrics,
            }
        )
        if verbose:
            print(
                f"  epoch {epoch}/{epochs} | train loss {history[-1]['train_loss']:.4f} "
                f"| cos_pair {metrics['cos_pair']:.4f} "
                f"| cos_other {metrics['cos_other']:.4f} | gap {metrics['gap']:.4f} "
                f"| top1 {100 * metrics['top1_in_batch']:.1f}% "
                f"| {history[-1]['seconds']:.1f}s"
            )

        key = (metrics["top1_in_batch"], -metrics["loss"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in vae.head.state_dict().items()
            }

    return {
        "history": history,
        "initial": initial,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_metrics": history[best_epoch - 1] if best_epoch else initial,
        "total_seconds": time.perf_counter() - started,
    }



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    set_seed(args.seed)

    print("=" * 84)
    print("STEP 0 - CLIP-style next-sentence fine-tuning of the VAE head")
    print("=" * 84)

    vae, loaded = load_vae(args.load, device)
    print(f"loaded {args.load}")
    print(
        f"  base encoder frozen: {vae.encoder.is_frozen} | "
        f"head params: {sum(p.numel() for p in vae.head.parameters()):,}"
    )

    pairs = load_pairs(args.pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
        print(f"training on the first {len(pairs)} pairs only (held-out control arm)")
    sentences, sentence_index = unique_sentences(pairs)
    embeddings = load_or_build_base_embeddings(
        vae,
        sentences,
        args.base_cache,
        force_reencode=args.force_reencode,
        batch_size=args.embed_batch_size,
    )
    anchor_rows = np.fromiter(
        (sentence_index[first] for first, _ in pairs), dtype=np.int64, count=len(pairs)
    )
    positive_rows = np.fromiter(
        (sentence_index[second] for _, second in pairs), dtype=np.int64, count=len(pairs)
    )
    base_anchor = torch.from_numpy(np.ascontiguousarray(embeddings[anchor_rows]))
    base_positive = torch.from_numpy(np.ascontiguousarray(embeddings[positive_rows]))
    print(
        f"pairs {len(pairs)} | unique sentences {len(sentences)} "
        f"| base anchor {tuple(base_anchor.shape)} "
        f"| base positive {tuple(base_positive.shape)}"
    )

    result = train_head(
        vae,
        base_anchor,
        base_positive,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        seed=args.seed,
    )

    training_meta = {
        "dataset": "wikitext-103 val sentence pairs",
        "pairs_file": str(args.pairs),
        "num_pairs": len(pairs),
        "max_pairs": args.max_pairs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "seed": args.seed,
        "base_encoder": "frozen all-MiniLM-L6-v2",
        "fine_tuned": "VAE head (fc, mu_layer, logvar_layer)",
        "decoder": "frozen and unused",
        "start_checkpoint": str(args.load),
        "objective": "CrossEntropy(anchors @ positives^T / T, arange(B))",
        "negatives": "in-batch (batch_size - 1 per anchor)",
        "note": (
            "logvar_layer is unfrozen as specified but receives no gradient: this "
            "loss has no KL term and only mu is used."
        ),
        "initial_metrics": result["initial"],
        "best_epoch": result["best_epoch"],
        "best_metrics": result["best_metrics"],
        "seconds": result["total_seconds"],
    }
    extras = {
        "task": "STEP 0 - continuation encoder (CLIP-style next-sentence signal)",
        "base_encoder_frozen": True,
        "head_finetuned": True,
    }
    final_size = save_checkpoint(
        vae, args.final_checkpoint, extras, {**training_meta, "epoch_used": args.epochs}
    )
    if result["best_state"] is not None:
        current = {k: v.detach().clone() for k, v in vae.head.state_dict().items()}
        vae.head.load_state_dict(result["best_state"])
        best_size = save_checkpoint(
            vae,
            args.checkpoint,
            extras,
            {**training_meta, "epoch_used": result["best_epoch"]},
        )
        vae.head.load_state_dict(current)  # leave the model on the final epoch
    else:  # no epoch beat the starting head - the starting head IS the best
        best_size = save_checkpoint(
            vae, args.checkpoint, extras, {**training_meta, "epoch_used": 0}
        )
    return _write_history(args, device, result, loaded, best_size, final_size)



def _write_history(
    args: argparse.Namespace,
    device: torch.device,
    result: Dict[str, Any],
    loaded: Dict[str, Any],
    best_size: float,
    final_size: float,
) -> Dict[str, Any]:
    """Write ``experiments/continuation_training.json`` and print the summary table."""
    payload = {
        "task": "STEP 0 - VAE continuation encoder (CLIP-style next-sentence signal)",
        "config": {
            "load": str(args.load),
            "pairs": str(args.pairs),
            "base_cache": str(args.base_cache),
            "checkpoint": str(args.checkpoint),
            "final_checkpoint": str(args.final_checkpoint),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_pairs": args.max_pairs,
            "device": str(device),
        },
        "method": {
            "anchors": "unit(mu_head(base(S_n)))",
            "positives": "unit(mu_head(base(S_n+1)))",
            "loss": "CrossEntropy(anchors @ positives^T / temperature, arange(B))",
            "negatives": "in-batch",
            "unfrozen": "VAE head only (fc, mu_layer, logvar_layer)",
            "frozen": "all-MiniLM-L6-v2 base encoder, decoder",
            "note": (
                "logvar_layer is unfrozen as specified but receives no gradient: "
                "this loss has no KL term and only mu is used."
            ),
        },
        "initial_metrics_vae_pt_head": result["initial"],
        "training_history": result["history"],
        "best": {
            "epoch": result["best_epoch"],
            "metrics": result["best_metrics"],
            "checkpoint": str(args.checkpoint),
            "size_mb": best_size,
        },
        "final": {
            "epoch": args.epochs,
            "metrics": result["history"][-1] if result["history"] else None,
            "checkpoint": str(args.final_checkpoint),
            "size_mb": final_size,
        },
        "improvement": {
            "loss": (
                result["initial"]["loss"] - result["best_metrics"]["loss"]
                if result["best_epoch"] else 0.0
            ),
            "gap": (
                result["best_metrics"]["gap"] - result["initial"]["gap"]
                if result["best_epoch"] else 0.0
            ),
            "top1_in_batch": (
                result["best_metrics"]["top1_in_batch"]
                - result["initial"]["top1_in_batch"]
                if result["best_epoch"] else 0.0
            ),
        },
        "loaded_checkpoint_metadata": {
            key: loaded.get(key)
            for key in ("epochs", "batch_size", "learning_rate", "final_loss")
        },
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved training history -> {args.history}")

    print("\n" + "=" * 84)
    print("STEP 0 SUMMARY - head fine-tuned on in-batch next-sentence InfoNCE")
    print("=" * 84)
    print("  epoch | train loss | cos_pair | cos_other |    gap | top1 (batch 32)")
    print("  " + "-" * 72)
    print(
        f"  {0:>5} | {'-':>10} | {result['initial']['cos_pair']:>8.4f} | "
        f"{result['initial']['cos_other']:>9.4f} | {result['initial']['gap']:>6.4f} | "
        f"{100 * result['initial']['top1_in_batch']:>12.1f}%   (vae.pt head)"
    )
    for entry in result["history"]:
        print(
            f"  {entry['epoch']:>5} | {entry['train_loss']:>10.4f} | "
            f"{entry['cos_pair']:>8.4f} | {entry['cos_other']:>9.4f} | "
            f"{entry['gap']:>6.4f} | {100 * entry['top1_in_batch']:>12.1f}%"
        )
    print("  " + "-" * 72)
    print(f"  best epoch {result['best_epoch']:>2} -> {args.checkpoint} ({best_size:.1f} MB)")
    print(f"  final epoch {args.epochs:>2} -> {args.final_checkpoint} ({final_size:.1f} MB)")
    print(
        f"  gap      {result['initial']['gap']:.4f} -> "
        f"{result['best_metrics']['gap']:.4f}  | "
        f"loss {result['initial']['loss']:.4f} -> {result['best_metrics']['loss']:.4f}"
    )
    print(
        f"  top1     {100 * result['initial']['top1_in_batch']:.1f}% -> "
        f"{100 * result['best_metrics']['top1_in_batch']:.1f}%"
    )
    print("=" * 84)
    return payload


if __name__ == "__main__":
    main()

