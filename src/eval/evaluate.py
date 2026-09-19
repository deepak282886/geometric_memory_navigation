"""Evaluation: geometric ending scoring (GMN), random and GPT-2 baselines.

``score_example`` is the GMN inference procedure. It needs **no trained head**:
the four candidate endings are shown to the model and each is scored by its
cosine similarity to the *navigated* latent position.

    ctx --(frozen VAE)--> unit query vector
        --(memory kNN + Jev sampler + fusion)--> navigation vector
    navigated = unit_normalise(query_vector + navigation_vector)
    score_k   = cos(unit_encode(ending_k), navigated)
    prediction = argmax_k score_k

This makes the architecture more faithful to the navigation metaphor: the
context is a position, the memories give it a direction, and each candidate
continuation is judged by how well it lies along that direction.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.fusion.fuse import fuse_directions  # noqa: E402
from src.memory.store import MemoryStore, normalize_rows  # noqa: E402
from src.sampler.jev import parallel_sample  # noqa: E402

DEFAULT_K = 10
DEFAULT_FUSE_TOP_K = 3
INFERENCE_OUTCOME_WEIGHT = 1.0
NUM_ENDINGS = 4
GPT2_MODEL_NAME = "gpt2"
SEED = 42

DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "experiments" / "gmn_val_predictions.json"


def score_example(
    ctx: str,
    endings: Sequence[str],
    store: MemoryStore,
    vae: VAE,
    k: int = DEFAULT_K,
    fuse_top_k: int = DEFAULT_FUSE_TOP_K,
    quiet: bool = True,
    target: Optional[str] = None,
    lambda_nav: float = 1.0,
) -> np.ndarray:
    """Geometric ending scoring for one example -> ``(num_endings,)`` scores.

    1. encode the context and L2-normalise it (``mu`` has norm ~7.2, not 1)
    2. retrieve ``k`` memories and fuse their directions (all weights 1.0 at
       inference, because the true label is unknown). Fusion projects each
       direction onto the context's tangent space (``use_tangent=True``): the raw
       chords point away from the context on the unit sphere and would otherwise
       destroy the scoring signal.
    3. score each ending ADDITIVELY::

           scores = cos(ending, query_vector) + lambda_nav * cos(ending, navigation_vector)

    ``target`` selects what navigation points at and defaults to the store's own
    :attr:`MemoryStore.default_target`: ``"correct_ending"`` for a store built
    with correct-ending vectors (the corrected model), ``"context"`` otherwise.

    The additive form replaces an earlier ``cos(ending, L2(ctx + nav))`` blend.
    Because the tangent ``nav_vector`` is orthogonal to ``query_vector``, that
    blend was exactly ``(cos(ending, ctx) + cos(ending, nav)) / sqrt(2)``: correct
    up to a positive constant for *ranking*, but it dilutes the context term and
    costs accuracy. Adding the two terms keeps the full context signal.

    ``lambda_nav=0.0`` therefore reduces to scoring on the context alone.

    ``argmax`` of the returned scores is the predicted label.
    """
    effective_target = target if target is not None else store.default_target
    with torch.no_grad():
        query_mu = vae.encode_texts([ctx])  # (1, 256)
    query_vector = normalize_rows(query_mu.cpu().numpy())[0].astype(np.float64)

    memories = store.query(query_vector, k=k)
    outcome_weights = [INFERENCE_OUTCOME_WEIGHT] * len(memories)

    if quiet:
        # parallel_sample announces itself once per call; 500 calls would be noise
        with contextlib.redirect_stdout(io.StringIO()):
            weighted_directions = parallel_sample(
                query_vector, memories, outcome_weights, target=effective_target
            )
    else:
        weighted_directions = parallel_sample(
            query_vector, memories, outcome_weights, target=effective_target
        )

    navigation_vector = fuse_directions(
        weighted_directions,
        top_k=fuse_top_k,
        use_tangent=True,
        query_vector=query_vector,
    )

    with torch.no_grad():
        ending_mu = vae.encode_texts(list(endings))  # (num_endings, 256)
    ending_vectors = normalize_rows(ending_mu.cpu().numpy()).astype(np.float64)

    # additive, no position blending: the query position is never moved
    context_scores = ending_vectors @ query_vector
    navigation_scores = ending_vectors @ navigation_vector
    return context_scores + lambda_nav * navigation_scores


def score_val_set(
    val_data: Sequence[Dict[str, Any]],
    vae: VAE,
    store: MemoryStore,
    k: int = DEFAULT_K,
    fuse_top_k: int = DEFAULT_FUSE_TOP_K,
) -> Dict[str, Any]:
    """Run geometric scoring over a whole split -> accuracy + per-example detail."""
    predictions: List[int] = []
    labels: List[int] = []
    scores: List[List[float]] = []
    examples: List[Dict[str, Any]] = []

    started = time.perf_counter()
    for index, example in enumerate(val_data):
        example_scores = score_example(
            example["ctx"], example["endings"], store, vae, k=k, fuse_top_k=fuse_top_k
        )
        prediction = int(np.argmax(example_scores))
        label = int(example["label"])
        predictions.append(prediction)
        labels.append(label)
        scores.append([float(value) for value in example_scores])
        examples.append(
            {
                "idx": index,
                "activity_label": example["activity_label"],
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
                "scores": [float(value) for value in example_scores],
            }
        )
    seconds = time.perf_counter() - started

    correct = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
    total = len(val_data)
    return {
        "accuracy": correct / max(total, 1),
        "num_examples": total,
        "correct": correct,
        "seconds": seconds,
        "ms_per_example": 1000.0 * seconds / max(total, 1),
        "predictions": predictions,
        "labels": labels,
        "scores": scores,
        "examples": examples,
        "retrieval_k": k,
        "fuse_top_k": fuse_top_k,
    }


def evaluate_gmn(
    val_data: Sequence[Dict[str, Any]],
    vae: VAE,
    store: MemoryStore,
    predictions_path: Path | None = DEFAULT_PREDICTIONS_PATH,
    k: int = DEFAULT_K,
    fuse_top_k: int = DEFAULT_FUSE_TOP_K,
) -> float:
    """GMN accuracy on ``val_data``; also writes per-example predictions.

    Returns the accuracy (float). Pass ``predictions_path=None`` to skip writing.
    """
    result = score_val_set(val_data, vae, store, k=k, fuse_top_k=fuse_top_k)

    if predictions_path is not None:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"GMN predictions saved to {path}")

    return float(result["accuracy"])


def evaluate_random(val_data: Sequence[Dict[str, Any]], seed: int = SEED) -> float:
    """Uniform random baseline over 4 options (expect ~25%)."""
    labels = np.asarray([int(example["label"]) for example in val_data])
    rng = np.random.default_rng(seed)
    predictions = rng.choice(NUM_ENDINGS, size=len(labels))
    return float((predictions == labels).mean())


@torch.no_grad()
def ending_log_prob_stats(
    model: Any,
    tokenizer: Any,
    ctx: str,
    endings: Sequence[str],
) -> Tuple[List[float], List[int]]:
    """Per-ending ``(summed log prob, ending token count)`` from one batched pass.

    The four ``ctx + ending`` sequences share their context prefix, so they are
    scored in one batch. The context is tokenised on its own and the continuation
    starts at ``len(ctx_ids)`` -- the standard HellaSwag LM baseline convention.

    Returning both statistics from the same forward pass lets the raw-sum and the
    length-normalised GPT-2 baselines be measured for the price of one.
    """
    ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
    start = len(ctx_ids)
    sequences = [
        tokenizer(f"{ctx} {ending}", add_special_tokens=False)["input_ids"]
        for ending in endings
    ]

    width = max(len(ids) for ids in sequences)
    pad_id = int(tokenizer.eos_token_id)  # GPT-2 has no pad token; masked anyway
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    sums: List[float] = []
    counts: List[int] = []
    for row, ids in enumerate(sequences):
        total = 0.0
        for position in range(start, len(ids)):
            total += float(log_probs[row, position - 1, ids[position]])
        sums.append(total)
        counts.append(max(len(ids) - start, 0))
    return sums, counts


def ending_log_probabilities(
    model: Any,
    tokenizer: Any,
    ctx: str,
    endings: Sequence[str],
    length_normalize: bool = False,
) -> List[float]:
    """Summed (or length-normalised) ``log P`` over the ending tokens only.

    ``length_normalize=False`` is the TASK 8 specification (raw sum). Raw sums
    systematically disadvantage longer endings, so the normalised variant is
    measured as well to keep the GPT-2 comparison honest.
    """
    sums, counts = ending_log_prob_stats(model, tokenizer, ctx, endings)
    if not length_normalize:
        return sums
    return [(total / count) if count else total for total, count in zip(sums, counts)]


def evaluate_gpt2(
    val_data: Sequence[Dict[str, Any]],
    model_name: str = GPT2_MODEL_NAME,
    progress_every: int = 100,
    length_normalize: bool = False,
) -> float:
    """GPT-2 small baseline: pick the ending with the highest summed log prob."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # heavy, lazy import

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    correct = 0
    started = time.perf_counter()
    for index, example in enumerate(val_data, start=1):
        scores = ending_log_probabilities(
            model, tokenizer, example["ctx"], example["endings"],
            length_normalize=length_normalize,
        )
        if int(np.argmax(scores)) == int(example["label"]):
            correct += 1
        if progress_every and index % progress_every == 0:
            print(
                f"    gpt2 {index}/{len(val_data)} correct={correct} "
                f"acc={correct / index:.4f} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    return correct / max(len(val_data), 1)


def evaluate_gpt2_variants(
    val_data: Sequence[Dict[str, Any]],
    model_name: str = GPT2_MODEL_NAME,
    progress_every: int = 100,
) -> Dict[str, float]:
    """One GPT-2 pass -> both aggregation variants.

    * ``sum``              : raw summed log prob (the TASK 8 specification)
    * ``length_normalized``: summed log prob / number of ending tokens

    Raw sums favour shorter endings, so reporting both keeps the comparison fair.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # heavy, lazy import

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    correct_sum = 0
    correct_normalized = 0
    length_bias_hits = 0
    started = time.perf_counter()

    for index, example in enumerate(val_data, start=1):
        sums, counts = ending_log_prob_stats(
            model, tokenizer, example["ctx"], example["endings"]
        )
        label = int(example["label"])
        normalized = [
            (total / count) if count else total for total, count in zip(sums, counts)
        ]
        if int(np.argmax(sums)) == label:
            correct_sum += 1
        if int(np.argmax(normalized)) == label:
            correct_normalized += 1
        # would raw-sum pick the LONGEST ending? (diagnostic for the length bias)
        if int(np.argmax(counts)) == int(np.argmax(sums)):
            length_bias_hits += 1
        if progress_every and index % progress_every == 0:
            print(
                f"    gpt2 {index}/{len(val_data)} | sum {correct_sum / index:.4f} "
                f"| normalised {correct_normalized / index:.4f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    total = max(len(val_data), 1)
    return {
        "sum": correct_sum / total,
        "length_normalized": correct_normalized / total,
        "longest_ending_picked_by_sum_rate": length_bias_hits / total,
        "seconds": time.perf_counter() - started,
    }
