"""TASK 29: the prediction loop - does memory get more useful as it grows?

The question
------------
TASK 27 ran the loop with a *static* experience memory and found the geometry, not
the memory, was the bottleneck. TASK 28 built the dynamic/region memory. This script
finally asks the V4 claim directly, with the predictor's weights FROZEN:

    does the memory's contribution to held-out prediction quality GROW as the
    memory grows?

Design (nothing here trains anything)
-------------------------------------
    encoder    checkpoints/vae_continuation.pt          frozen
    predictor  checkpoints/latent_predictor_continuation.pt frozen, same weights in
                                                         both arms
    memory     RegionStore: 256 regions, probe_regions=4, max_size=50000,
               score_mode="spec_prior", auto_fit_after=2048

    stream     10,000 pairs from environment/data/wikitext/train_pairs.json
               (streamed line by line - the file is 966 MB and is never loaded whole)
    warm-up    the first 2,048 adds: regions do not exist yet, so C = zeros(256),
               the predictor still runs, the loss is still recorded, the memory still
               grows, and every step is labelled "warmup"
    after fit  C = fuse_directions(parallel_sample(top-10 memories), top_k=3,
                                   use_tangent=True, query_vector=P_current)

    TWO ARMS, same pairs, same order, same frozen predictor:
      ARM A  context from the RegionStore
      ARM B  C = zeros(256) always
    so the delta is the memory's contribution with learning held fixed.

The honest metric (decision 5)
------------------------------
A FIXED held-out eval set: the last 500 pairs of ``val_pairs.json`` (indices
7166-7665), which are NEVER added to the memory. Every 500 steps both arms are scored
on those 500 pairs: query the memory with P_current, fuse, predict P_next, and take
cos(P_predicted, P_actual). The mean over the 500 pairs is what the table reports -
the training-step quality is also logged, but it is not the headline.

Two things to read carefully in the output:

* the eval pairs are held out from the MEMORY but not from the plane the predictor
  was trained on - TASK 27 trained it on all 7,666 val pairs, including these 500.
  Both arms share the predictor, so the delta is still a clean memory effect, but
  "held-out" here means "never memorised", not "never seen by the model".
* during warm-up the two arms are identical BY CONSTRUCTION (both get C = zeros), so
  the script asserts delta == 0 there as a harness self-check, and reports the
  memory-active trend (steps 2,048+) separately from the literal 500-to-10,000 span.

Artifacts
---------
    experiments/prediction_loop_results.json      every number printed below
    checkpoints/prediction_loop_train_vecs.npz    cached latents of the streamed pairs
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.fusion.fuse import fuse_directions  # noqa: E402
from src.memory.region_store import RegionStore  # noqa: E402
from src.memory.store import normalize_rows  # noqa: E402
from src.predictor.latent_predictor import (  # noqa: E402
    LATENT_DIM,
    LatentPredictor,
    load_latent_predictor,
)
from src.sampler.jev import (  # noqa: E402
    TARGET_CORRECT_ENDING,
    parallel_sample,
    retrieval_confidence,
)

DEFAULT_VAE = PROJECT_ROOT / "checkpoints" / "vae_continuation.pt"
DEFAULT_PREDICTOR = PROJECT_ROOT / "checkpoints" / "latent_predictor_continuation.pt"
TRAIN_PAIRS_PATH = PROJECT_ROOT / "environment" / "data" / "wikitext" / "train_pairs.json"
VAL_PAIRS_PATH = PROJECT_ROOT / "environment" / "data" / "wikitext" / "val_pairs.json"
VAL_VEC_CACHE = PROJECT_ROOT / "checkpoints" / "wikitext_val_predictor_vecs_continuation.npz"
TRAIN_VEC_CACHE = PROJECT_ROOT / "checkpoints" / "prediction_loop_train_vecs.npz"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "prediction_loop_results.json"

# --- decision 1: RegionStore configuration -------------------------------------
NUM_REGIONS = 256
PROBE_REGIONS = 4
MAX_SIZE = 50000
SCORE_MODE = "spec_prior"
AUTO_FIT_AFTER = 2048

# --- loop configuration --------------------------------------------------------
STEPS = 10000
EVAL_EVERY = 500
EVAL_SIZE = 500
EVAL_START = 7166  # pairs 7166-7665 of val_pairs.json, never added to memory
KNN_K = 10
FUSE_TOP_K = 3
#: TASK 30: trust the memory context only when the top-3 neighbour cosine clears this
GATE_THRESHOLD = 0.5
#: how many neighbours the gate averages over (matches the fused top-3)
GATE_TOP_N = 3
MIN_MEMORY_ENTRIES = 5
OUTCOME_FLOOR = 0.1
#: eval pairs never enter the memory; this keeps their source ids out of the
#: training range so exclude_source can never collide with a training step
EVAL_SOURCE_BASE = 1000000
EMBED_BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 256
SEED = 42


# ---------------------------------------------------------------------------
# CLI + data (streamed, never loaded whole)
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TASK 29: prediction loop over 10,000 WikiText pairs."
    )
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--train-pairs", type=Path, default=TRAIN_PAIRS_PATH)
    parser.add_argument("--val-pairs", type=Path, default=VAL_PAIRS_PATH)
    parser.add_argument("--train-cache", type=Path, default=TRAIN_VEC_CACHE)
    parser.add_argument("--val-cache", type=Path, default=VAL_VEC_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--eval-start", type=int, default=EVAL_START)
    parser.add_argument("--num-regions", type=int, default=NUM_REGIONS)
    parser.add_argument("--probe-regions", type=int, default=PROBE_REGIONS)
    parser.add_argument("--max-size", type=int, default=MAX_SIZE)
    parser.add_argument("--auto-fit-after", type=int, default=AUTO_FIT_AFTER)
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=GATE_THRESHOLD,
        help=(
            "TASK 30: use the memory context only when the mean cosine of the top-"
            "GATE_TOP_N neighbours is >= this; otherwise C = zeros"
        ),
    )
    parser.add_argument(
        "--gate-top-n",
        type=int,
        default=GATE_TOP_N,
        help="how many neighbours the confidence gate averages over (default 3 = fused top-k)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="disable the gate: always use the memory context (TASK 29's ARM B behaviour)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-reencode", action="store_true")
    return parser.parse_args(argv)


def stream_pairs(path: Path, limit: int, skip: int = 0) -> List[Tuple[str, str]]:
    """Stream ``limit`` sentence pairs out of a one-JSON-pair-per-line file.

    ``train_pairs.json`` is 966 MB, so it is read line by line and never parsed as one
    document. Format: a JSON array whose opening bracket is glued to the first pair
    (``[[...],``), one pair per line, and whose closing bracket is glued to the last
    line. A line that fails to parse as a pair is retried with the trailing array
    bracket removed, which is what the final line needs.
    """
    pairs: List[Tuple[str, str]] = []
    seen = 0
    first_line = True
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            if first_line:
                first_line = False
                if text.startswith("["):
                    text = text[1:]  # the array bracket, first line only
            if text.endswith(","):
                text = text[:-1]
            text = text.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                if not text.endswith("]"):
                    raise
                item = json.loads(text[:-1])  # final line: pair + array bracket
            if not (isinstance(item, list) and len(item) == 2):
                continue
            if not all(isinstance(value, str) for value in item):
                continue
            seen += 1
            if seen <= skip:
                continue
            pairs.append((item[0], item[1]))
            if len(pairs) >= limit:
                break
    if len(pairs) < limit:
        print(f"  note: only {len(pairs)} pairs available (asked for {limit})")
    return pairs


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    """Fully load a small pairs file (val_pairs.json is 2 MB)."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [(str(first), str(second)) for first, second in raw]


def unique_sentences(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], Dict[str, int]]:
    sentences: List[str] = []
    index: Dict[str, int] = {}
    for first, second in pairs:
        for sentence in (first, second):
            if sentence not in index:
                index[sentence] = len(sentences)
                sentences.append(sentence)
    return sentences, index



def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
    """Rebuild the frozen VAE (same recipe as the other scripts)."""
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
    return vae


def encode_sentences_cached(
    vae: VAE,
    sentences: Sequence[str],
    cache_path: Path,
    force_reencode: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """``(len(sentences), 256)`` unit-norm latents, cached (keyed on the sentence list)."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_reencode:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_sentences = [str(value) for value in cached["sentences"]]
            if cached_sentences == list(sentences):
                vectors = cached["vectors"].astype(np.float32)
                print(f"  loaded {vectors.shape} cached latents from {cache_path.name}")
                return vectors
    print(f"  encoding {len(sentences)} unique sentences with the frozen VAE ...")
    started = time.perf_counter()
    with torch.no_grad():
        base = vae.encode_texts(list(sentences), batch_size=batch_size).cpu().numpy()
    vectors = normalize_rows(base)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, sentences=np.asarray(list(sentences)), vectors=vectors)
    print(f"  encoded in {time.perf_counter() - started:.1f}s -> {vectors.shape}")
    return vectors


@torch.no_grad()
def predict(predictor: LatentPredictor, current: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    """Batched unit predictions for ``(P_current, C)`` rows (no gradient)."""
    total = int(current.shape[0])
    out = np.zeros((total, LATENT_DIM), dtype=np.float32)
    for start in range(0, total, PREDICT_BATCH_SIZE):
        stop = min(start + PREDICT_BATCH_SIZE, total)
        prediction = predictor(
            torch.from_numpy(np.ascontiguousarray(current[start:stop], dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(contexts[start:stop], dtype=np.float32)),
        )
        out[start:stop] = prediction.cpu().numpy()
    return out


def cosine_to_actual(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-example ``cos(P_predicted, P_actual)`` (both unit rows)."""
    return np.einsum("nd,nd->n", predictions.astype(np.float64), actual.astype(np.float64))



def fuse_context(
    query_vector: np.ndarray,
    records: Sequence[Dict[str, Any]],
    probe_regions: int = PROBE_REGIONS,
    knn_k: int = KNN_K,
    fuse_top_k: int = FUSE_TOP_K,
) -> np.ndarray:
    """Context ``C`` from a RegionStore result: the TASK 27 fusion path.

    ``parallel_sample`` (weight, direction per memory) -> ``fuse_directions`` (weighted
    top-``fuse_top_k`` sum, projected onto the query's tangent space, L2-normalised).
    Fewer than ``MIN_MEMORY_ENTRIES`` usable neighbours gives ``zeros(256)``, matching
    TASK 27's cold-start rule. ``parallel_sample`` prints a line per call for
    interactive use, and this runs ~30k times, so its stdout goes to a throw-away
    buffer. ``probe_regions``/``knn_k`` are accepted so a caller can see the settings
    the loop used, and are documented rather than re-read from the store.
    """
    usable = [record for record in records if "target" in record]
    if len(usable) < MIN_MEMORY_ENTRIES:
        return np.zeros(LATENT_DIM, dtype=np.float64)
    memories = [
        {"vector": record["vector"], "correct_ending_vector": record["target"]}
        for record in usable
    ]
    outcome_weights = [float(record["outcome_weight"]) for record in usable]
    with contextlib.redirect_stdout(io.StringIO()):
        sampled = parallel_sample(
            query_vector,
            memories,
            outcome_weights=outcome_weights,
            target=TARGET_CORRECT_ENDING,
        )
    return fuse_directions(
        sampled,
        top_k=fuse_top_k,
        dim=LATENT_DIM,
        use_tangent=True,
        query_vector=query_vector,
    )


def contexts_for_arm_a(
    memory: RegionStore,
    current: np.ndarray,
    sources: np.ndarray,
    probe_regions: int = PROBE_REGIONS,
    gate_threshold: Optional[float] = None,
    gate_top_n: int = GATE_TOP_N,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ARM A contexts: memory-derived, gated, or zeros while the regions are not fitted.

    ``gate_threshold is None`` reproduces TASK 29 (always use the memory). With a
    threshold, each query first asks :func:`retrieval_confidence` whether the retrieved
    neighbours are close enough to be trusted; if not, that query gets ``C = zeros``
    instead of a context built from memories in the wrong neighbourhood - the failure
    mode TASK 29 measured (-0.0049 out-of-corpus).

    Returns ``(contexts, stats)`` with ``stats`` =:

        queries        number of queries asked (0 during warm-up)
        fused          how many got a memory context
        gated_out      how many were rejected by the gate
        nonzero        how many ended up with a non-zero context
        mean_confidence  mean top-n neighbour cosine over the queries
        confidences    per-query values (used for the distribution in the sweep)
    """
    total = int(current.shape[0])
    contexts = np.zeros((total, LATENT_DIM), dtype=np.float32)
    stats: Dict[str, Any] = {
        "queries": 0,
        "fused": 0,
        "gated_out": 0,
        "nonzero": 0,
        "mean_confidence": 0.0,
        "confidences": [],
    }
    if not memory.is_fitted:
        return contexts, stats  # decision 2: no regions yet, so C = zeros

    confidences: List[float] = []
    for index in range(total):
        records = memory.query(
            current[index],
            k=KNN_K,
            probe_regions=probe_regions,
            exclude_source=int(sources[index]),
        )
        stats["queries"] += 1
        trusted = True
        if gate_threshold is not None:
            trusted, confidence = retrieval_confidence(
                current[index], records, threshold=gate_threshold, top_n=gate_top_n
            )
            confidences.append(confidence)
            if not trusted:
                stats["gated_out"] += 1
                continue  # leave C at zeros for this query
        context = fuse_context(current[index], records, probe_regions=probe_regions)
        stats["fused"] += 1
        if float(np.abs(context).sum()) > 0.0:
            stats["nonzero"] += 1
        contexts[index] = context
    stats["confidences"] = confidences
    stats["mean_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    return contexts, stats


def paired_stats(arm_a: np.ndarray, arm_b: np.ndarray) -> Dict[str, Any]:
    """Paired per-example comparison of the two arms on the same eval pairs."""
    delta = arm_a.astype(np.float64) - arm_b.astype(np.float64)
    total = int(delta.size)
    std = float(delta.std(ddof=1)) if total > 1 else 0.0
    standard_error = std / float(np.sqrt(total)) if total else 0.0
    return {
        "num_examples": total,
        "mean_cos_arm_a": float(arm_a.mean()),
        "mean_cos_arm_b": float(arm_b.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)) if total else 0.0,
        "delta_std": std,
        "paired_t_statistic": float(delta.mean() / standard_error) if standard_error > 0 else 0.0,
        "arm_a_wins": int((delta > 0).sum()),
        "arm_b_wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "share_arm_a_wins": float((delta > 0).mean()) if total else 0.0,
    }


def spearman_rho(first: Sequence[float], second: Sequence[float]) -> float:
    """Rank correlation (no scipy needed for the trend flag)."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.size < 2:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = float(np.sqrt((rank_a**2).sum() * (rank_b**2).sum()))
    return float((rank_a * rank_b).sum() / denominator) if denominator > 0 else 0.0



def evaluate_arms(
    predictor: LatentPredictor,
    memory: RegionStore,
    eval_current: np.ndarray,
    eval_actual: np.ndarray,
    eval_sources: np.ndarray,
    step: int,
    probe_regions: int = PROBE_REGIONS,
    gate_threshold: Optional[float] = GATE_THRESHOLD,
    gate_top_n: int = GATE_TOP_N,
) -> Dict[str, Any]:
    """Score all three arms on the fixed held-out eval set (TASK 30).

    ARM A  gated memory   (C only when the gate trusts the neighbours, else zeros)
    ARM B  always memory  (TASK 29's behaviour - the arm the gate is meant to fix)
    ARM C  never memory   (C = zeros always; the baseline)

    Same 500 pairs for every arm and every checkpoint, so the arms are compared
    pairwise on identical queries. ARM B costs nothing extra: it is the same memory
    queried without the gate, from the same records ARM A already fetched.

    The eval queries are real queries - they increment region visits and memory hit
    counts, which feeds the adaptive-detail budget. All three arms are treated the same
    way here (ARM C does not query at all), and the JSON says so rather than hiding it.
    """
    contexts_a, gate_stats = contexts_for_arm_a(
        memory, eval_current, eval_sources, probe_regions, gate_threshold, gate_top_n
    )
    cos_a = cosine_to_actual(predict(predictor, eval_current, contexts_a), eval_actual)

    contexts_b, _ = contexts_for_arm_a(
        memory, eval_current, eval_sources, probe_regions, None, gate_top_n
    )
    cos_b = cosine_to_actual(predict(predictor, eval_current, contexts_b), eval_actual)

    cos_c = cosine_to_actual(
        predict(predictor, eval_current, np.zeros_like(contexts_a)), eval_actual
    )

    return {
        "step": int(step),
        "warmup": not memory.is_fitted,
        "memory_size": int(memory.size),
        "n_regions": int(memory.num_regions) if memory.is_fitted else 0,
        "num_eval_pairs": int(eval_current.shape[0]),
        "gate_threshold": None if gate_threshold is None else float(gate_threshold),
        "gate_queries": int(gate_stats["queries"]),
        "gate_used": int(gate_stats["fused"]),
        "gate_rate": (
            float(gate_stats["fused"] / gate_stats["queries"])
            if gate_stats["queries"]
            else 0.0
        ),
        "mean_confidence": float(gate_stats["mean_confidence"]),
        # flat convenience fields (kept flat so the band/trend helpers stay readable)
        "mean_cos_arm_a": float(cos_a.mean()),
        "mean_cos_arm_b": float(cos_b.mean()),
        "mean_cos_arm_c": float(cos_c.mean()),
        "delta_a_vs_c": float(cos_a.mean() - cos_c.mean()),
        "delta_a_vs_b": float(cos_a.mean() - cos_b.mean()),
        "delta_b_vs_c": float(cos_b.mean() - cos_c.mean()),
        "arm_a_gated": paired_stats(cos_a, cos_c),
        "arm_a_vs_b": paired_stats(cos_a, cos_b),
        "arm_b_ungated": paired_stats(cos_b, cos_c),
        # per-pair arrays, so the sweep can run paired tests against any baseline
        "_cos_a": cos_a,
        "_cos_b": cos_b,
        "_cos_c": cos_c,
    }


def band_summary(
    checkpoints: Sequence[Dict[str, Any]], bands: Sequence[Tuple[str, int, int]]
) -> List[Dict[str, Any]]:
    """Aggregate checkpoint evaluations into the step bands the task asks for."""
    rows: List[Dict[str, Any]] = []
    for label, low, high in bands:
        selected = [row for row in checkpoints if low <= row["step"] <= high]
        if not selected:
            continue
        rows.append(
            {
                "band": label,
                "steps": [low, high],
                "checkpoints": [row["step"] for row in selected],
                "warmup_only": all(row["warmup"] for row in selected),
                "mean_cos_arm_a": float(np.mean([row["mean_cos_arm_a"] for row in selected])),
                "mean_cos_arm_b": float(np.mean([row["mean_cos_arm_b"] for row in selected])),
                "mean_cos_arm_c": float(np.mean([row["mean_cos_arm_c"] for row in selected])),
                "mean_delta_a_vs_c": float(
                    np.mean([row["delta_a_vs_c"] for row in selected])
                ),
                "mean_delta_a_vs_b": float(
                    np.mean([row["delta_a_vs_b"] for row in selected])
                ),
                "mean_gate_rate": float(np.mean([row["gate_rate"] for row in selected])),
                "mean_memory_size": float(np.mean([row["memory_size"] for row in selected])),
            }
        )
    return rows


def trend_analysis(checkpoints: Sequence[Dict[str, Any]], auto_fit_after: int) -> Dict[str, Any]:
    """Does the memory's contribution grow with the memory? (delta A vs C now)

    Read three ways, because they answer different questions:
      * the literal span the task asks for (step 500 -> step 10,000), which includes
        the warm-up checkpoints where delta is ZERO by construction;
      * the memory-active checkpoints only (step > auto_fit_after), which is the
        question that actually matters;
      * the last two bands, i.e. is a 5k-step memory better than a 2k-step one?
    """
    steps = [row["step"] for row in checkpoints]
    deltas = [row["delta_a_vs_c"] for row in checkpoints]
    active = [row for row in checkpoints if row["step"] > auto_fit_after]
    active_steps = [row["step"] for row in active]
    active_deltas = [row["delta_a_vs_c"] for row in active]
    first = checkpoints[0] if checkpoints else None
    last = checkpoints[-1] if checkpoints else None
    first_active = active[0] if active else None
    return {
        "delta_at_first_checkpoint": None if first is None else first["delta_a_vs_c"],
        "delta_at_first_checkpoint_step": None if first is None else first["step"],
        "delta_at_first_memory_active_checkpoint": (
            None if first_active is None else first_active["delta_a_vs_c"]
        ),
        "first_memory_active_step": None if first_active is None else first_active["step"],
        "delta_at_last_checkpoint": None if last is None else last["delta_a_vs_c"],
        "last_checkpoint_step": None if last is None else last["step"],
        "spearman_rho_delta_vs_step_all": spearman_rho(steps, deltas),
        "spearman_rho_delta_vs_step_memory_active": spearman_rho(active_steps, active_deltas),
        "memory_active_growth": (
            None
            if first_active is None or last is None
            else last["delta_a_vs_c"] - first_active["delta_a_vs_c"]
        ),
        "warmup_delta_is_zero": bool(
            all(abs(row["delta_a_vs_c"]) < 1e-12 for row in checkpoints if row["warmup"])
        ),
        "memory_active_checkpoints": len(active),
    }



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 96)
    print("TASK 29 - the prediction loop: two arms, one frozen predictor, 10,000 pairs")
    print("=" * 96)

    # ---------------------------------------------------------------- data
    print("\ndata")
    train_pairs = stream_pairs(args.train_pairs, args.steps)
    print(
        f"  streamed {len(train_pairs)} pairs from {args.train_pairs.name} "
        f"(line by line - the file is never parsed as one document)"
    )
    val_pairs = load_pairs(args.val_pairs)
    eval_pairs = val_pairs[args.eval_start : args.eval_start + args.eval_size]
    if not eval_pairs:
        raise ValueError(f"no eval pairs at [{args.eval_start}:{args.eval_start + args.eval_size}]")
    print(
        f"  eval set: val_pairs[{args.eval_start}:{args.eval_start + len(eval_pairs)}] "
        f"= {len(eval_pairs)} pairs, NEVER added to the memory"
    )

    vae = load_vae(args.vae, device)
    train_sentences, train_index = unique_sentences(train_pairs)
    train_latents = encode_sentences_cached(
        vae, train_sentences, args.train_cache, force_reencode=args.force_reencode
    )
    train_current = np.ascontiguousarray(
        train_latents[[train_index[first] for first, _ in train_pairs]]
    )
    train_actual = np.ascontiguousarray(
        train_latents[[train_index[second] for _, second in train_pairs]]
    )
    # eval latents come from the TASK 27 cache (same encoder, same sentences)
    val_sentences, val_index = unique_sentences(val_pairs)
    val_latents = encode_sentences_cached(
        vae, val_sentences, args.val_cache, force_reencode=args.force_reencode
    )
    eval_current = np.ascontiguousarray(
        val_latents[[val_index[first] for first, _ in eval_pairs]]
    )
    eval_actual = np.ascontiguousarray(
        val_latents[[val_index[second] for _, second in eval_pairs]]
    )
    eval_sources = np.arange(EVAL_SOURCE_BASE, EVAL_SOURCE_BASE + len(eval_pairs), dtype=np.int64)
    print(
        f"  latents: train {train_current.shape} / {train_actual.shape} | "
        f"eval {eval_current.shape} / {eval_actual.shape}"
    )

    # ------------------------------------------- frozen predictor + memory
    predictor = load_latent_predictor(args.predictor, device)
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    predictor.eval()
    print(
        f"\npredictor: {predictor.describe()} | {predictor.count_parameters():,} params "
        f"| FROZEN (identical in both arms; nothing trains in this loop)"
    )

    memory = RegionStore(
        num_regions=args.num_regions,
        latent_dim=LATENT_DIM,
        max_size=args.max_size,
        auto_fit_after=args.auto_fit_after,
        score_mode=SCORE_MODE,
        seed=args.seed,
    )
    print(
        f"memory   : RegionStore(num_regions={memory.num_regions}, "
        f"probe_regions={args.probe_regions}, max_size={memory.max_size}, "
        f"score_mode={memory.score_mode}, auto_fit_after={memory.auto_fit_after})"
    )
    print(
        "gate     : "
        + (
            "OFF (ARM B behaviour: always use the memory)"
            if args.no_gate
            else f"top-{args.gate_top_n} neighbour cosine >= {args.gate_threshold} "
            f"-> else C = zeros"
        )
    )
    return _run_loop(
        args, device, memory, predictor, train_pairs, train_current, train_actual,
        eval_current, eval_actual, eval_sources,
    )



def run_loop_core(
    predictor: LatentPredictor,
    memory: RegionStore,
    train_current: np.ndarray,
    train_actual: np.ndarray,
    eval_current: np.ndarray,
    eval_actual: np.ndarray,
    eval_sources: np.ndarray,
    eval_every: int = EVAL_EVERY,
    probe_regions: int = PROBE_REGIONS,
    gate_threshold: Optional[float] = GATE_THRESHOLD,
    gate_top_n: int = GATE_TOP_N,
    collect_steps: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """The reusable loop: predict -> remember -> (every ``eval_every`` steps) evaluate.

    Shared by ``main`` (``verbose=True``) and ``scripts/run_gated_sweep.py`` (one call
    per threshold, ``verbose=False``). Returns the checkpoint evaluations and, when
    ``collect_steps`` is set, the per-step training records.
    """
    zero_context = np.zeros((1, LATENT_DIM), dtype=np.float32)
    if verbose:
        print(
            "\n  step | A gated | B always | C none |  A-C   |  A-B   | gate_used"
            " | memory_size | regions"
        )
        print("  " + "-" * 96)

    step_records: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(len(train_current)):
        current = train_current[step]
        actual = train_actual[step]
        warmup = not memory.is_fitted
        step_sources = np.asarray([step], dtype=np.int64)

        contexts_a, gate_stats = contexts_for_arm_a(
            memory, current[None, :], step_sources, probe_regions, gate_threshold, gate_top_n
        )
        cos_a = float(
            cosine_to_actual(predict(predictor, current[None, :], contexts_a), actual[None, :])[0]
        )
        cos_c = float(
            cosine_to_actual(predict(predictor, current[None, :], zero_context), actual[None, :])[0]
        )

        # Step 4: remember AFTER predicting, weighted by how good the (gated) prediction was
        memory.add(current, max(OUTCOME_FLOOR, cos_a), target_vector=actual, source_index=step)
        if collect_steps:
            step_records.append(
                {
                    "step": step + 1,
                    "phase": "warmup" if warmup else "memory",
                    "arm_a_cos": cos_a,
                    "arm_c_cos": cos_c,
                    "delta_a_vs_c": cos_a - cos_c,
                    "gate_used": bool(gate_stats["fused"]),
                    "gated_out": bool(gate_stats["gated_out"]),
                    "confidence": (
                        gate_stats["confidences"][0] if gate_stats["confidences"] else None
                    ),
                    "memory_size": int(memory.size),
                    "memory_fitted": bool(memory.is_fitted),
                }
            )

        step_number = step + 1
        if step_number % eval_every == 0 or step_number == len(train_current):
            record = evaluate_arms(
                predictor,
                memory,
                eval_current,
                eval_actual,
                eval_sources,
                step_number,
                probe_regions,
                gate_threshold,
                gate_top_n,
            )
            checkpoints.append(record)
            if verbose:
                print(
                    f"  {record['step']:>4} | {record['mean_cos_arm_a']:>7.4f} | "
                    f"{record['mean_cos_arm_b']:>8.4f} | {record['mean_cos_arm_c']:>6.4f} | "
                    f"{record['delta_a_vs_c']:>+6.4f} | {record['delta_a_vs_b']:>+6.4f} | "
                    f"{100 * record['gate_rate']:>8.1f}% | {record['memory_size']:>11} | "
                    f"{record['n_regions']:>7} [{time.perf_counter() - started:.0f}s]"
                )

    return {
        "checkpoints": checkpoints,
        "step_records": step_records,
        "seconds": time.perf_counter() - started,
    }


def _run_loop(
    args: argparse.Namespace,
    device: torch.device,
    memory: RegionStore,
    predictor: LatentPredictor,
    train_pairs: Sequence[Tuple[str, str]],
    train_current: np.ndarray,
    train_actual: np.ndarray,
    eval_current: np.ndarray,
    eval_actual: np.ndarray,
    eval_sources: np.ndarray,
) -> Dict[str, Any]:
    """The loop itself: predict, remember, and evaluate all three arms every 500 steps.

    ``gate_threshold is None`` (``--no-gate``) reproduces TASK 29's ARM B: the memory
    context is always used. With a threshold, ARM A is the gated arm -- C only when the
    retrieved neighbours are close enough -- which is what this task adds.
    """
    gate_threshold = None if args.no_gate else float(args.gate_threshold)
    bands = [
        ("0-500 (warmup)", 1, 500),
        ("500-2048 (warmup)", 501, args.auto_fit_after),
        ("2048-5000 (memory active)", args.auto_fit_after + 1, 5000),
        ("5000-10000 (memory active)", 5001, len(train_pairs)),
    ]

    print(
        f"\nloop: {len(train_pairs)} steps | eval every {args.eval_every} steps on "
        f"{len(eval_current)} held-out pairs | warm-up until {args.auto_fit_after} adds "
        f"| gate: "
        + (
            "OFF (always memory = ARM B behaviour)"
            if gate_threshold is None
            else f"top-{args.gate_top_n} neighbour cosine >= {gate_threshold}"
        )
    )
    result = run_loop_core(
        predictor,
        memory,
        train_current,
        train_actual,
        eval_current,
        eval_actual,
        eval_sources,
        eval_every=args.eval_every,
        probe_regions=args.probe_regions,
        gate_threshold=gate_threshold,
        gate_top_n=args.gate_top_n,
        collect_steps=True,
        verbose=True,
    )
    return _report(
        args,
        memory,
        predictor,
        result["step_records"],
        result["checkpoints"],
        bands,
        eval_current,
        result["seconds"],
        gate_threshold,
    )



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(
    args: argparse.Namespace,
    memory: RegionStore,
    predictor: LatentPredictor,
    step_records: Sequence[Dict[str, Any]],
    checkpoints: Sequence[Dict[str, Any]],
    bands: Sequence[Tuple[str, int, int]],
    eval_current: np.ndarray,
    seconds: float,
    gate_threshold: Optional[float],
) -> Dict[str, Any]:
    """Print the final tables and write ``experiments/prediction_loop_results.json``."""
    band_rows = band_summary(checkpoints, bands)
    trend = trend_analysis(checkpoints, args.auto_fit_after)
    final = checkpoints[-1] if checkpoints else {}
    warmup_steps = [row for row in step_records if row["phase"] == "warmup"]
    active_steps = [row for row in step_records if row["phase"] == "memory"]

    print("\n" + "=" * 96)
    print("FINAL COMPARISON - held-out eval set (500 pairs that never enter the memory)")
    print("=" * 96)
    print(
        "  Steps                    | A gated | B always | C none |   A-C  |   A-B  |"
        " gate used | eval pts"
    )
    print("  " + "-" * 100)
    for row in band_rows:
        print(
            f"  {row['band']:<24} | {row['mean_cos_arm_a']:>7.4f} | "
            f"{row['mean_cos_arm_b']:>8.4f} | {row['mean_cos_arm_c']:>6.4f} | "
            f"{row['mean_delta_a_vs_c']:>+6.4f} | {row['mean_delta_a_vs_b']:>+6.4f} | "
            f"{100 * row['mean_gate_rate']:>9.1f}% | {len(row['checkpoints']):>8}"
        )
    print("  " + "-" * 100)
    if warmup_steps and active_steps:
        print(
            f"  train-step quality (not the headline): warm-up "
            f"{np.mean([row['arm_a_cos'] for row in warmup_steps]):.4f} (A) / "
            f"{np.mean([row['arm_c_cos'] for row in warmup_steps]):.4f} (C) | "
            f"memory-active "
            f"{np.mean([row['arm_a_cos'] for row in active_steps]):.4f} (A) / "
            f"{np.mean([row['arm_c_cos'] for row in active_steps]):.4f} (C) | "
            f"gate used on "
            f"{100 * np.mean([1.0 if row['gate_used'] else 0.0 for row in active_steps]):.1f}% "
            f"of memory-active steps"
        )


    print("\nKEY QUESTION - does the memory's contribution grow as the memory grows?")
    print(
        f"  delta at step {trend['delta_at_first_checkpoint_step']} "
        f"(warm-up, identical by construction)  : {trend['delta_at_first_checkpoint']:+.4f}"
    )
    if trend["delta_at_first_memory_active_checkpoint"] is not None:
        print(
            f"  delta at step {trend['first_memory_active_step']} "
            f"(first memory-active checkpoint)      : "
            f"{trend['delta_at_first_memory_active_checkpoint']:+.4f}"
        )
    print(
        f"  delta at step {trend['last_checkpoint_step']} "
        f"(last checkpoint)                     : {trend['delta_at_last_checkpoint']:+.4f}"
    )
    print(
        f"  change across the memory-active span    : "
        f"{trend['memory_active_growth']:+.4f} "
        f"(rho vs step {trend['spearman_rho_delta_vs_step_memory_active']:+.2f})"
    )
    print(
        f"  rho over ALL checkpoints (incl. warm-up): "
        f"{trend['spearman_rho_delta_vs_step_all']:+.2f} | "
        f"warm-up delta exactly zero: {trend['warmup_delta_is_zero']}"
    )

    print("\nSIGNIFICANCE at the last checkpoint (paired over the 500 eval pairs)")
    for label, stats in (
        ("ARM A (gated) vs ARM C (no memory)", final.get("arm_a_gated", {})),
        ("ARM A (gated) vs ARM B (always memory)", final.get("arm_a_vs_b", {})),
        ("ARM B (always) vs ARM C (no memory)", final.get("arm_b_ungated", {})),
    ):
        print(
            f"  {label:<38} delta {stats.get('mean_delta', 0):+.4f} | "
            f"A wins {100 * stats.get('share_arm_a_wins', 0):.1f}% "
            f"({stats.get('arm_a_wins', 0)}/{stats.get('num_examples', 0)}) | "
            f"paired t {stats.get('paired_t_statistic', 0):+.2f}"
        )
    gated_significant = abs(final.get("arm_a_gated", {}).get("paired_t_statistic", 0.0)) > 1.96
    gated_positive = final.get("arm_a_gated", {}).get("mean_delta", 0.0) > 0
    print(
        f"  |t| > 1.96 on A vs C -> "
        f"{'SIGNIFICANT' if gated_significant else 'NOT significant'} at 0.05; "
        f"gated arm {'BEATS' if gated_positive else 'does NOT beat'} no-memory "
        f"(examples are consecutive sentences from the same documents, so read the "
        f"t-statistic as an effect-size guide, not an exact p-value)"
    )


    print("\nMEMORY AT THE END")
    summary = memory.summary()
    print(
        f"  size {summary['size']} | routed {summary['routed_size']} | pending "
        f"{summary['pending_size']} | regions {summary['num_regions']} | "
        f"refits {summary['region_refits']} | mean outcome "
        f"{summary['mean_outcome_weight']:.4f}"
    )
    print(
        f"  queries {summary['queries']} | mean probes/query "
        f"{summary['mean_probes_per_query']:.2f} | visits: "
        f"{summary['visit_stats']['visited_regions']}/{summary['num_regions']} regions, "
        f"top-10% share {100 * summary['visit_stats']['top_10pct_visit_share']:.1f}% | "
        f"corr(visits, size) {summary['detail_stats']['pearson_visits_size']:+.3f}"
    )
    return _write_results(
        args, memory, predictor, step_records, checkpoints, band_rows, trend,
        final, gated_significant, summary, eval_current, seconds, gate_threshold,
    )



def _write_results(
    args: argparse.Namespace,
    memory: RegionStore,
    predictor: LatentPredictor,
    step_records: Sequence[Dict[str, Any]],
    checkpoints: Sequence[Dict[str, Any]],
    band_rows: Sequence[Dict[str, Any]],
    trend: Dict[str, Any],
    final: Dict[str, Any],
    gated_significant: bool,
    summary: Dict[str, Any],
    eval_current: np.ndarray,
    seconds: float,
    gate_threshold: Optional[float],
) -> Dict[str, Any]:
    """Build and save the results JSON (everything the DONE block quotes).

    Per-pair arrays (``_cos_a``/``_cos_b``/``_cos_c``) are dropped here: they exist for
    in-memory statistics, not for the artifact, and numpy arrays are not JSON.
    """
    serialisable = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in checkpoints
    ]
    payload = {
        "task": "TASK 30 - gated prediction loop (TASK 29 loop + retrieval-confidence gate)",
        "gate": {
            "enabled": gate_threshold is not None,
            "threshold": gate_threshold,
            "top_n": int(args.gate_top_n),
            "metric": "mean cosine of the top-n retrieved neighbours vs the query",
            "fallback": "C = zeros(256) when the gate rejects a query",
        },
        "config": {
            "vae": str(args.vae),
            "predictor": str(args.predictor),
            "train_pairs": str(args.train_pairs),
            "val_pairs": str(args.val_pairs),
            "steps": len(step_records),
            "eval_every": args.eval_every,
            "eval_size": int(eval_current.shape[0]),
            "eval_pair_indices": [args.eval_start, args.eval_start + int(eval_current.shape[0]) - 1],
            "num_regions": memory.num_regions,
            "probe_regions": args.probe_regions,
            "max_size": memory.max_size,
            "score_mode": memory.score_mode,
            "auto_fit_after": args.auto_fit_after,
            "knn_k": KNN_K,
            "fuse_top_k": FUSE_TOP_K,
            "outcome_floor": OUTCOME_FLOOR,
            "predictor_frozen": True,
            "predictor_parameters": predictor.count_parameters(),
            "seed": args.seed,
        },
        "protocol": {
            "arms": (
                "ARM A: C = fuse_directions(top-3 of the 10 nearest region memories, "
                "tangent-projected, self-masked) BUT only when the retrieval-confidence "
                "gate trusts the neighbours, else C = zeros; C = zeros during warm-up. "
                "ARM B: the same memory context without the gate (TASK 29 behaviour). "
                "ARM C: C = zeros always. Same pairs, same order, one frozen predictor."
            ),
            "eval_set": (
                f"val_pairs[{args.eval_start}:{args.eval_start + int(eval_current.shape[0])}] "
                "- never added to the memory (their source ids are reserved at 1000000+)"
            ),
            "caveat_memory_vs_model": (
                "these 500 pairs are held out from the MEMORY, not from the model: TASK 27 "
                "trained the predictor on all 7,666 val pairs, including these. The arms "
                "share the predictor, so the deltas are still clean memory effects, but "
                "'held-out' means 'never memorised' rather than 'never seen'."
            ),
            "state_note": (
                "eval queries are real queries: they increment region visits and hit counts, "
                "which feeds the adaptive-detail budget. ARM C does not query at all, so its "
                "baseline is unaffected; ARM A and ARM B query the same memory identically."
            ),
        },
        "checkpoints": serialisable,
        "bands": list(band_rows),
        "trend": trend,
        "final_comparison": {
            key: value for key, value in final.items() if not key.startswith("_")
        },
        "significance": {
            "arm_a_vs_c_paired_t": final.get("arm_a_gated", {}).get("paired_t_statistic"),
            "arm_a_vs_c_delta": final.get("arm_a_gated", {}).get("mean_delta"),
            "arm_a_vs_b_paired_t": final.get("arm_a_vs_b", {}).get("paired_t_statistic"),
            "arm_a_vs_b_delta": final.get("arm_a_vs_b", {}).get("mean_delta"),
            "arm_b_vs_c_paired_t": final.get("arm_b_ungated", {}).get("paired_t_statistic"),
            "arm_b_vs_c_delta": final.get("arm_b_ungated", {}).get("mean_delta"),
            "significant_at_0.05_approx": bool(gated_significant),
            "note": (
                "paired t over the 500 eval pairs; consecutive sentences from one document "
                "are correlated, so this mildly overstates significance"
            ),
        },
        "memory_final": summary,
        "region_stats_final": memory.region_stats(),
        "train_steps": list(step_records),
        "wall_seconds": seconds,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nsaved {args.results}")
    print("=" * 96)
    return payload


if __name__ == "__main__":
    main()

