"""TASK 27: train the latent predictor - does prediction improve as memory grows?

The claim under test
--------------------
    "Prediction accuracy in latent space improves as dynamic
     outcome-weighted memory grows. The model gets smarter with every
     inference."

Loop per sentence pair ``(S_n, S_n+1)`` of ``environment/data/wikitext/val_pairs.json``
(7,666 pairs), with the frozen VAE from ``checkpoints/vae.pt``:

    P_current = mu(S_n)          (256, unit-norm)
    C         = fuse_directions(top-3 memories)      <- dynamic memory
    P_pred    = LatentPredictor(P_current, C)
    loss      = 1 - cos(P_pred, mu(S_n+1))
    memory   += (P_current, mu(S_n+1), max(0.1, cos(P_pred, mu(S_n+1))))

Five passes over the same pairs simulate five "experiences" of the same data, and
the reported curve is the answer to the key question.

Two leak guards (both are methodological, not tuning)
-----------------------------------------------------
1. **Self-retrieval is excluded.** The memory is built from the very pairs being
   replayed, so from pass 2 onwards an example's own memory entry (identical
   ``P_current``, and therefore the exact answer as its target) would be its
   nearest neighbour. Every query drops the entries whose source pair index
   equals the example's, so what remains is genuine experience.
2. **Recall is reported twice.** ``top1_recall`` scores against the whole memory
   (the task's definition) and ``top1_recall_excl_self`` repeats it with the
   example's own entries masked out; the two agree on pass 1 and diverge as the
   replay accumulates, which is exactly the memorisation/generalisation split.

What is measured
----------------
* per pass: training loss, and a FROZEN evaluation (predictor weights and memory
  both fixed) giving mean cosine loss, mean cos, top-1 recall (both variants) and
  memory size - so "loss decreases over the passes" cannot be an artefact of the
  memory being updated mid-measurement;
* a memory-growth curve: the trained predictor, frozen, is replayed from an EMPTY
  memory, and the loss is bucketed by memory size. This isolates the effect of
  memory size from any further learning;
* two no-memory baselines: the same predictor with ``C = 0`` (isolates the
  memory's contribution to the trained model) and a separately trained predictor
  that never sees a context (controls for the predictor simply re-learning).

Reused assets
-------------
``checkpoints/vae.pt``                (frozen encoder - never retrained)
``src/memory/store.py``               (``normalize_rows``; FAISS inner product)
``src/sampler/jev.py``                (``parallel_sample``: weight, direction)
``src/fusion/fuse.py``                (``fuse_directions``, tangent fusion)
``src/predictor/latent_predictor.py`` (the module this task adds)

Artifacts
---------
    checkpoints/wikitext_val_predictor_vecs.npz   cached val sentence latents
    checkpoints/latent_predictor.pt               memory-guided predictor (TASK 29)
    checkpoints/latent_predictor_nomemory.pt      no-context baseline predictor
    experiments/predictor_history.json            every curve printed below
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

import faiss
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE  # noqa: E402
from src.fusion.fuse import fuse_directions  # noqa: E402
from src.memory.store import normalize_rows  # noqa: E402
from src.predictor.latent_predictor import (  # noqa: E402
    LATENT_DIM,
    LatentPredictor,
    cosine_prediction_loss,
    save_latent_predictor,
    top1_recall,
)
from src.sampler.jev import TARGET_CORRECT_ENDING, parallel_sample  # noqa: E402

DEFAULT_VAE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vae.pt"
DEFAULT_PAIRS_PATH = (
    PROJECT_ROOT / "environment" / "data" / "wikitext" / "val_pairs.json"
)
DEFAULT_VEC_CACHE = PROJECT_ROOT / "checkpoints" / "wikitext_val_predictor_vecs.npz"
DEFAULT_PREDICTOR_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "latent_predictor.pt"
DEFAULT_BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "latent_predictor_nomemory.pt"
)
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "experiments" / "predictor_history.json"

#: top-k memories retrieved per query (same as the Paper 1 memory store default)
KNN_K = 10
#: memories whose fused direction forms C (TASK 6 fusion)
FUSE_TOP_K = 3
#: TASK 27: with fewer than this many memories C is the zero vector
MIN_MEMORY_ENTRIES = 5
#: rebuild the FAISS index every N additions (IndexFlatIP is an exact search)
MEMORY_REBUILD_INTERVAL = 256
#: TASK 27: outcome weight floor - a wrong prediction still counts a little
OUTCOME_FLOOR = 0.1
DEFAULT_PASSES = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_GROWTH_BUCKETS = 10
#: examples used by the controlled memory-size sweep (fixed set, growing memory)
DEFAULT_SWEEP_SAMPLE = 1024
DEFAULT_EMBED_BATCH_SIZE = 128
#: rows per chunk when scoring the top-1 recall against the whole memory
RECALL_CHUNK = 512


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TASK 27: train the latent predictor on WikiText sentence pairs."
    )
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_VEC_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PREDICTOR_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    parser.add_argument("--fuse-top-k", type=int, default=FUSE_TOP_K)
    parser.add_argument("--growth-buckets", type=int, default=DEFAULT_GROWTH_BUCKETS)
    parser.add_argument(
        "--sweep-sample",
        type=int,
        default=DEFAULT_SWEEP_SAMPLE,
        help="examples scored in the controlled memory-size sweep",
    )
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="use only the first N pairs (smoke runs); 0 = all pairs",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "skip the first N pairs; with --limit this selects a slice, which the "
            "held-out arm uses to evaluate on pairs the encoder never trained on"
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="skip the separately trained no-memory predictor",
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="ignore the cached sentence latents and re-encode with the VAE",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Frozen VAE + data
# ---------------------------------------------------------------------------


def load_vae(checkpoint_path: Path, device: torch.device) -> VAE:
    """Rebuild the frozen VAE from ``vae.pt`` (same recipe as TASK 2/8 scripts)."""
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


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    """WikiText pairs: list of ``[S_n, S_n+1]`` (utf-8, never paired across documents)."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    pairs = [(str(first), str(second)) for first, second in raw]
    if not pairs:
        raise ValueError(f"{path} contains no sentence pairs")
    return pairs



def build_latent_vectors(
    vae: VAE,
    pairs: Sequence[Tuple[str, str]],
    cache_path: Path,
    force_reencode: bool = False,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> Tuple[List[str], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Encode every unique sentence once -> ``(sentences, index, current, actual, sources)``.

    ``mu`` is the VAE's deterministic latent (``encode_texts``) and is L2-normalised
    so cosine similarity is a dot product, exactly like ``MemoryStore`` and the
    fusion code expect. Encoding is the only expensive part of this task, so the
    vectors are cached (compare ``checkpoints/structured_val_vecs/``, which caches
    HellaSwag latents the same way); the cache is keyed on the sentence list, so a
    cache written for a different pair file is ignored rather than silently reused.

    Returns the unique sentence list, the ``sentence -> row`` mapping, and the
    per-pair ``(P_current, P_actual)`` matrices plus their pair indices.
    """
    sentences: List[str] = []
    sentence_index: Dict[str, int] = {}
    for first, second in pairs:
        for sentence in (first, second):
            if sentence not in sentence_index:
                sentence_index[sentence] = len(sentences)
                sentences.append(sentence)

    latent: Optional[np.ndarray] = None
    if cache_path.exists() and not force_reencode:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_sentences = [str(value) for value in cached["sentences"]]
            if cached_sentences == sentences:
                latent = cached["vectors"].astype(np.float32)
                print(f"loaded {latent.shape} cached sentence latents from {cache_path}")
        if latent is None:
            print(f"cache {cache_path} does not match the pair file - re-encoding")

    if latent is None:
        print(f"encoding {len(sentences)} unique sentences with the frozen VAE ...")
        started = time.perf_counter()
        with torch.no_grad():
            base = vae.encode_texts(sentences, batch_size=embed_batch_size).cpu().numpy()
        latent = normalize_rows(base)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, sentences=np.asarray(sentences), vectors=latent)
        print(
            f"  encoded in {time.perf_counter() - started:.1f}s -> {latent.shape}, "
            f"cached to {cache_path}"
        )

    current_rows = np.fromiter(
        (sentence_index[first] for first, _ in pairs), dtype=np.int64, count=len(pairs)
    )
    actual_rows = np.fromiter(
        (sentence_index[second] for _, second in pairs), dtype=np.int64, count=len(pairs)
    )
    return (
        sentences,
        sentence_index,
        latent[current_rows],
        latent[actual_rows],
        np.arange(len(pairs), dtype=np.int64),
    )



# ---------------------------------------------------------------------------
# Dynamic experience memory (current -> actual, outcome weighted)
# ---------------------------------------------------------------------------


class ExperienceMemory:
    """Growing ``(P_current -> P_actual)`` memory queried by inner product.

    Deliberately not ``src.memory.store.MemoryStore``: that store is a *static*
    HellaSwag store whose record schema (four candidate endings, label, activity
    label, ctx) does not describe a WikiText sentence pair, and it is built once
    from a fixed training file. The FAISS index type and the project's row
    normalisation (``normalize_rows``) are reused, and TASK 28 replaces this class
    with the full ``DynamicMemoryStore`` (forget policy, hit counts, EMA outcome
    updates).

    Storage is preallocated and grown geometrically, because ``np.vstack`` per
    addition would copy up to 7,666 x 256 float32 at every step; additions are O(1)
    and the FAISS index is rebuilt from the filled slice only every
    ``rebuild_interval`` additions (``IndexFlatIP`` is exact, so this only delays
    the visibility of the newest neighbours, never the correctness of a search).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        rebuild_interval: int = MEMORY_REBUILD_INTERVAL,
        initial_capacity: int = 1024,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.rebuild_interval = int(rebuild_interval)
        self._capacity = max(1, int(initial_capacity))
        self._size = 0
        self._vectors = np.zeros((self._capacity, self.latent_dim), dtype=np.float32)
        self._targets = np.zeros((self._capacity, self.latent_dim), dtype=np.float32)
        self._outcomes = np.zeros(self._capacity, dtype=np.float64)
        self._sources = np.full(self._capacity, -1, dtype=np.int64)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._indexed_size = 0
        self.rebuilds = 0

    # ------------------------------------------------------------------ views
    @property
    def size(self) -> int:
        """Number of stored experiences."""
        return self._size

    @property
    def current_vectors(self) -> np.ndarray:
        """``(M, 256)`` stored positions (filled slice, unit-norm)."""
        return self._vectors[: self._size]

    @property
    def target_vectors(self) -> np.ndarray:
        """``(M, 256)`` stored next positions (filled slice, unit-norm)."""
        return self._targets[: self._size]

    @property
    def outcome_weights(self) -> np.ndarray:
        """``(M,)`` outcome weight of every experience."""
        return self._outcomes[: self._size]

    @property
    def source_indices(self) -> np.ndarray:
        """``(M,)`` index of the pair each experience came from (leak masking)."""
        return self._sources[: self._size]

    @property
    def mean_outcome_weight(self) -> float:
        return float(self._outcomes[: self._size].mean()) if self._size else 0.0

    # ------------------------------------------------------------- additions
    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        capacity = self._capacity
        while capacity < needed:
            capacity *= 2
        for name in ("_vectors", "_targets"):
            grown = np.zeros((capacity, self.latent_dim), dtype=np.float32)
            grown[: self._size] = getattr(self, name)[: self._size]
            setattr(self, name, grown)
        outcomes = np.zeros(capacity, dtype=np.float64)
        outcomes[: self._size] = self._outcomes[: self._size]
        self._outcomes = outcomes
        sources = np.full(capacity, -1, dtype=np.int64)
        sources[: self._size] = self._sources[: self._size]
        self._sources = sources
        self._capacity = capacity

    def add(
        self,
        current_vector: np.ndarray,
        actual_vector: np.ndarray,
        outcome_weight: float,
        source_index: int,
    ) -> int:
        """Append one experience; returns its position in memory."""
        self._ensure_capacity(self._size + 1)
        self._vectors[self._size] = normalize_rows(
            np.asarray(current_vector, dtype=np.float32).reshape(1, -1)
        )[0]
        self._targets[self._size] = normalize_rows(
            np.asarray(actual_vector, dtype=np.float32).reshape(1, -1)
        )[0]
        self._outcomes[self._size] = float(outcome_weight)
        self._sources[self._size] = int(source_index)
        position = self._size
        self._size += 1
        return position


    # -------------------------------------------------------------- retrieval
    def _ensure_index(self) -> None:
        """Rebuild the FAISS index when it has drifted far enough behind."""
        if (
            self._index is not None
            and self._size - self._indexed_size < self.rebuild_interval
        ):
            return
        index = faiss.IndexFlatIP(self.latent_dim)
        index.add(np.ascontiguousarray(self.current_vectors, dtype=np.float32))
        self._index = index
        self._indexed_size = self._size
        self.rebuilds += 1

    def query(
        self,
        vector: np.ndarray,
        k: int = KNN_K,
        exclude_source: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Top-``k`` nearest experiences, sorted by cosine similarity.

        ``exclude_source`` drops every experience that came from that pair index.
        The search asks for one extra row when an exclusion is active: an example
        has at most one entry per pass, and from pass 2 onwards that entry is its
        exact replay (cosine 1.0, target = the answer), so it would otherwise be
        the nearest neighbour and leak the label.
        """
        if self._size == 0:
            return []
        self._ensure_index()
        wanted = max(1, int(k))
        search_k = min(self._size, wanted + (1 if exclude_source is not None else 0))
        if search_k <= 0:
            return []

        query_vector = normalize_rows(
            np.asarray(vector, dtype=np.float32).reshape(1, -1)
        ).astype(np.float32)
        similarities, positions = self._index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32), search_k
        )

        records: List[Dict[str, Any]] = []
        for similarity, position in zip(similarities[0], positions[0]):
            if position < 0:  # FAISS pads with -1 when fewer than k rows exist
                continue
            position = int(position)
            if exclude_source is not None and int(
                self._sources[position]
            ) == int(exclude_source):
                continue
            records.append(
                {
                    "position": position,
                    "similarity": float(similarity),
                    "vector": self._vectors[position],
                    "target": self._targets[position],
                    "outcome": float(self._outcomes[position]),
                    "source": int(self._sources[position]),
                }
            )
        return records

    def summary(self) -> Dict[str, Any]:
        return {
            "size": self._size,
            "latent_dim": self.latent_dim,
            "mean_outcome_weight": self.mean_outcome_weight,
            "faiss_rebuilds": self.rebuilds,
        }


# ---------------------------------------------------------------------------
# Context C: fused neighbour directions (existing sampler + fusion code)
# ---------------------------------------------------------------------------


def fuse_context(
    query_vector: np.ndarray,
    memory: Optional[ExperienceMemory],
    exclude_source: Optional[int] = None,
    knn_k: int = KNN_K,
    fuse_top_k: int = FUSE_TOP_K,
) -> np.ndarray:
    """Context vector ``C`` for one query = tangent-fused top-``k`` memory directions.

    Exactly the Paper 1 retrieval path, now used as the predictor's input rather
    than as a score:

        MemoryStore.query -> top-``knn_k`` neighbours
        parallel_sample   -> (weight, direction) per neighbour
        fuse_directions   -> weighted top-``fuse_top_k`` sum, projected onto the
                             query's tangent space and L2-normalised

    ``memory is None`` (or fewer than ``MIN_MEMORY_ENTRIES`` usable experiences)
    gives ``zeros(256)``, which is TASK 27's cold-start rule and is also how the
    no-memory baseline is expressed.

    ``parallel_sample`` prints one line per call for interactive use; the loop
    calls it ~150k times, so its stdout is redirected into a throw-away buffer
    instead of flooding the log.
    """
    if memory is None or memory.size == 0:
        return np.zeros(LATENT_DIM, dtype=np.float64)

    records = memory.query(query_vector, k=knn_k, exclude_source=exclude_source)
    if len(records) < MIN_MEMORY_ENTRIES:
        return np.zeros(LATENT_DIM, dtype=np.float64)

    memories = [
        {"vector": record["vector"], "correct_ending_vector": record["target"]}
        for record in records
    ]
    outcome_weights = [record["outcome"] for record in records]
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


def build_contexts(
    memory: Optional[ExperienceMemory],
    current_vectors: np.ndarray,
    sources: np.ndarray,
    knn_k: int = KNN_K,
    fuse_top_k: int = FUSE_TOP_K,
) -> np.ndarray:
    """``(N, 256)`` context matrix: ``fuse_context`` for every row of ``current_vectors``.

    ``memory is None`` yields the all-zero context matrix, i.e. the no-memory
    condition. Keeping this in one place means the training loop, the frozen
    evaluation and the ablations all build ``C`` the same way.
    """
    total = int(current_vectors.shape[0])
    contexts = np.zeros((total, LATENT_DIM), dtype=np.float32)
    for step in range(total):
        contexts[step] = fuse_context(
            current_vectors[step], memory, exclude_source=int(sources[step]),
            knn_k=knn_k, fuse_top_k=fuse_top_k,
        )
    return contexts




# ---------------------------------------------------------------------------
# Prediction, loss and recall
# ---------------------------------------------------------------------------


def _float32_rows(vectors) -> np.ndarray:
    """Row-major float32 view/copy (``torch.from_numpy`` needs both properties)."""
    return np.ascontiguousarray(vectors, dtype=np.float32)


@torch.no_grad()
def predict_matrix(predictor, current_vectors, contexts, batch_size: int) -> np.ndarray:
    """Batched ``(N, 256)`` unit predictions for the ``(P_current, C)`` rows."""
    predictor.eval()
    total = int(current_vectors.shape[0])
    predicted = np.zeros((total, LATENT_DIM), dtype=np.float32)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        out = predictor(
            _float32_rows(current_vectors[start:stop]),
            _float32_rows(contexts[start:stop]),
        )
        predicted[start:stop] = out.detach().cpu().numpy()
    return predicted


def per_example_loss(predicted, actual_vectors) -> Tuple[np.ndarray, np.ndarray]:
    """``(loss, cos)`` arrays of length N for already-computed predictions."""
    with torch.no_grad():
        loss, cos = cosine_prediction_loss(
            torch.from_numpy(_float32_rows(predicted)),
            torch.from_numpy(_float32_rows(actual_vectors)),
        )
    return loss.numpy(), cos.numpy()


def recall_stats(
    predicted: np.ndarray,
    actual_vectors: np.ndarray,
    memory: ExperienceMemory,
    sources: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(hits_all, hits_excl_self)``: is the true next position the nearest memory row?

    Chunked over examples because the similarity matrix is ``(N, memory size)``.
    The caller freezes the memory, so every chunk is scored against the same rows.
    """
    total = int(predicted.shape[0])
    hits_all = np.zeros(total, dtype=bool)
    hits_excl = np.zeros(total, dtype=bool)
    if memory.size == 0:
        return hits_all, hits_excl
    targets = memory.target_vectors
    source_ids = memory.source_indices
    for start in range(0, total, RECALL_CHUNK):
        stop = min(start + RECALL_CHUNK, total)
        block = predicted[start:stop].astype(np.float64)
        actual = actual_vectors[start:stop].astype(np.float64)
        hits_all[start:stop] = top1_recall(block, targets, actual)
        allowed = source_ids[None, :] != sources[start:stop, None]
        hits_excl[start:stop] = top1_recall(block, targets, actual, allowed_mask=allowed)
    return hits_all, hits_excl


def paired_cos_comparison(
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    actual_vectors: np.ndarray,
) -> Dict[str, Any]:
    """Paired per-example comparison of two prediction sets for the SAME examples.

    Every example is scored under both conditions, so the difference is not
    polluted by which examples happen to be easier (which is exactly what makes an
    unpaired bucket curve unreadable on this corpus: WikiText pairs are ordered by
    document, so a bucket is a run of related topics, not a random sample).

    Reported: mean cosine per condition, the paired delta, how many examples each
    condition wins, and a paired t-statistic. The t-statistic treats examples as
    independent, which they are not (neighbouring pairs share sentences and
    topics), so read it as a rough effect-size guide rather than an exact p-value -
    Paper 1's exact McNemar tests were for binary accuracy, not for cosines.
    """
    _, cos_a = per_example_loss(predicted_a, actual_vectors)
    _, cos_b = per_example_loss(predicted_b, actual_vectors)
    delta = cos_a.astype(np.float64) - cos_b.astype(np.float64)
    total = int(delta.size)
    std = float(delta.std(ddof=1)) if total > 1 else 0.0
    standard_error = std / float(np.sqrt(total)) if total else 0.0
    return {
        "num_examples": total,
        "mean_cos_a": float(cos_a.mean()),
        "mean_cos_b": float(cos_b.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)) if total else 0.0,
        "delta_std": std,
        "paired_t_statistic": float(delta.mean() / standard_error) if standard_error > 0 else 0.0,
        "a_wins": int((delta > 0).sum()),
        "b_wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "share_a_wins": float((delta > 0).mean()) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# The experience loop
# ---------------------------------------------------------------------------


def train_pass(
    predictor: LatentPredictor,
    optimizer: torch.optim.Optimizer,
    memory: Optional[ExperienceMemory],
    current_vectors: np.ndarray,
    actual_vectors: np.ndarray,
    sources: np.ndarray,
    batch_size: int,
    label: str,
) -> Dict[str, Any]:
    """One sequential pass: predict, learn, then remember each experience.

    Per example: build ``C`` from the memory as it stands *now*, predict, record the
    outcome weight ``max(0.1, cos)`` and append ``(P_current, P_actual, weight)``.
    Gradients are not applied inside the loop: the per-example losses are kept as
    tensors and the accumulated mini-batch is stepped once every ``batch_size``
    examples (one forward per example, no staleness beyond the current batch).
    """
    predictor.train()
    total = int(current_vectors.shape[0])
    loss_sum = 0.0
    cos_sum = 0.0
    pending: List[torch.Tensor] = []
    gradient_steps = 0
    contexts_used = 0
    started = time.perf_counter()
    report_every = max(1, total // 10)

    for step in range(total):
        current = current_vectors[step]
        context = fuse_context(
            current, memory, exclude_source=int(sources[step])
        )
        if float(np.abs(context).sum()) > 0.0:
            contexts_used += 1
        predicted = predictor(
            torch.from_numpy(current).unsqueeze(0),
            torch.from_numpy(_float32_rows(context)).unsqueeze(0),
        )
        actual = torch.from_numpy(actual_vectors[step]).unsqueeze(0)
        loss, cos = cosine_prediction_loss(predicted, actual)
        loss_sum += float(loss.detach())
        cos_sum += float(cos.detach())

        if memory is not None:
            # TASK 27: outcome weight == how good this prediction turned out to be
            quality = float(cos.detach())
            memory.add(
                current,
                actual_vectors[step],
                max(OUTCOME_FLOOR, quality),
                int(sources[step]),
            )

        pending.append(loss.squeeze(0))
        if len(pending) >= batch_size or step == total - 1:
            batch_loss = torch.stack(pending).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            pending = []
            gradient_steps += 1
            if gradient_steps % report_every == 0 or step == total - 1:
                print(
                    f"    [{label}] {step + 1:>6}/{total} | loss {loss_sum / (step + 1):.4f} "
                    f"| cos {cos_sum / (step + 1):.4f} "
                    f"| memory {memory.size if memory is not None else 0} "
                    f"| {time.perf_counter() - started:.0f}s"
                )

    return {
        "num_examples": total,
        "mean_loss": loss_sum / total,
        "mean_cos": cos_sum / total,
        "contexts_used": contexts_used,
        "memory_size": memory.size if memory is not None else 0,
        "gradient_steps": gradient_steps,
        "seconds": time.perf_counter() - started,
    }



@torch.no_grad()
def evaluate_pass(
    predictor: LatentPredictor,
    memory: Optional[ExperienceMemory],
    current_vectors: np.ndarray,
    actual_vectors: np.ndarray,
    sources: np.ndarray,
    batch_size: int,
    contexts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Frozen evaluation: no weight updates and no memory updates.

    Freezing both is what makes the per-pass curve interpretable - between two of
    these evaluations only the memory contents and the weights learned in between
    have changed, and the measurement itself cannot move the goalposts.

    ``memory is None`` is the no-memory condition of TASK 27 item 3: every context
    is ``zeros(256)``, so the predictor sees the current position only. A caller
    that has already built the contexts (e.g. for a paired comparison) can pass
    them in instead of rebuilding them.
    """
    if contexts is None:
        contexts = build_contexts(memory, current_vectors, sources)
    total = int(current_vectors.shape[0])
    zero_context = int((np.abs(contexts).sum(axis=1) == 0.0).sum())
    predicted = predict_matrix(predictor, current_vectors, contexts, batch_size)
    loss, cos = per_example_loss(predicted, actual_vectors)

    stats: Dict[str, Any] = {
        "num_examples": total,
        "mean_loss": float(loss.mean()),
        "mean_cos": float(cos.mean()),
        "cos_std": float(cos.std()),
        "cos_p10": float(np.percentile(cos, 10)),
        "cos_p90": float(np.percentile(cos, 90)),
        "memory_size": memory.size if memory is not None else 0,
        "zero_context_fraction": zero_context / total,
    }
    if memory is not None and memory.size > 0:
        hits_all, hits_excl = recall_stats(predicted, actual_vectors, memory, sources)
        stats["top1_recall"] = float(hits_all.mean())
        stats["top1_recall_excl_self"] = float(hits_excl.mean())
    return stats


def run_experiences(
    predictor: LatentPredictor,
    optimizer: torch.optim.Optimizer,
    memory: Optional[ExperienceMemory],
    current_vectors: np.ndarray,
    actual_vectors: np.ndarray,
    sources: np.ndarray,
    passes: int,
    batch_size: int,
    tag: str,
) -> List[Dict[str, Any]]:
    """``passes`` sequential experience passes, each followed by a frozen evaluation."""
    records: List[Dict[str, Any]] = []
    for number in range(1, passes + 1):
        size_at_start = memory.size if memory is not None else 0
        print(
            f"\n  [{tag}] pass {number}/{passes} | memory {size_at_start} entries at start"
        )
        train_stats = train_pass(
            predictor, optimizer, memory, current_vectors, actual_vectors, sources,
            batch_size, label=f"{tag} p{number}",
        )
        eval_stats = evaluate_pass(
            predictor, memory, current_vectors, actual_vectors, sources, batch_size
        )
        recall = (
            f" | recall {eval_stats['top1_recall']:.4f} "
            f"(excl. self {eval_stats['top1_recall_excl_self']:.4f})"
            if "top1_recall" in eval_stats
            else ""
        )
        print(
            f"  [{tag}] pass {number} train loss {train_stats['mean_loss']:.4f} "
            f"| frozen eval loss {eval_stats['mean_loss']:.4f} "
            f"cos {eval_stats['mean_cos']:.4f}{recall}"
        )
        records.append({"pass": number, "train": train_stats, "eval": eval_stats})
    return records


@torch.no_grad()
def memory_growth_curve(
    predictor: LatentPredictor,
    current_vectors: np.ndarray,
    actual_vectors: np.ndarray,
    sources: np.ndarray,
    buckets: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Loss vs memory size with a FROZEN predictor, replaying from an empty memory.

    The predictor is not trained here and the memory starts empty, so any trend in
    this curve is caused by memory size alone. One experience is added per example,
    so the memory size during bucket ``b`` is exactly ``[lo, hi)`` of the pair
    indices - the honest x-axis for the "grows with experience" claim.
    """
    predictor.eval()
    memory = ExperienceMemory()
    total = int(current_vectors.shape[0])
    edges = np.linspace(0, total, buckets + 1).astype(int)
    curve: List[Dict[str, Any]] = []
    for index in range(buckets):
        low, high = int(edges[index]), int(edges[index + 1])
        if high <= low:
            continue
        losses: List[float] = []
        cosines: List[float] = []
        for step in range(low, high):
            context = fuse_context(
                current_vectors[step], memory, exclude_source=int(sources[step])
            )
            predicted = predictor(
                torch.from_numpy(current_vectors[step]).unsqueeze(0),
                torch.from_numpy(_float32_rows(context)).unsqueeze(0),
            )
            loss, cos = cosine_prediction_loss(
                predicted, torch.from_numpy(actual_vectors[step]).unsqueeze(0)
            )
            losses.append(float(loss))
            cosines.append(float(cos))
            quality = float(cos)
            memory.add(
                current_vectors[step],
                actual_vectors[step],
                max(OUTCOME_FLOOR, quality),
                int(sources[step]),
            )
        curve.append(
            {
                "bucket": index + 1,
                "memory_size_start": low,
                "memory_size_end": high,
                "num_examples": high - low,
                "mean_loss": float(np.mean(losses)),
                "mean_cos": float(np.mean(cosines)),
            }
        )
        print(
            f"  growth bucket {index + 1}/{buckets} | memory {low}-{high} "
            f"| loss {curve[-1]['mean_loss']:.4f} | cos {curve[-1]['mean_cos']:.4f}"
        )
    return curve, memory.size


def prefix_memory(memory: ExperienceMemory, size: int) -> ExperienceMemory:
    """A new memory holding the first ``size`` experiences of ``memory``.

    Used to vary how much experience is available while everything else (the
    examples being scored, the predictor, the fusion code) is held fixed.
    """
    size = min(int(size), memory.size)
    prefix = ExperienceMemory(
        latent_dim=memory.latent_dim, rebuild_interval=memory.rebuild_interval
    )
    for position in range(size):
        prefix.add(
            memory.current_vectors[position],
            memory.target_vectors[position],
            float(memory.outcome_weights[position]),
            int(memory.source_indices[position]),
        )
    return prefix


@torch.no_grad()
def memory_size_sweep(
    predictor: LatentPredictor,
    memory: ExperienceMemory,
    current_vectors: np.ndarray,
    actual_vectors: np.ndarray,
    sources: np.ndarray,
    sizes: Sequence[int],
    sample_size: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Same examples, growing memory: does more experience make the loss smaller?

    The sequential growth curve (``memory_growth_curve``) confounds memory size
    with which examples fall in which bucket. This sweep removes that confound:
    one FIXED sample of pairs is scored against memories of increasing size (the
    prefixes of the trained memory), so the only variable left is how much
    experience is available. Each size is also paired against the memoryless
    prediction (``size = 0``, i.e. ``C = 0``) on the very same examples.

    The sample is taken from the END of the pair file, so a small prefix cannot
    contain the sample's own experiences; where the full memory does contain them,
    ``exclude_source`` masks them out.
    """
    total = int(current_vectors.shape[0])
    sample_size = max(1, min(int(sample_size), total))
    sample = np.arange(total - sample_size, total)
    sample_current = current_vectors[sample]
    sample_actual = actual_vectors[sample]
    sample_sources = sources[sample]

    curve: List[Dict[str, Any]] = []
    memoryless_predicted: Optional[np.ndarray] = None
    for raw_size in sizes:
        size = max(0, min(int(raw_size), memory.size))
        prefix = prefix_memory(memory, size)
        contexts = build_contexts(prefix, sample_current, sample_sources)
        predicted = predict_matrix(predictor, sample_current, contexts, batch_size)
        loss, cos = per_example_loss(predicted, sample_actual)
        entry: Dict[str, Any] = {
            "memory_size": prefix.size,
            "num_examples": sample_size,
            "mean_loss": float(loss.mean()),
            "mean_cos": float(cos.mean()),
            "zero_context_fraction": float((np.abs(contexts).sum(axis=1) == 0.0).mean()),
        }
        if memoryless_predicted is None:
            memoryless_predicted = predicted  # the size = 0 case is the C = 0 reference
        else:
            entry["paired_vs_memoryless"] = paired_cos_comparison(
                predicted, memoryless_predicted, sample_actual
            )
        curve.append(entry)
        paired = entry.get("paired_vs_memoryless")
        print(
            f"  sweep memory {prefix.size:>5} | loss {entry['mean_loss']:.4f} "
            f"| cos {entry['mean_cos']:.4f}"
            + (
                f" | paired delta {paired['mean_delta']:+.4f} "
                f"(wins {100 * paired['share_a_wins']:.1f}%, t {paired['paired_t_statistic']:+.2f})"
                if paired
                else " | (memoryless reference)"
            )
        )
    return curve



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(history: Dict[str, Any]) -> None:
    """Print the table the DONE block is built from."""
    memory_run = history["with_memory"]
    comparison = history["comparison"]

    print("\n" + "=" * 84)
    print("TASK 27 SUMMARY - latent predictor / dynamic memory")
    print("=" * 84)
    print(f"  predictor parameters        : {history['predictor']['parameter_count']:,}")
    print(f"  architecture                : {history['predictor']['description']}")
    print(
        f"  pairs / passes / batch size : {history['data']['num_pairs']} / "
        f"{history['data']['passes']} / {history['config']['batch_size']}"
    )
    print(f"  latent dim / knn k / fuse k : {LATENT_DIM} / {history['config']['knn_k']} / "
          f"{history['config']['fuse_top_k']}")

    print("\n  memory-guided run (frozen evaluation after every pass)")
    print("  pass |  train loss |  eval loss |  eval cos | top1 recall | excl. self | memory")
    print("  " + "-" * 82)
    for record in memory_run["passes"]:
        stats = record["eval"]
        recall = stats.get("top1_recall")
        recall_excl = stats.get("top1_recall_excl_self")
        print(
            f"  {record['pass']:>4} | {record['train']['mean_loss']:>11.4f} "
            f"| {stats['mean_loss']:>10.4f} | {stats['mean_cos']:>9.4f} "
            f"| {'n/a' if recall is None else f'{recall:.4f}':>11} "
            f"| {'n/a' if recall_excl is None else f'{recall_excl:.4f}':>9} "
            f"| {stats['memory_size']:>6}"
        )

    print("\n  frozen-predictor memory-growth curve (memory replayed from empty)")
    print("  bucket | memory size |  eval loss |  eval cos")
    print("  " + "-" * 48)
    for bucket in memory_run["growth_curve"]:
        print(
            f"  {bucket['bucket']:>6} | {bucket['memory_size_start']:>4}-"
            f"{bucket['memory_size_end']:<5} | {bucket['mean_loss']:>10.4f} "
            f"| {bucket['mean_cos']:>9.4f}"
        )
    print(
        "    (unpaired - each bucket is a different run of topically ordered "
        "examples, so this mixes memory size with example difficulty)"
    )

    print("\n  controlled memory-size sweep (FIXED example set, memory prefixes)")
    print("  memory size |  eval loss |  eval cos | paired delta vs C=0 | wins |     t")
    print("  " + "-" * 78)
    for entry in memory_run["memory_size_sweep"]:
        paired = entry.get("paired_vs_memoryless")
        if paired is None:
            print(
                f"  {entry['memory_size']:>11} | {entry['mean_loss']:>10.4f} "
                f"| {entry['mean_cos']:>9.4f} | {'(memoryless reference)':>19} "
                f"| {'':>4} |"
            )
        else:
            print(
                f"  {entry['memory_size']:>11} | {entry['mean_loss']:>10.4f} "
                f"| {entry['mean_cos']:>9.4f} | {paired['mean_delta']:>+19.4f} "
                f"| {100 * paired['share_a_wins']:>3.0f}% | {paired['paired_t_statistic']:>+6.2f}"
            )

    print("\n  no-memory baselines")
    zero = comparison["zero_context_same_predictor"]
    print(
        f"    same predictor, C = 0            : loss {zero['mean_loss']:.4f} "
        f"| cos {zero['mean_cos']:.4f}"
    )
    baseline = comparison.get("no_memory_baseline_predictor")
    if baseline is None:
        print("    separately trained no-memory model: skipped")
    else:
        print(
            f"    separately trained no-memory model: loss {baseline['mean_loss']:.4f} "
            f"| cos {baseline['mean_cos']:.4f}"
        )

    print("\n  key answers")
    with_memory = comparison["with_memory"]
    paired_zero = comparison["paired_memory_vs_zero_context"]
    print(
        f"    eval loss decreased over the {len(memory_run['passes'])} passes : "
        f"{with_memory['eval_loss_decreased_over_passes']} "
        f"({memory_run['passes'][0]['eval']['mean_loss']:.4f} -> "
        f"{with_memory['final_eval_loss']:.4f})"
    )
    print(
        f"    memory vs C = 0 (same model, paired): delta "
        f"{paired_zero['mean_delta']:+.4f} cos | wins "
        f"{100 * paired_zero['share_a_wins']:.1f}% | t "
        f"{paired_zero['paired_t_statistic']:+.2f} -> helps: "
        f"{comparison['memory_helps_vs_zero_context_same_predictor']}"
    )
    paired_baseline = comparison.get("paired_memory_vs_no_memory_predictor")
    if "memory_helps_vs_no_memory_predictor" in comparison:
        print(
            f"    memory vs no-memory model (paired) : delta "
            f"{paired_baseline['mean_delta']:+.4f} cos | wins "
            f"{100 * paired_baseline['share_a_wins']:.1f}% | t "
            f"{paired_baseline['paired_t_statistic']:+.2f} -> helps: "
            f"{comparison['memory_helps_vs_no_memory_predictor']}"
        )
    growth = comparison["memory_growth"]
    print(
        f"    growth curve last vs first bucket  : {growth['first_bucket_loss']:.4f} -> "
        f"{growth['last_bucket_loss']:.4f} "
        f"(improves: {growth['loss_improves_with_memory']}, unpaired)"
    )
    sweep_summary = comparison["memory_size_sweep"]
    print(
        f"    sweep loss at smallest vs largest  : {sweep_summary['losses'][0]:.4f} "
        f"-> {sweep_summary['losses'][-1]:.4f} "
        f"(improves: {sweep_summary['loss_improves_with_memory']}, paired on "
        f"{sweep_summary['sample_size']} examples)"
    )
    print(
        f"    sweep last paired t vs C = 0       : "
        f"{sweep_summary['paired_t_vs_memoryless'][-1]:+.2f} "
        f"(delta {sweep_summary['paired_deltas_vs_memoryless'][-1]:+.4f})"
    )
    print(f"    final memory size                  : {with_memory['memory_size']}")
    print("=" * 84)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 84)
    print("TASK 27 - latent predictor: prediction in latent space")
    print("=" * 84)

    vae = load_vae(args.vae, device)
    pairs = load_pairs(args.pairs)
    if args.offset or args.limit:
        start = max(0, int(args.offset))
        stop = len(pairs) if not args.limit else start + int(args.limit)
        pairs = pairs[start:stop]
        print(f"pair slice [{start}:{stop}] -> {len(pairs)} pairs")
    sentences, _, current_vectors, actual_vectors, sources = build_latent_vectors(
        vae,
        pairs,
        args.cache,
        force_reencode=args.force_reencode,
        embed_batch_size=args.embed_batch_size,
    )
    current_vectors = _float32_rows(current_vectors)
    actual_vectors = _float32_rows(actual_vectors)
    print(
        f"pairs {len(pairs)} | unique sentences {len(sentences)} "
        f"| P_current {current_vectors.shape} | P_actual {actual_vectors.shape}"
    )

    predictor = LatentPredictor()
    print(f"predictor   : {predictor.describe()}")
    print(f"parameters  : {predictor.count_parameters():,}")

    # ---- the memory-guided run -------------------------------------------
    print("\n" + "-" * 84)
    print("memory-guided: kNN -> parallel sampler -> tangent fusion -> predictor")
    print("-" * 84)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    memory = ExperienceMemory()
    records = run_experiences(
        predictor,
        optimizer,
        memory,
        current_vectors,
        actual_vectors,
        sources,
        args.passes,
        args.batch_size,
        tag="memory",
    )
    save_latent_predictor(
        predictor,
        args.checkpoint,
        extras={
            "task": "TASK 27 latent predictor",
            "pairs_file": str(args.pairs),
            "passes": args.passes,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "final_memory_size": memory.size,
            "final_eval": records[-1]["eval"],
        },
    )
    print(f"\nsaved memory-guided predictor -> {args.checkpoint}")

    # ---- frozen-predictor memory-growth curve -----------------------------
    print("\n" + "-" * 84)
    print("memory-growth curve: trained predictor FROZEN, memory replayed from empty")
    print("-" * 84)
    growth_curve, growth_memory_size = memory_growth_curve(
        predictor, current_vectors, actual_vectors, sources, args.growth_buckets
    )

    # ---- no-memory baselines ---------------------------------------------
    print("\n" + "-" * 84)
    print("no-memory baseline: C = zeros(256) everywhere")
    print("-" * 84)
    same_predictor_no_memory = evaluate_pass(
        predictor, None, current_vectors, actual_vectors, sources, args.batch_size
    )
    print(
        f"  same trained predictor, C = 0 : loss "
        f"{same_predictor_no_memory['mean_loss']:.4f} | cos "
        f"{same_predictor_no_memory['mean_cos']:.4f}"
    )

    baseline_records: List[Dict[str, Any]] = []
    baseline_predictor: Optional[LatentPredictor] = None
    if args.skip_baseline:
        print("  separately trained no-memory predictor: skipped (--skip-baseline)")
    else:
        # same seed / data / schedule, the only difference is that C is never built
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        baseline_predictor = LatentPredictor()
        baseline_optimizer = torch.optim.Adam(baseline_predictor.parameters(), lr=args.lr)
        baseline_records = run_experiences(
            baseline_predictor,
            baseline_optimizer,
            None,
            current_vectors,
            actual_vectors,
            sources,
            args.passes,
            args.batch_size,
            tag="no-memory",
        )
        save_latent_predictor(
            baseline_predictor,
            args.baseline_checkpoint,
            extras={
                "task": "TASK 27 latent predictor - no-memory baseline",
                "pairs_file": str(args.pairs),
                "passes": args.passes,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "context": "zeros(256)",
                "final_eval": baseline_records[-1]["eval"],
            },
        )
        print(f"saved no-memory baseline predictor -> {args.baseline_checkpoint}")

    # ---- paired comparisons on the same examples --------------------------
    # (unpaired means are dominated by example difficulty on this corpus, so every
    #  "does the memory help?" answer below is measured per example, paired)
    print("\n" + "-" * 84)
    print("paired comparison on the same examples (frozen final predictor + memory)")
    print("-" * 84)
    memory_contexts = build_contexts(memory, current_vectors, sources)
    zero_contexts = np.zeros((len(current_vectors), LATENT_DIM), dtype=np.float32)
    predicted_memory = predict_matrix(
        predictor, current_vectors, memory_contexts, args.batch_size
    )
    predicted_zero = predict_matrix(
        predictor, current_vectors, zero_contexts, args.batch_size
    )
    paired_vs_zero = paired_cos_comparison(predicted_memory, predicted_zero, actual_vectors)
    print(
        f"  memory C vs C = 0 (same model)   : cos {paired_vs_zero['mean_cos_a']:.4f} vs "
        f"{paired_vs_zero['mean_cos_b']:.4f} | delta {paired_vs_zero['mean_delta']:+.4f} "
        f"| wins {100 * paired_vs_zero['share_a_wins']:.1f}% "
        f"| t {paired_vs_zero['paired_t_statistic']:+.2f}"
    )

    paired_vs_baseline: Optional[Dict[str, Any]] = None
    if baseline_predictor is not None:
        predicted_baseline = predict_matrix(
            baseline_predictor, current_vectors, zero_contexts, args.batch_size
        )
        paired_vs_baseline = paired_cos_comparison(
            predicted_memory, predicted_baseline, actual_vectors
        )
        print(
            f"  memory model vs no-memory model  : cos "
            f"{paired_vs_baseline['mean_cos_a']:.4f} vs "
            f"{paired_vs_baseline['mean_cos_b']:.4f} | delta "
            f"{paired_vs_baseline['mean_delta']:+.4f} | wins "
            f"{100 * paired_vs_baseline['share_a_wins']:.1f}% "
            f"| t {paired_vs_baseline['paired_t_statistic']:+.2f}"
        )

    # ---- controlled memory-size sweep (fixed examples, growing memory) -----
    print("\n" + "-" * 84)
    print("memory-size sweep: FIXED example set, memory prefixes of growing size")
    print("-" * 84)
    sweep_sizes = sorted(
        {0, 256, 512, 1024, 2048, 4096, memory.size}
        - {size for size in (256, 512, 1024, 2048, 4096) if size > memory.size}
    )
    sweep = memory_size_sweep(
        predictor,
        memory,
        current_vectors,
        actual_vectors,
        sources,
        sweep_sizes,
        args.sweep_sample,
        args.batch_size,
    )


    # ---- comparison + history --------------------------------------------
    first_eval = records[0]["eval"]
    final_eval = records[-1]["eval"]
    baseline_final = baseline_records[-1]["eval"] if baseline_records else None

    comparison: Dict[str, Any] = {
        "with_memory": {
            "pass1_eval_loss": first_eval["mean_loss"],
            "pass1_eval_cos": first_eval["mean_cos"],
            "final_eval_loss": final_eval["mean_loss"],
            "final_eval_cos": final_eval["mean_cos"],
            "final_top1_recall": final_eval.get("top1_recall"),
            "final_top1_recall_excl_self": final_eval.get("top1_recall_excl_self"),
            "memory_size": final_eval["memory_size"],
            "eval_loss_decreased_over_passes": bool(
                final_eval["mean_loss"] < first_eval["mean_loss"]
            ),
            "eval_cos_gain_pass1_to_final": final_eval["mean_cos"] - first_eval["mean_cos"],
        },
        "zero_context_same_predictor": {
            "mean_loss": same_predictor_no_memory["mean_loss"],
            "mean_cos": same_predictor_no_memory["mean_cos"],
        },
        "memory_growth": {
            "buckets": len(growth_curve),
            "first_bucket_loss": growth_curve[0]["mean_loss"],
            "last_bucket_loss": growth_curve[-1]["mean_loss"],
            "first_bucket_cos": growth_curve[0]["mean_cos"],
            "last_bucket_cos": growth_curve[-1]["mean_cos"],
            "loss_improves_with_memory": bool(
                growth_curve[-1]["mean_loss"] < growth_curve[0]["mean_loss"]
            ),
            "memory_size_after_replay": growth_memory_size,
            "note": (
                "unpaired: each bucket is a different, topically ordered run of "
                "examples, so it mixes the memory effect with example difficulty - "
                "use memory_size_sweep for the controlled version"
            ),
        },
        "memory_size_sweep": {
            "sample_size": sweep[0]["num_examples"] if sweep else 0,
            "sizes": [entry["memory_size"] for entry in sweep],
            "losses": [entry["mean_loss"] for entry in sweep],
            "cosines": [entry["mean_cos"] for entry in sweep],
            "paired_deltas_vs_memoryless": [
                entry.get("paired_vs_memoryless", {}).get("mean_delta") for entry in sweep
            ],
            "paired_t_vs_memoryless": [
                entry.get("paired_vs_memoryless", {}).get("paired_t_statistic")
                for entry in sweep
            ],
            "loss_improves_with_memory": bool(
                sweep and sweep[-1]["mean_loss"] < sweep[0]["mean_loss"]
            ),
        },
        "paired_memory_vs_zero_context": paired_vs_zero,
        "memory_helps_vs_zero_context_same_predictor": bool(
            final_eval["mean_cos"] > same_predictor_no_memory["mean_cos"]
        ),
    }
    if paired_vs_baseline is not None:
        comparison["paired_memory_vs_no_memory_predictor"] = paired_vs_baseline
    if baseline_final is not None:
        comparison["no_memory_baseline_predictor"] = {
            "mean_loss": baseline_final["mean_loss"],
            "mean_cos": baseline_final["mean_cos"],
            "passes": [record["train"]["mean_loss"] for record in baseline_records],
        }
        comparison["memory_helps_vs_no_memory_predictor"] = bool(
            final_eval["mean_cos"] > baseline_final["mean_cos"]
        )

    history: Dict[str, Any] = {
        "task": "TASK 27 - latent predictor (prediction in latent space)",
        "config": {
            "vae_checkpoint": str(args.vae),
            "pairs_file": str(args.pairs),
            "passes": args.passes,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "knn_k": args.knn_k,
            "fuse_top_k": args.fuse_top_k,
            "min_memory_entries": MIN_MEMORY_ENTRIES,
            "outcome_floor": OUTCOME_FLOOR,
            "growth_buckets": args.growth_buckets,
            "seed": args.seed,
            "device": str(device),
        },
        "data": {
            "num_pairs": len(pairs),
            "unique_sentences": len(sentences),
            "passes": args.passes,
            "latent_dim": LATENT_DIM,
            "vector_cache": str(args.cache),
        },
        "predictor": {
            "parameter_count": predictor.count_parameters(),
            "description": predictor.describe(),
            "checkpoint": str(args.checkpoint),
            "baseline_checkpoint": None if args.skip_baseline else str(args.baseline_checkpoint),
        },
        "leak_guards": {
            "self_retrieval_excluded": True,
            "recall_reported_with_and_without_self": True,
        },
        "with_memory": {
            "passes": records,
            "growth_curve": growth_curve,
            "memory_size_sweep": sweep,
            "zero_context_eval": same_predictor_no_memory,
            "paired_vs_zero_context": paired_vs_zero,
            "paired_vs_no_memory_predictor": paired_vs_baseline,
        },
        "no_memory_baseline": {
            "passes": baseline_records,
            "final_eval": baseline_final,
        },
        "comparison": comparison,
    }

    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"\nsaved history -> {args.history}")

    print_summary(history)
    return history


if __name__ == "__main__":
    main()

