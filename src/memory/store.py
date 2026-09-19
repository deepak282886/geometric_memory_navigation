"""Memory store: a FAISS inner-product index over VAE latents of training ctxs.

Each memory holds the information the rest of the GMN pipeline needs:

    vector         256-d latent of the training ``ctx`` (L2-normalised)
    label          index (0-3) of the correct ending
    activity_label e.g. "Removing ice from car"
    endings        the 4 candidate endings
    ctx            the raw context (debugging / inspection)
    idx            position of the example in data/train.json

All vectors are L2-normalised before being added to the index and query vectors
are normalised the same way, so FAISS inner product is exactly cosine similarity.

Outcome weights
---------------
At *query* (inference) time the true label of the query is unknown, so every
neighbour is returned with ``outcome_weight = 1.0``. The supervised weighting
(1.0 for a label match, 0.1 otherwise) is exposed through
``get_training_outcome_weight`` and is used only when training the classifier
(TASK 7).

The FAISS index is not pickled; ``save`` writes the vectors plus metadata and
``load`` rebuilds an identical ``IndexFlatIP`` from them, so the store can be
reopened without re-running the VAE.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("faiss-cpu is required for the memory store: pip install faiss-cpu") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUM_ENDINGS = 4

#: weight used at inference time (query label unknown)
INFERENCE_OUTCOME_WEIGHT = 1.0
#: weight used during classifier training when the label matches / differs
MATCH_OUTCOME_WEIGHT = 1.0
MISMATCH_OUTCOME_WEIGHT = 0.1

DEFAULT_K = 10
STORE_VERSION = 1


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row so that inner product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class MemoryStore:
    """kNN memory over training contexts, backed by ``faiss.IndexFlatIP``."""

    def __init__(
        self,
        vae: Any,
        train_data: Sequence[Dict[str, Any]],
        normalize: bool = True,
    ) -> None:
        """Encode every training ``ctx`` and build the index."""
        self.normalize = normalize
        self.source = "vae-encoded train.json"

        contexts = [str(example["ctx"]) for example in train_data]
        vectors = vae.encode_texts(contexts)
        vectors = vectors.detach().cpu().numpy()

        # The known-correct continuation of every memory. Navigation targets these
        # rather than the neighbouring contexts (STEP 2 / TASK "continuation").
        correct_endings = [
            str(example["endings"][int(example["label"])]) for example in train_data
        ]
        correct_ending_vectors = (
            vae.encode_texts(correct_endings).detach().cpu().numpy()
        )

        self._init_arrays(
            vectors=vectors,
            labels=[int(example["label"]) for example in train_data],
            activity_labels=[str(example["activity_label"]) for example in train_data],
            endings=[list(example["endings"]) for example in train_data],
            ctxs=contexts,
            correct_ending_vectors=correct_ending_vectors,
        )
        self._build_index()
        print(f"Memory store built with {self.size} vectors")

    # ------------------------------------------------------------------ setup
    def _init_arrays(
        self,
        vectors: np.ndarray,
        labels: Sequence[int],
        activity_labels: Sequence[str],
        endings: Sequence[Sequence[str]],
        ctxs: Sequence[str],
        indices: Optional[Sequence[int]] = None,
        correct_ending_vectors: Optional[np.ndarray] = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-d, got shape {vectors.shape}")
        if self.normalize:
            vectors = normalize_rows(vectors)

        count = vectors.shape[0]
        for name, values in (
            ("labels", labels),
            ("activity_labels", activity_labels),
            ("endings", endings),
            ("ctxs", ctxs),
        ):
            if len(values) != count:
                raise ValueError(f"{name} has {len(values)} entries but {count} vectors")
        for ending_list in endings:
            if len(ending_list) != NUM_ENDINGS:
                raise ValueError(f"expected {NUM_ENDINGS} endings, got {len(ending_list)}")

        if correct_ending_vectors is not None:
            correct_ending_vectors = np.asarray(correct_ending_vectors, dtype=np.float32)
            if correct_ending_vectors.shape[0] != count:
                raise ValueError(
                    f"correct_ending_vectors has {correct_ending_vectors.shape[0]} rows "
                    f"but {count} vectors were supplied"
                )
            if self.normalize:
                correct_ending_vectors = normalize_rows(correct_ending_vectors)

        self.vectors = vectors
        self.dim = int(vectors.shape[1])
        self.labels = np.asarray(labels, dtype=np.int64)
        self.activity_labels: List[str] = list(activity_labels)
        self.endings: List[List[str]] = [list(item) for item in endings]
        self.ctxs: List[str] = list(ctxs)
        self.indices: List[int] = (
            list(range(count)) if indices is None else [int(i) for i in indices]
        )
        #: latent of each memory's KNOWN-CORRECT continuation. Navigation points
        #: here rather than at the neighbouring context (see src/sampler/jev.py).
        self.correct_ending_vectors: Optional[np.ndarray] = correct_ending_vectors
        self.last_query_ms: float = 0.0

    @property
    def has_correct_ending_vectors(self) -> bool:
        return self.correct_ending_vectors is not None

    @property
    def default_target(self) -> str:
        """``"correct_ending"`` when this store carries them, else ``"context"``.

        Lets the high-level ``score_example`` work with either store generation
        without silently pretending to use a target the store does not have.
        """
        return "correct_ending" if self.has_correct_ending_vectors else "context"

    def _build_index(self) -> None:
        # Inner product on unit vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))

    # --------------------------------------------------------------- dunders
    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def size(self) -> int:
        return len(self)

    def __getitem__(self, position: int) -> Dict[str, Any]:
        return self._make_record(position, distance=None, outcome_weight=INFERENCE_OUTCOME_WEIGHT)

    def __repr__(self) -> str:
        return (
            f"MemoryStore(size={self.size}, dim={self.dim}, "
            f"index={type(self.index).__name__}, normalize={self.normalize})"
        )

    # --------------------------------------------------------------- helpers
    def _make_record(
        self,
        position: int,
        distance: Optional[float],
        outcome_weight: float,
    ) -> Dict[str, Any]:
        return {
            "vector": self.vectors[position].copy(),
            "correct_ending_vector": (
                None
                if self.correct_ending_vectors is None
                else self.correct_ending_vectors[position].copy()
            ),
            "label": int(self.labels[position]),
            "activity_label": self.activity_labels[position],
            "endings": list(self.endings[position]),
            "ctx": self.ctxs[position],
            "idx": int(self.indices[position]),
            "distance": None if distance is None else float(distance),
            "outcome_weight": float(outcome_weight),
        }

    # ------------------------------------------------------- outcome weights
    def get_training_outcome_weight(self, memory_label: int, query_label: int) -> float:
        """Training-only weight: 1.0 if the labels match, else 0.1.

        At inference the query label is unknown, so ``query`` returns 1.0 for
        every neighbour instead. Used by the classifier training loop (TASK 7).
        """
        return (
            MATCH_OUTCOME_WEIGHT
            if int(memory_label) == int(query_label)
            else MISMATCH_OUTCOME_WEIGHT
        )

    # TASK 4 spec name for the same behaviour.
    get_outcome_weight = get_training_outcome_weight

    # ------------------------------------------------------------------- query
    def query(
        self,
        vector: np.ndarray,
        k: int = DEFAULT_K,
        normalize: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Top-``k`` memories for a 256-d query vector, sorted by cosine similarity."""
        if self.size == 0:
            return []
        k = max(1, min(int(k), self.size))

        query_vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"query vector has dim {query_vector.shape[1]}, expected {self.dim}"
            )
        should_normalize = self.normalize if normalize is None else bool(normalize)
        if should_normalize:
            # Normalise the query too, otherwise inner product != cosine similarity.
            query_vector = normalize_rows(query_vector)

        started = time.perf_counter()
        distances, positions = self.index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32), k
        )
        self.last_query_ms = (time.perf_counter() - started) * 1000.0

        results: List[Dict[str, Any]] = []
        for distance, position in zip(distances[0], positions[0]):
            if position < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            results.append(
                self._make_record(int(position), distance, INFERENCE_OUTCOME_WEIGHT)
            )
        return results

    def query_excluding(
        self,
        vector: np.ndarray,
        exclude_idx: int,
        k: int = DEFAULT_K,
        normalize: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Top-``k`` memories, skipping the memory whose ``idx`` is ``exclude_idx``.

        Training examples live in the store themselves, so a plain ``query``
        would return the example as its own nearest neighbour (similarity 1.0)
        and leak the answer. We search ``k + 1`` rows and drop the excluded one;
        if it was not among the results the first ``k`` are returned unchanged.
        """
        if self.size == 0:
            return []
        exclude_idx = int(exclude_idx)
        wanted = max(1, int(k))
        # one extra row so that removing the excluded memory still leaves k
        search_k = min(self.size, wanted + 1)
        results = self.query(vector, k=search_k, normalize=normalize)
        filtered = [record for record in results if record["idx"] != exclude_idx]
        # if exclude_idx was absent, filtered == results and this is the first k
        return filtered[:wanted]
        """Average single-vector query latency in milliseconds."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        timings: List[float] = []
        for index in range(repeats):
            self.query(vectors[index % len(vectors)], k=k)
            timings.append(self.last_query_ms)
        values = np.asarray(timings)
        return {
            "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "repeats": int(repeats),
        }

    # ---------------------------------------------------------- persistence
    def save(self, path: Path) -> Path:
        """Pickle vectors + metadata (the FAISS index is rebuilt on load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_VERSION,
            "dim": self.dim,
            "normalize": self.normalize,
            "source": self.source,
            "vectors": self.vectors,
            "correct_ending_vectors": self.correct_ending_vectors,
            "labels": self.labels,
            "activity_labels": self.activity_labels,
            "endings": self.endings,
            "ctxs": self.ctxs,
            "indices": self.indices,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Memory store saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "MemoryStore":
        """Rebuild a store from a pickle written by :meth:`save` (no VAE needed)."""
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        store = cls.__new__(cls)
        store.normalize = bool(payload.get("normalize", True))
        store.source = str(payload.get("source", path))
        store._init_arrays(
            vectors=payload["vectors"],
            labels=payload["labels"],
            activity_labels=payload["activity_labels"],
            endings=payload["endings"],
            ctxs=payload["ctxs"],
            indices=payload.get("indices"),
            correct_ending_vectors=payload.get("correct_ending_vectors"),
        )
        store._build_index()
        print(f"Memory store loaded from {path} ({store.size} vectors)")
        return store

    @classmethod
    def from_vectors(
        cls,
        vectors: np.ndarray,
        labels: Sequence[int],
        activity_labels: Sequence[str],
        endings: Sequence[Sequence[str]],
        ctxs: Sequence[str],
        indices: Optional[Sequence[int]] = None,
        normalize: bool = True,
        source: str = "precomputed vectors",
        correct_ending_vectors: Optional[np.ndarray] = None,
    ) -> "MemoryStore":
        """Build a store from already-computed latents, skipping VAE encoding.

        Used for the full 39,905-example store, which would otherwise be encoded
        twice (once to time the encoding, once inside ``__init__``), and for the
        "v2" store that reuses the existing context vectors so the only thing
        that changes is the navigation target.
        """
        store = cls.__new__(cls)
        store.normalize = normalize
        store.source = source
        store._init_arrays(
            vectors=vectors,
            labels=labels,
            activity_labels=activity_labels,
            endings=endings,
            ctxs=ctxs,
            indices=indices,
            correct_ending_vectors=correct_ending_vectors,
        )
        store._build_index()
        print(f"Memory store built with {store.size} vectors")
        return store

    # --------------------------------------------------------------- reports
    def label_counts(self) -> Dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    def summary(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "vector_shape": tuple(self.vectors.shape),
            "dim": self.dim,
            "index_type": type(self.index).__name__,
            "normalized": self.normalize,
            "unique_activity_labels": len(set(self.activity_labels)),
            "label_counts": self.label_counts(),
            "distinct_ctxs": len(set(self.ctxs)),
        }
