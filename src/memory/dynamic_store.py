"""TASK 28: dynamic memory with outcome weights, hit counts and a forget policy.

This is the Level-2 substrate of the hierarchical design: the store that actually
holds memories, scores them by how *useful* they turned out to be, and forgets the
entries that were neither accurate nor frequently retrieved.

    vectors      (M, 256)  unit-norm latent positions
    outcomes     (M,)      outcome weight in [0, 1] - how good the prediction that
                           created this memory turned out to be
    hit_counts   (M,)      how often the entry has been retrieved
    timestamps   (M,)      the addition step that created it
    targets      (M, 256)  optional: the "next position" this memory points at
    sources      (M,)      optional: which experience produced it (leak masking)

The task sketch writes these as Python lists; they are contiguous numpy arrays
here instead, because ``add`` must stay O(1) (``np.vstack`` per call copies the
whole store every time) and the forget score has to be vectorised. The observable
behaviour - append, kNN, hit counting, EMA outcome update, forget the bottom 10%
by ``outcome * log(1 + hit_count)`` - is exactly as specified.

Scoring / forgetting
--------------------
``score = outcome_weight * log(1 + hit_count)`` rewards memories that were BOTH
accurate and repeatedly retrieved, and therefore forgets memories that are both
inaccurate and rarely used. ``forget_policy`` removes the bottom
``forget_fraction`` (10%) by score, and is triggered automatically whenever the
store grows past ``max_size`` (removing 10% means ``add`` stays amortised O(1):
forgetting happens roughly once per 10% of capacity).

Index maintenance
-----------------
The FAISS ``IndexFlatIP`` is rebuilt every ``rebuild_every`` additions (1000, as
specified) rather than per addition; ``IndexFlatIP`` is an exact search, so this
only delays the visibility of the newest entries, never the correctness of a
search. A forget pass invalidates the index outright (positions shift), so it is
dropped and rebuilt on the next query.

Positions vs identity
---------------------
``query`` returns ``position`` (the row index in the current arrays), which is the
handle ``update_outcome`` expects. A forget pass COMPACTS the arrays, so every
position handed out before it becomes stale - ``forgotten`` counts removals so a
caller can detect that it happened, and the honest rule is: re-query after a
forget rather than reusing an old position.
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
    raise ImportError("faiss-cpu is required for the dynamic store") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.store import normalize_rows  # noqa: E402

DEFAULT_MAX_SIZE = 100000
DEFAULT_LATENT_DIM = 256
#: TASK 28: rebuild the FAISS index every N additions
INDEX_REBUILD_EVERY = 1000
#: ...but never leave more than this fraction of the store invisible to search.
#: A flat 1000-addition policy is fine for a 100k store (1% stale) and disastrous
#: for a small one: a region of 40 memories would answer every query from its first
#: memory only. Small stores therefore rebuild as they grow (cheap, because they are
#: small), large stores keep the 1000-addition cadence.
STALE_FRACTION = 0.25
#: TASK 28: remove the bottom 10% of entries by score when forgetting
FORGET_FRACTION = 0.1
#: TASK 28: outcome_weight = 0.9 * old + 0.1 * new
EMA_OLD_WEIGHT = 0.9
EMA_NEW_WEIGHT = 0.1
DEFAULT_K = 10
STORE_VERSION = 1

#: TASK 28's formula: ``outcome * log(1 + hit_count)``.
SCORE_MODE_SPEC = "spec"
#: Same, plus a usage prior: ``outcome * (1 + log(1 + hit_count))``.
#:
#: The spec formula scores EVERY entry 0 while nothing has ever been retrieved
#: (``log(1 + 0) = 0``), so a store that has not been queried yet forgets in plain
#: insertion order and an accurate-but-unused memory is evicted exactly like an
#: inaccurate one. The prior mode multiplies by ``1 + log(1 + hits)`` instead, which
#: keeps "useful AND accurate" ordering (0.9/0 hits beats 0.1/0 hits) while a
#: frequently used memory of any accuracy still beats an unused one. Both are
#: available; ``spec`` is the default so TASK 28's behaviour is what you get unless
#: you ask otherwise.
SCORE_MODE_PRIOR = "spec_prior"
SCORE_MODES = (SCORE_MODE_SPEC, SCORE_MODE_PRIOR)


class DynamicMemoryStore:
    """Outcome-weighted kNN memory that forgets its least useful entries."""

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        latent_dim: int = DEFAULT_LATENT_DIM,
        rebuild_every: int = INDEX_REBUILD_EVERY,
        forget_fraction: float = FORGET_FRACTION,
        stale_fraction: float = STALE_FRACTION,
        initial_capacity: int = 1024,
        normalize: bool = True,
        score_mode: str = SCORE_MODE_SPEC,
        name: str = "memory",
    ) -> None:
        if score_mode not in SCORE_MODES:
            raise ValueError(f"unknown score_mode {score_mode!r}; expected {SCORE_MODES}")
        self.max_size = int(max_size)
        self.latent_dim = int(latent_dim)
        self.rebuild_every = int(rebuild_every)
        self.forget_fraction = float(forget_fraction)
        self.stale_fraction = float(stale_fraction)
        self.score_mode = str(score_mode)
        self.normalize = bool(normalize)
        self.name = str(name)

        self.step = 0        # additions attempted (also the clock used for timestamps)
        self.forgotten = 0   # entries removed by forget_policy
        self.forget_events = 0

        self._capacity = max(1, int(initial_capacity))
        self._size = 0
        self._vectors = np.zeros((self._capacity, self.latent_dim), dtype=np.float32)
        self._targets = np.zeros((self._capacity, self.latent_dim), dtype=np.float32)
        self._sources = np.full(self._capacity, -1, dtype=np.int64)
        self._outcomes = np.zeros(self._capacity, dtype=np.float64)
        self._timestamps = np.zeros(self._capacity, dtype=np.int64)
        self._hit_counts = np.zeros(self._capacity, dtype=np.int64)
        self._has_targets = False
        self._has_sources = False

        self._index: Optional[Any] = None
        self._indexed_size = 0
        self._added_since_rebuild = 0
        self.index_rebuilds = 0
        self.last_query_ms = 0.0

    # ------------------------------------------------------------------ views
    @property
    def size(self) -> int:
        """Number of memories currently held."""
        return self._size

    @property
    def vectors(self) -> np.ndarray:
        """``(M, 256)`` unit-norm positions (filled slice)."""
        return self._vectors[: self._size]

    @property
    def outcomes(self) -> np.ndarray:
        """``(M,)`` outcome weights."""
        return self._outcomes[: self._size]

    @property
    def timestamps(self) -> np.ndarray:
        """``(M,)`` addition steps."""
        return self._timestamps[: self._size]

    @property
    def hit_counts(self) -> np.ndarray:
        """``(M,)`` retrieval counts."""
        return self._hit_counts[: self._size]

    @property
    def targets(self) -> np.ndarray:
        """``(M, 256)`` optional target positions (zero rows when unused)."""
        return self._targets[: self._size]

    @property
    def sources(self) -> np.ndarray:
        """``(M,)`` optional source indices (``-1`` when unused)."""
        return self._sources[: self._size]

    @property
    def has_targets(self) -> bool:
        """True once any entry has been added with a target vector."""
        return self._has_targets

    @property
    def has_sources(self) -> bool:
        """True once any entry has been added with a ``source_index``."""
        return self._has_sources

    @property
    def mean_outcome_weight(self) -> float:
        """Mean outcome weight over the live entries."""
        return float(self._outcomes[: self._size].mean()) if self._size else 0.0

    @property
    def total_hits(self) -> int:
        """Sum of the retrieval counts (how much use this store has had)."""
        return int(self._hit_counts[: self._size].sum())

    def __len__(self) -> int:
        return self._size


    # --------------------------------------------------------------- storage
    def _ensure_capacity(self, needed: int) -> None:
        """Grow the backing arrays geometrically (amortised O(1) per addition)."""
        if needed <= self._capacity:
            return
        capacity = self._capacity
        while capacity < needed:
            capacity *= 2
        for name in ("_vectors", "_targets"):
            grown = np.zeros((capacity, self.latent_dim), dtype=np.float32)
            grown[: self._size] = getattr(self, name)[: self._size]
            setattr(self, name, grown)
        for name, dtype, fill in (
            ("_outcomes", np.float64, 0.0),
            ("_timestamps", np.int64, 0),
            ("_hit_counts", np.int64, 0),
            ("_sources", np.int64, -1),
        ):
            grown = np.full(capacity, fill, dtype=dtype)
            grown[: self._size] = getattr(self, name)[: self._size]
            setattr(self, name, grown)
        self._capacity = capacity

    def add(
        self,
        position_vector: np.ndarray,
        outcome_weight: float,
        target_vector: Optional[np.ndarray] = None,
        source_index: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> int:
        """Append one memory; returns its position (the handle ``update_outcome`` takes).

        ``position_vector`` is L2-normalised on the way in (inner product == cosine,
        as everywhere else in this project). ``outcome_weight`` is clamped to
        ``[0, 1]``. ``target_vector``/``source_index`` are optional extras the V4
        prediction loop stores alongside a position (where the memory points, and
        which experience produced it - the latter is what lets a replayed example be
        masked out of its own retrieval).

        When the store grows past ``max_size`` the forget policy fires immediately,
        exactly as the task specifies.
        """
        self._ensure_capacity(self._size + 1)
        normalised = normalize_rows(
            np.asarray(position_vector, dtype=np.float32).reshape(1, -1)
        )
        if normalised.shape[1] != self.latent_dim:
            raise ValueError(
                f"position_vector has dim {normalised.shape[1]}, expected {self.latent_dim}"
            )
        self._vectors[self._size] = normalised[0]
        if target_vector is not None:
            self._targets[self._size] = normalize_rows(
                np.asarray(target_vector, dtype=np.float32).reshape(1, -1)
            )[0]
            self._has_targets = True
        self._sources[self._size] = -1 if source_index is None else int(source_index)
        if source_index is not None:
            self._has_sources = True
        self._outcomes[self._size] = float(np.clip(outcome_weight, 0.0, 1.0))
        self._timestamps[self._size] = (
            int(self.step) if timestamp is None else int(timestamp)
        )
        self._hit_counts[self._size] = 0

        position = self._size
        self._size += 1
        self.step += 1
        self._added_since_rebuild += 1

        if self._size > self.max_size:
            self.forget_policy()
        self._maybe_rebuild_index()
        return position

    def _maybe_rebuild_index(self, for_query: bool = False) -> None:
        """Rebuild the FAISS index every ``rebuild_every`` additions (or when stale).

        ``for_query`` distinguishes *adding* from *searching*:

          * ``add`` only honours the ``rebuild_every`` cadence (1000, per TASK 28),
            so appending stays cheap however stale the index is;
          * ``query`` additionally honours ``stale_fraction``, so a caller that sets
            it to 0 (every region store does) gets an exactly current index on every
            search while pays nothing for it during insertion.

        ``IndexFlatIP`` is an exact search, so an unindexed entry is invisible, not
        approximate - hence the second trigger, which is what keeps a 40-memory
        region searchable while a 100k store still only rebuilds every 1000 adds.
        """
        if self._index is not None:
            interval_due = self._added_since_rebuild >= self.rebuild_every
            stale_due = for_query and (self._size - self._indexed_size) > int(
                self.stale_fraction * self._size
            )
            if not (interval_due or stale_due):
                return
        index = faiss.IndexFlatIP(self.latent_dim)
        if self._size:
            index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
        self._index = index
        self._indexed_size = self._size
        self._added_since_rebuild = 0
        self.index_rebuilds += 1


    # -------------------------------------------------------------- retrieval
    def refresh_index(self) -> int:
        """Force a full FAISS rebuild so the next search sees every entry.

        ``add`` only rebuilds on the ``rebuild_every`` cadence and ``query`` only when
        more than ``stale_fraction`` of the store is unindexed, so a store can be
        intentionally slightly behind. Anything that claims *exact* retrieval (the
        ground truth in a comparison, or a final evaluation) should call this first -
        the invariant test in ``scripts/test_region_store.py`` is what surfaced the
        need: the flat reference store was quietly missing its newest 171 memories,
        which made the hierarchy look wrong when it was the reference that was stale.
        """
        self._index = None
        self._indexed_size = 0
        self._added_since_rebuild = self.rebuild_every
        self._maybe_rebuild_index()
        return self._indexed_size

    def scores(self) -> np.ndarray:
        """Per-entry keep/forget score.

        ``score_mode="spec"``  -> ``outcome_weight * log(1 + hit_count)``  (TASK 28)
        ``score_mode="spec_prior"`` -> ``outcome_weight * (1 + log(1 + hit_count))``

        Multiplicative on purpose: an entry scores high only when it was both
        accurate (high outcome) and actually used (hit count > 0). The ``1 +`` in the
        prior mode is what stops a never-queried store from scoring everything 0,
        where forgetting would silently degrade to "evict in insertion order".
        """
        if not self._size:
            return np.zeros(0, dtype=np.float64)
        usage = np.log1p(self.hit_counts.astype(np.float64))
        if self.score_mode == SCORE_MODE_PRIOR:
            usage = 1.0 + usage
        return self.outcomes * usage

    def query(
        self,
        vector: np.ndarray,
        k: int = DEFAULT_K,
        normalize: Optional[bool] = None,
        update_hits: bool = True,
        exclude_source: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Top-``k`` memories by cosine similarity; increments their hit counts.

        One record per neighbour: ``position``, ``similarity``, ``vector``,
        ``outcome_weight``, ``hit_count``, ``timestamp`` (plus ``target``/``source``
        when the store carries them). ``hit_count`` in the record is the count as it
        stood *before* this retrieval (the value the score used); the stored counter
        for each returned entry is then incremented. ``position`` is what
        ``update_outcome`` wants, and it goes stale after a forget pass.

        ``exclude_source`` drops every entry whose ``source`` equals it, which is how
        a replayed example is stopped from retrieving itself (TASK 29's loop, and the
        TASK 27 measurement that showed it matters). ``IndexFlatIP`` has no metadata
        filtering, so the search asks for a few extra rows and filters afterwards;
        the number of excluded rows is bounded by how often one source was added.
        """
        if self._size == 0:
            return []
        wanted = max(1, int(k))
        # one excluded row can occupy one of the nearest slots, so over-fetch and filter
        extra = max(1, wanted // 2) if exclude_source is not None else 0
        search_k = min(wanted + extra, self._size)
        self._maybe_rebuild_index(for_query=True)

        should_normalize = self.normalize if normalize is None else bool(normalize)
        query_vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query_vector.shape[1] != self.latent_dim:
            raise ValueError(
                f"query vector has dim {query_vector.shape[1]}, expected {self.latent_dim}"
            )
        if should_normalize:
            query_vector = normalize_rows(query_vector)

        started = time.perf_counter()
        similarities, positions = self._index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32), search_k
        )
        self.last_query_ms = (time.perf_counter() - started) * 1000.0

        records: List[Dict[str, Any]] = []
        for similarity, position in zip(similarities[0], positions[0]):
            if position < 0:  # FAISS pads with -1 when fewer than k rows exist
                continue
            position = int(position)
            if exclude_source is not None and int(self._sources[position]) == int(exclude_source):
                continue
            records.append(self._record(position, float(similarity)))
            if len(records) >= wanted:
                break
        if update_hits and records:
            self._hit_counts[[record["position"] for record in records]] += 1
        return records

    def _record(self, position: int, similarity: float) -> Dict[str, Any]:
        """One neighbour record (shared by ``query`` and ``most_useful_k``)."""
        record: Dict[str, Any] = {
            "position": int(position),
            "similarity": float(similarity),
            "vector": self._vectors[position],
            "outcome_weight": float(self._outcomes[position]),
            "hit_count": int(self._hit_counts[position]),
            "timestamp": int(self._timestamps[position]),
        }
        # each optional field is exposed independently of the other
        if self._has_targets:
            record["target"] = self._targets[position]
        if self._has_sources:
            record["source"] = int(self._sources[position])
        return record

    def update_outcome(self, idx: int, new_outcome: float) -> float:
        """EMA-update the outcome weight of entry ``idx`` -> the new value.

        ``new_outcome`` is the freshly measured quality of that memory (in the V4
        loop, the cosine similarity between the prediction it guided and what
        actually happened), so a memory that keeps helping drifts towards 1 and one
        that keeps misleading drifts towards 0.
        """
        position = int(idx)
        if not 0 <= position < self._size:
            raise IndexError(f"position {position} out of range for size {self._size}")
        old = float(self._outcomes[position])
        new = float(np.clip(new_outcome, 0.0, 1.0))
        updated = EMA_OLD_WEIGHT * old + EMA_NEW_WEIGHT * new
        self._outcomes[position] = updated
        return updated

    def most_useful_k(self, k: int = 5) -> List[Dict[str, Any]]:
        """The ``k`` entries with the highest score (accuracy x usage)."""
        if not self._size:
            return []
        scores = self.scores()
        order = np.argsort(-scores, kind="stable")[: max(1, int(k))]
        records: List[Dict[str, Any]] = []
        for raw_position in order:
            position = int(raw_position)
            record = self._record(position, similarity=float("nan"))
            record["score"] = float(scores[position])
            records.append(record)
        return records


    # ------------------------------------------------------------- forgetting
    def forget_policy(self, keep_fraction: Optional[float] = None) -> Dict[str, Any]:
        """Drop the lowest-scoring entries (default: the bottom 10%).

        ``score = outcome_weight * log(1 + hit_count)``, so this sheds entries that
        are inaccurate AND rarely retrieved while keeping the accurate or heavily
        used ones. Ties are broken by position (stable sort), so forgetting is
        deterministic - important, because a non-deterministic memory would make
        every downstream number irreproducible.

        The FAISS index is dropped (positions shift) and rebuilt by the next query,
        and the backing capacity is released so the footprint tracks the live size.
        """
        if not self._size:
            return {"removed": 0, "kept": 0, "score_threshold": 0.0}
        fraction = self.forget_fraction if keep_fraction is None else float(keep_fraction)
        fraction = min(max(fraction, 0.0), 1.0)
        remove_count = max(1, int(round(fraction * self._size)))
        if remove_count >= self._size:
            remove_count = self._size - 1  # never drop the whole store

        scores = self.scores()
        order = np.argsort(scores, kind="stable")  # lowest score first
        remove = order[:remove_count]
        keep = order[remove_count:]
        threshold = float(scores[remove[-1]])

        self._vectors[: keep.size] = self._vectors[keep]
        self._targets[: keep.size] = self._targets[keep]
        self._sources[: keep.size] = self._sources[keep]
        self._outcomes[: keep.size] = self._outcomes[keep]
        self._timestamps[: keep.size] = self._timestamps[keep]
        self._hit_counts[: keep.size] = self._hit_counts[keep]
        self._size = int(keep.size)
        self.forgotten += int(remove_count)
        self.forget_events += 1
        self._index = None
        self._indexed_size = 0
        self._added_since_rebuild = self.rebuild_every  # force a rebuild

        if self._capacity > max(1024, 4 * self._size):  # release unused capacity
            self._capacity = max(1, 2 * self._size)
            for name in (
                "_vectors",
                "_targets",
                "_sources",
                "_outcomes",
                "_timestamps",
                "_hit_counts",
            ):
                setattr(
                    self, name, np.ascontiguousarray(getattr(self, name)[: self._capacity])
                )

        return {
            "removed": int(remove_count),
            "kept": int(keep.size),
            "score_threshold": threshold,
            "mean_score_removed": float(scores[remove].mean()),
            "mean_score_kept": float(scores[keep].mean()) if keep.size else 0.0,
        }

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, Any]:
        """Small summary used by the tests, logs and the DONE block."""
        return {
            "name": self.name,
            "size": self._size,
            "max_size": self.max_size,
            "mean_outcome_weight": self.mean_outcome_weight,
            "total_hits": self.total_hits,
            "max_hit_count": (
                int(self._hit_counts[: self._size].max()) if self._size else 0
            ),
            "forgotten": self.forgotten,
            "forget_events": self.forget_events,
            "step": self.step,
            "index_rebuilds": self.index_rebuilds,
            "mean_score": float(self.scores().mean()) if self._size else 0.0,
        }


    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        """Pickle the arrays + counters (the FAISS index is rebuilt on load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_VERSION,
            "config": {
                "max_size": self.max_size,
                "latent_dim": self.latent_dim,
                "rebuild_every": self.rebuild_every,
                "forget_fraction": self.forget_fraction,
                "stale_fraction": self.stale_fraction,
                "score_mode": self.score_mode,
                "normalize": self.normalize,
                "name": self.name,
            },
            "step": self.step,
            "forgotten": self.forgotten,
            "forget_events": self.forget_events,
            "has_targets": self._has_targets,
            "has_sources": self._has_sources,
            "vectors": self.vectors,
            "targets": self.targets,
            "sources": self.sources,
            "outcomes": self.outcomes,
            "timestamps": self.timestamps,
            "hit_counts": self.hit_counts,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> "DynamicMemoryStore":
        """Reload a store written by :meth:`save` (FAISS index rebuilt lazily)."""
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        config = payload["config"]
        vectors = payload["vectors"]
        store = cls(
            max_size=config["max_size"],
            latent_dim=config["latent_dim"],
            rebuild_every=config["rebuild_every"],
            forget_fraction=config["forget_fraction"],
            stale_fraction=config.get("stale_fraction", STALE_FRACTION),
            score_mode=config.get("score_mode", SCORE_MODE_SPEC),
            initial_capacity=max(1024, 2 * len(vectors) + 1),
            normalize=config["normalize"],
            name=config["name"],
        )
        count = int(vectors.shape[0])
        store._size = count
        store._vectors[:count] = vectors
        store._targets[:count] = payload["targets"]
        store._sources[:count] = payload["sources"]
        store._outcomes[:count] = payload["outcomes"]
        store._timestamps[:count] = payload["timestamps"]
        store._hit_counts[:count] = payload["hit_counts"]
        store._has_targets = bool(payload.get("has_targets", False))
        store._has_sources = bool(payload.get("has_sources", False))
        store.step = int(payload["step"])
        store.forgotten = int(payload["forgotten"])
        store.forget_events = int(payload.get("forget_events", 0))
        store._added_since_rebuild = store.rebuild_every  # index rebuilt on first query
        return store

