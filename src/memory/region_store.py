"""Two-level latent memory: REGIONS (centroids) over MEMORIES (outcome-weighted).

The design this implements
--------------------------
    Level 1  ~100-500 region centroids covering neighbourhoods of the space
    Level 2  memories, each belonging to its nearest region
    Query    1. find the nearest region to the query (~100-500 comparisons)
             2. search the memories inside that region (dense, relevant)
             3. regions with more visits become more detailed (more memories)
             4. rarely visited regions stay sparse (few memories)

Why two levels
--------------
* **Cost**: retrieval is ``num_regions`` centroid comparisons plus a dense search
  over one (small) region, instead of a flat scan over the whole memory. At 100k
  memories / 500 regions that is ~1/200th of the distance computations.
* **Locality / relevance**: region membership is a coarse semantic filter, so the
  neighbours returned are denser (topically closer) for the same ``k``. The TASK 27
  result is the motivation: retrieved neighbours only help when they are *relevant*
  (``+0.0081`` at 38k memories vs ``-0.0153`` when the memory was unrelated), so
  retrieving from the right neighbourhood matters more than retrieving more.
* **Budget follows use**: each region's retention capacity tracks its share of
  visits, so the space spends detail where queries actually go.

How it is built
---------------
* **Centroids**: ``faiss.Kmeans`` (already a project dependency) over L2-normalised
  vectors. On unit vectors L2 and cosine rank identically
  (``||a - b||^2 = 2 - 2 cos``), so L2 k-means assignment IS spherical k-means
  assignment; the centroids are re-normalised so region scoring is a plain inner
  product. Refitting is available (``fit_regions`` again) and tracked
  (``region_refits``).
* **Memories per region**: each region owns a :class:`DynamicMemoryStore` (TASK 28
  rules: outcome weights, hit counts, EMA updates, forget policy), so forgetting is
  *local* - a region that is under pressure sheds its own least useful memories
  rather than competing with the whole store.
* **Warm-up**: before there is enough data to cluster, memories go into a pending
  store. Once ``auto_fit_after`` memories have arrived (default 2048), regions are
  fitted and every pending memory is routed to its nearest region. That keeps the
  "regions emerge from experience" story honest rather than pre-seeding regions from
  a training corpus.
* **Spillover**: the query keeps probing the next-nearest region while it still has
  fewer than ``k`` memories, so a sparse or rarely visited region still answers
  (this is exactly the case "rarely visited regions stay sparse" creates - a strict
  one-region lookup would return nothing).
* **Detail growth**: ``region_capacity = clip(base * (visits_r / mean_visits),
  base/8, base*8)`` with ``base = max_size / num_regions``. Queries refresh the
  capacities every ``capacity_refresh_every`` queries.

Records returned by :meth:`query` carry the TASK 28 fields plus ``region`` and
``region_similarity``, so a caller can see which neighbourhood answered.
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
    raise ImportError("faiss-cpu is required for the region store") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.dynamic_store import (  # noqa: E402
    FORGET_FRACTION,
    INDEX_REBUILD_EVERY,
    SCORE_MODE_PRIOR,
    DynamicMemoryStore,
)
from src.memory.store import normalize_rows  # noqa: E402

#: Level 1 size: the design asks for ~100-500 regions
DEFAULT_NUM_REGIONS = 256
DEFAULT_LATENT_DIM = 256
DEFAULT_MAX_SIZE = 100000
#: preallocation per region: small, because a region holds few memories and each
#: ``DynamicMemoryStore`` would otherwise reserve 1024 x 256 x 4 B = 1 MB, i.e. 256 MB
#: for 256 regions (500 MB at 500 regions) before a single memory is stored
MIN_REGION_PREALLOC = 64
#: regions keep their FAISS index EXACTLY current (no staleness allowance). A region
#: is small by construction, so re-indexing it costs microseconds, and the hierarchy
#: only preserves the flat top-k if the search inside a region is exact - a 25%
#: staleness window (fine for the 100k flat store) silently drops neighbours *within*
#: a region, which is exactly what test_region_store.py's invariant check catches.
REGION_STALE_FRACTION = 0.0
#: memories collected before regions are fitted (warm-up)
DEFAULT_AUTO_FIT_AFTER = 2048
KMEANS_ITERATIONS = 25
#: per-region forgetting uses the usage-prior score, NOT the bare spec formula.
#: Measured in scripts/test_dynamic_memory.py: ``outcome * log(1 + hits)`` scores
#: every entry 0 until something has been retrieved, so the first overflow evicts in
#: insertion order (500 good entries inserted first -> 0 survivors, 360 bad survivors).
#: A region store is exactly that situation - small, freshly created, few hits - so
#: every region would shed its earliest memories first. ``spec_prior`` keeps the
#: "accurate AND used" ordering (500/500 good vs 0/500 bad survivors on the same test).
DEFAULT_SCORE_MODE = SCORE_MODE_PRIOR
#: how far a region's capacity may drift from the uniform budget
MIN_CAPACITY_FACTOR = 0.125
MAX_CAPACITY_FACTOR = 8.0
#: refresh per-region capacities every N queries
CAPACITY_REFRESH_EVERY = 64
STORE_VERSION = 1


class RegionStore:
    """Hierarchical memory: nearest-region routing over per-region dynamic stores."""

    def __init__(
        self,
        num_regions: int = DEFAULT_NUM_REGIONS,
        latent_dim: int = DEFAULT_LATENT_DIM,
        max_size: int = DEFAULT_MAX_SIZE,
        auto_fit_after: int = DEFAULT_AUTO_FIT_AFTER,
        kmeans_iterations: int = KMEANS_ITERATIONS,
        rebuild_every: int = INDEX_REBUILD_EVERY,
        forget_fraction: float = FORGET_FRACTION,
        capacity_refresh_every: int = CAPACITY_REFRESH_EVERY,
        min_capacity_factor: float = MIN_CAPACITY_FACTOR,
        max_capacity_factor: float = MAX_CAPACITY_FACTOR,
        score_mode: str = DEFAULT_SCORE_MODE,
        seed: int = 0,
        name: str = "regions",
    ) -> None:
        self.num_regions = int(num_regions)
        self.latent_dim = int(latent_dim)
        self.max_size = int(max_size)
        self.auto_fit_after = int(auto_fit_after)
        self.kmeans_iterations = int(kmeans_iterations)
        self.rebuild_every = int(rebuild_every)
        self.forget_fraction = float(forget_fraction)
        self.capacity_refresh_every = int(capacity_refresh_every)
        self.min_capacity_factor = float(min_capacity_factor)
        self.max_capacity_factor = float(max_capacity_factor)
        self.score_mode = str(score_mode)
        self.seed = int(seed)
        self.name = str(name)

        #: Level 1 - centroids (None until fitted)
        self.centroids: Optional[np.ndarray] = None
        #: Level 2 - one dynamic store per region (created at fit time)
        self.regions: List[DynamicMemoryStore] = []
        #: warm-up buffer, used before the regions exist
        self.pending = DynamicMemoryStore(
            max_size=max(self.auto_fit_after, 1),
            latent_dim=self.latent_dim,
            rebuild_every=max(1, min(self.rebuild_every, 256)),
            forget_fraction=self.forget_fraction,
            score_mode=self.score_mode,
            initial_capacity=min(self.auto_fit_after, 1024),
            name=f"{self.name}:pending",
        )

        self._visits = np.zeros(self.num_regions, dtype=np.int64)
        self._region_hits = np.zeros(self.num_regions, dtype=np.int64)
        self._region_forgotten = np.zeros(self.num_regions, dtype=np.int64)
        self._capacities = np.zeros(self.num_regions, dtype=np.int64)
        self.region_queries = 0
        self.total_probes = 0
        self.region_refits = 0
        self.fit_steps: List[int] = []
        self.last_query_ms = 0.0
        self.last_probe_ms = 0.0

    # ------------------------------------------------------------------ views
    @property
    def is_fitted(self) -> bool:
        """True once the region centroids exist."""
        return self.centroids is not None

    @property
    def size(self) -> int:
        """Total memories held (pending + routed)."""
        return int(self.pending.size + sum(region.size for region in self.regions))

    @property
    def routed_size(self) -> int:
        """Memories that live in a region (excludes the warm-up buffer)."""
        return int(sum(region.size for region in self.regions))

    @property
    def mean_outcome_weight(self) -> float:
        """Mean outcome weight over every live memory."""
        total = 0.0
        count = 0
        for store in ([self.pending] if self.pending.size else []) + self.regions:
            if store.size:
                total += float(store.outcomes.sum())
                count += store.size
        return total / count if count else 0.0

    @property
    def visits(self) -> np.ndarray:
        """``(R,)`` queries whose nearest region was ``r``."""
        return self._visits.copy()

    @property
    def region_hits(self) -> np.ndarray:
        """``(R,)`` retrievals actually served from region ``r``."""
        return self._region_hits.copy()

    @property
    def capacities(self) -> np.ndarray:
        """``(R,)`` current per-region retention capacity."""
        return self._capacities.copy()

    @property
    def region_sizes(self) -> np.ndarray:
        """``(R,)`` live memory count per region."""
        return np.asarray([region.size for region in self.regions], dtype=np.int64)

    def base_capacity(self) -> int:
        """Uniform per-region budget: ``max_size / num_regions``."""
        return max(1, int(self.max_size // max(1, self.num_regions)))


    # ------------------------------------------------------------- Level 1 fit
    def fit_regions(
        self,
        vectors: Optional[np.ndarray] = None,
        iterations: Optional[int] = None,
        reroute_pending: bool = True,
    ) -> Dict[str, Any]:
        """Fit ``num_regions`` centroids (faiss k-means) and route the memories.

        ``vectors`` defaults to the warm-up buffer's content, so the normal call is
        just ``fit_regions()``. Because the vectors are unit-norm, L2 k-means
        assignment is spherical k-means assignment (``||a-b||^2 = 2 - 2cos``); the
        k-means gets ``seed``, so a refit is reproducible.
        """
        source = (
            self.pending.vectors if vectors is None else np.asarray(vectors, dtype=np.float32)
        )
        if source.ndim != 2 or source.shape[0] < self.num_regions:
            raise ValueError(
                f"need at least {self.num_regions} vectors to fit regions, got "
                f"{source.shape[0] if source.ndim == 2 else 0}"
            )
        iterations = self.kmeans_iterations if iterations is None else int(iterations)
        kmeans = faiss.Kmeans(
            self.latent_dim,
            self.num_regions,
            niter=iterations,
            verbose=False,
            seed=self.seed,
        )
        kmeans.train(np.ascontiguousarray(source, dtype=np.float32))
        self.centroids = normalize_rows(np.asarray(kmeans.centroids, dtype=np.float32))
        region_prealloc = max(
            MIN_REGION_PREALLOC, min(1024, self.base_capacity())
        )
        self.regions = [
            DynamicMemoryStore(
                max_size=self.base_capacity(),
                latent_dim=self.latent_dim,
                rebuild_every=self.rebuild_every,
                forget_fraction=self.forget_fraction,
                score_mode=self.score_mode,
                stale_fraction=REGION_STALE_FRACTION,
                initial_capacity=region_prealloc,
                name=f"{self.name}:r{index}",
            )
            for index in range(self.num_regions)
        ]
        self._region_forgotten = np.zeros(self.num_regions, dtype=np.int64)
        self.region_refits += 1
        self.fit_steps.append(int(self.pending.step))

        routed = 0
        if reroute_pending and self.pending.size:
            buffer_vectors = self.pending.vectors.copy()
            buffer_outcomes = self.pending.outcomes.copy()
            buffer_targets = self.pending.targets.copy() if self.pending.has_targets else None
            buffer_sources = self.pending.sources.copy()
            self.pending = DynamicMemoryStore(
                max_size=max(self.auto_fit_after, 1),
                latent_dim=self.latent_dim,
                rebuild_every=max(1, min(self.rebuild_every, 256)),
                forget_fraction=self.forget_fraction,
                score_mode=self.score_mode,
                name=f"{self.name}:pending",
            )
            for position in range(buffer_vectors.shape[0]):
                self.add(
                    buffer_vectors[position],
                    float(buffer_outcomes[position]),
                    target_vector=None if buffer_targets is None else buffer_targets[position],
                    source_index=int(buffer_sources[position]),
                )
                routed += 1

        self._refresh_capacities()
        return {
            "num_regions": self.num_regions,
            "iterations": iterations,
            "fit_on": int(source.shape[0]),
            "routed_memories": routed,
            "region_sizes": self.region_sizes.tolist(),
            "empty_regions": int((self.region_sizes == 0).sum()),
        }

    def assign_region(self, vector: np.ndarray) -> int:
        """Index of the nearest region centroid to ``vector`` (cosine)."""
        if not self.is_fitted:
            raise RuntimeError("regions are not fitted yet")
        query = normalize_rows(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        return int(np.argmax(self.centroids @ query))

    def nearest_regions(self, vector: np.ndarray, k: int = 1) -> List[Dict[str, Any]]:
        """The ``k`` nearest regions, best first (~``num_regions`` comparisons)."""
        if not self.is_fitted:
            raise RuntimeError("regions are not fitted yet")
        started = time.perf_counter()
        query = normalize_rows(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        similarities = self.centroids @ query  # (R,) - the ~100-500 comparisons
        wanted = max(1, min(int(k), self.num_regions))
        if wanted < self.num_regions:  # argpartition is O(R); sort only the heads
            heads = np.argpartition(-similarities, wanted - 1)[:wanted]
        else:
            heads = np.arange(self.num_regions)
        order = heads[np.argsort(-similarities[heads], kind="stable")]
        self.last_probe_ms = (time.perf_counter() - started) * 1000.0
        return [
            {"region": int(index), "similarity": float(similarities[index])}
            for index in order
        ]


    # ------------------------------------------------------------- Level 2 add
    def add(
        self,
        position_vector: np.ndarray,
        outcome_weight: float,
        target_vector: Optional[np.ndarray] = None,
        source_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Store one memory in its nearest region (or in the warm-up buffer).

        Returns ``{"region", "position", "fitted"}``. While ``fitted`` is False the
        memory waits in the buffer; once ``auto_fit_after`` memories have gathered,
        the regions are fitted and every buffered memory is routed to its nearest
        region, so "each memory belongs to its nearest region" holds for all of them.
        """
        if not self.is_fitted:
            position = self.pending.add(
                position_vector,
                outcome_weight,
                target_vector=target_vector,
                source_index=source_index,
            )
            fitted = False
            if self.pending.size >= self.auto_fit_after:
                self.fit_regions()
                fitted = True
            return {"region": None, "position": int(position), "fitted": fitted}

        region = self.assign_region(position_vector)
        position = self.regions[region].add(
            position_vector,
            outcome_weight,
            target_vector=target_vector,
            source_index=source_index,
        )
        return {"region": int(region), "position": int(position), "fitted": True}

    # ------------------------------------------------------------- retrieval
    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
        probe_regions: int = 1,
        max_probes: Optional[int] = None,
        update_hits: bool = True,
        exclude_source: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Two-level retrieval: route to regions, then search inside them.

        1. rank all region centroids by cosine to the query (~100-500 comparisons);
        2. probe them best-first - at least ``probe_regions`` of them (an IVF-style
           ``nprobe`` minimum), and more while the result is still short of ``k``, so
           a sparse region spills over into its neighbours instead of returning
           nothing;
        3. concatenate the in-region hits, de-duplicate, keep the best ``k`` by
           similarity.

        Records carry the TASK 28 fields plus ``region`` and ``region_similarity``.
        The nearest region's visit counter and the serving regions' hit counters are
        updated, and the per-region capacities are refreshed every
        ``capacity_refresh_every`` queries. ``max_probes`` caps the spillover.

        ``exclude_source`` is forwarded to every region store, so a memory whose
        ``source_index`` matches is never returned (TASK 29 uses it so an example
        cannot retrieve itself). Because the spillover loop counts only the records
        that survive that filter, excluding entries still yields ``k`` neighbours as
        long as the regions probed hold enough of them.
        """
        if not self.is_fitted:
            return self.pending.query(
                vector, k=k, update_hits=update_hits, exclude_source=exclude_source
            )

        wanted = max(1, int(k))
        minimum = max(1, int(probe_regions))
        limit = self.num_regions if max_probes is None else max(1, int(max_probes))
        started = time.perf_counter()
        ranked = self.nearest_regions(vector, k=min(self.num_regions, minimum))
        self.region_queries += 1
        primary = ranked[0]["region"]
        self._visits[primary] += 1

        records: List[Dict[str, Any]] = []
        seen: set = set()
        probes = 0
        # probe at least `minimum` regions, and keep going while short of k
        while probes < limit and (probes < minimum or len(records) < wanted):
            if probes >= len(ranked):
                extra = self.nearest_regions(vector, k=min(self.num_regions, probes + 1))
                ranked = extra
                if probes + 1 > len(ranked):
                    break
            entry = ranked[probes]
            region = int(entry["region"])
            probes += 1
            self.total_probes += 1
            if self.regions[region].size == 0:
                continue
            for record in self.regions[region].query(
                vector, k=wanted, update_hits=update_hits, exclude_source=exclude_source
            ):
                key = (region, record["position"])
                if key in seen:
                    continue
                seen.add(key)
                record["region"] = region
                record["region_similarity"] = float(entry["similarity"])
                records.append(record)

        records.sort(key=lambda item: item["similarity"], reverse=True)
        records = records[:wanted]
        if records:
            served = np.asarray([record["region"] for record in records], dtype=np.int64)
            self._region_hits += np.bincount(served, minlength=self.num_regions)

        # a query that found nothing local still gets an answer from the buffer
        if not records and self.pending.size:
            records = self.pending.query(vector, k=wanted, update_hits=update_hits)
            for record in records:
                record["region"] = -1
                record["region_similarity"] = float("nan")

        self.last_query_ms = (time.perf_counter() - started) * 1000.0
        if self.capacity_refresh_every and self.region_queries % self.capacity_refresh_every == 0:
            self._refresh_capacities()
        return records


    # ------------------------------------------------- adaptive detail budget
    def _refresh_capacities(self) -> np.ndarray:
        """Set each region's retention capacity from its share of visits.

        ``factor_r = visits_r / mean_visits`` (1.0 = average traffic, 0 = never
        visited), and ``capacity_r = clip(base * factor_r, base/8, base*8)``.
        A region whose traffic is above average is allowed to keep more memories
        (it becomes more detailed), a rarely visited one is squeezed towards
        ``base/8`` (it stays sparse), so the storage budget follows where queries
        actually go instead of being spread uniformly.
        """
        base = self.base_capacity()
        visits = self._visits.astype(np.float64)
        total = float(visits.sum())
        if total <= 0:  # no queries yet: everyone gets the uniform budget
            factors = np.ones(self.num_regions, dtype=np.float64)
        else:
            factors = visits / (total / float(self.num_regions))
        capacities = np.clip(
            base * factors,
            base * self.min_capacity_factor,
            base * self.max_capacity_factor,
        ).astype(np.int64)
        capacities = np.maximum(capacities, 1)
        self._capacities = capacities
        for region, capacity in zip(self.regions, capacities.tolist()):
            region.max_size = int(capacity)
        return capacities

    def region_stats(self) -> List[Dict[str, Any]]:
        """Per-region detail: size, capacity, traffic, hits, outcome, forgetting."""
        rows: List[Dict[str, Any]] = []
        for index, region in enumerate(self.regions):
            rows.append(
                {
                    "region": index,
                    "size": int(region.size),
                    "capacity": int(self._capacities[index]),
                    "visits": int(self._visits[index]),
                    "hits": int(self._region_hits[index]),
                    "mean_outcome_weight": region.mean_outcome_weight,
                    "forgotten": int(region.forgotten),
                    "max_hit_count": (
                        int(region.hit_counts.max()) if region.size else 0
                    ),
                }
            )
        return rows

    def visit_stats(self) -> Dict[str, Any]:
        """How concentrated the traffic is (drives how concentrated the detail is)."""
        visits = self._visits
        total = int(visits.sum())
        ordered = np.sort(visits)[::-1]
        top_share = float(ordered[: max(1, self.num_regions // 10)].sum() / total) if total else 0.0
        probabilities = visits[visits > 0] / total if total else np.zeros(0)
        entropy = float(-(probabilities * np.log(probabilities)).sum()) if total else 0.0
        return {
            "region_queries": int(self.region_queries),
            "visited_regions": int((visits > 0).sum()),
            "num_regions": self.num_regions,
            "top_10pct_visit_share": top_share,
            "max_visits": int(visits.max()) if self.num_regions else 0,
            "mean_visits_when_visited": (
                float(visits[visits > 0].mean()) if (visits > 0).any() else 0.0
            ),
            "visit_entropy_nats": entropy,
            "effective_regions": float(np.exp(entropy)) if total else 0.0,
        }

    def detail_stats(self) -> Dict[str, Any]:
        """Does detail actually follow traffic? (visits vs retained memory count)"""
        if not self.is_fitted:
            return {"pearson_visits_size": 0.0, "quartiles": []}
        sizes = self.region_sizes.astype(np.float64)
        visits = self._visits.astype(np.float64)
        if sizes.std() == 0 or visits.std() == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(visits, sizes)[0, 1])
        order = np.argsort(visits)
        chunks = np.array_split(order, 4)
        quartiles = []
        for index, chunk in enumerate(chunks):
            quartiles.append(
                {
                    "quartile": index + 1,
                    "regions": int(chunk.size),
                    "mean_visits": float(visits[chunk].mean()) if chunk.size else 0.0,
                    "mean_size": float(sizes[chunk].mean()) if chunk.size else 0.0,
                    "mean_capacity": (
                        float(self._capacities[chunk].mean()) if chunk.size else 0.0
                    ),
                }
            )
        return {
            "pearson_visits_size": correlation,
            "quartiles": quartiles,
            "sparse_regions": int((sizes < self.base_capacity()).sum()),
            "detailed_regions": int((sizes > self.base_capacity()).sum()),
        }

    def summary(self) -> Dict[str, Any]:
        """Everything a log line or the DONE block needs."""
        return {
            "name": self.name,
            "fitted": self.is_fitted,
            "num_regions": self.num_regions,
            "size": self.size,
            "routed_size": self.routed_size,
            "pending_size": int(self.pending.size),
            "base_capacity": self.base_capacity(),
            "mean_outcome_weight": self.mean_outcome_weight,
            "region_refits": self.region_refits,
            "queries": int(self.region_queries),
            "mean_probes_per_query": (
                float(self.total_probes / self.region_queries) if self.region_queries else 0.0
            ),
            "visit_stats": self.visit_stats(),
            "detail_stats": self.detail_stats(),
        }


    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        """Pickle centroids + counters + every region's arrays (FAISS is rebuilt)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_VERSION,
            "config": {
                "num_regions": self.num_regions,
                "latent_dim": self.latent_dim,
                "max_size": self.max_size,
                "auto_fit_after": self.auto_fit_after,
                "kmeans_iterations": self.kmeans_iterations,
                "rebuild_every": self.rebuild_every,
                "forget_fraction": self.forget_fraction,
                "capacity_refresh_every": self.capacity_refresh_every,
                "min_capacity_factor": self.min_capacity_factor,
                "max_capacity_factor": self.max_capacity_factor,
                "score_mode": self.score_mode,
                "seed": self.seed,
                "name": self.name,
            },
            "centroids": self.centroids,
            "visits": self._visits,
            "region_hits": self._region_hits,
            "capacities": self._capacities,
            "region_forgotten": self._region_forgotten,
            "region_queries": self.region_queries,
            "total_probes": self.total_probes,
            "region_refits": self.region_refits,
            "fit_steps": self.fit_steps,
            "pending": self._store_payload(self.pending),
            "regions": [self._store_payload(region) for region in self.regions],
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def _store_payload(store: DynamicMemoryStore) -> Dict[str, Any]:
        return {
            "max_size": store.max_size,
            "rebuild_every": store.rebuild_every,
            "stale_fraction": store.stale_fraction,
            "score_mode": store.score_mode,
            "vectors": store.vectors,
            "targets": store.targets,
            "sources": store.sources,
            "outcomes": store.outcomes,
            "timestamps": store.timestamps,
            "hit_counts": store.hit_counts,
            "has_targets": store.has_targets,
            "has_sources": store.has_sources,
            "step": store.step,
            "forgotten": store.forgotten,
            "forget_events": store.forget_events,
            "name": store.name,
        }

    @staticmethod
    def _restore_store(
        payload: Dict[str, Any], latent_dim: int, forget_fraction: float
    ) -> DynamicMemoryStore:
        vectors = payload["vectors"]
        store = DynamicMemoryStore(
            max_size=payload["max_size"],
            latent_dim=latent_dim,
            rebuild_every=payload["rebuild_every"],
            forget_fraction=forget_fraction,
            stale_fraction=payload.get("stale_fraction", 0.25),
            score_mode=payload.get("score_mode", DEFAULT_SCORE_MODE),
            initial_capacity=max(1024, 2 * len(vectors) + 1),
            name=payload["name"],
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
        store._added_since_rebuild = store.rebuild_every
        return store

    @classmethod
    def load(cls, path: Path) -> "RegionStore":
        """Reload a store written by :meth:`save`."""
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        config = payload["config"]
        store = cls(
            num_regions=config["num_regions"],
            latent_dim=config["latent_dim"],
            max_size=config["max_size"],
            auto_fit_after=config["auto_fit_after"],
            kmeans_iterations=config["kmeans_iterations"],
            rebuild_every=config["rebuild_every"],
            forget_fraction=config["forget_fraction"],
            capacity_refresh_every=config["capacity_refresh_every"],
            min_capacity_factor=config["min_capacity_factor"],
            max_capacity_factor=config["max_capacity_factor"],
            score_mode=config.get("score_mode", DEFAULT_SCORE_MODE),
            seed=config["seed"],
            name=config["name"],
        )
        store.centroids = (
            None
            if payload["centroids"] is None
            else np.asarray(payload["centroids"], dtype=np.float32)
        )
        store.regions = [
            cls._restore_store(region, store.latent_dim, store.forget_fraction)
            for region in payload["regions"]
        ]
        store.pending = cls._restore_store(
            payload["pending"], store.latent_dim, store.forget_fraction
        )
        store._visits = np.asarray(payload["visits"], dtype=np.int64)
        store._region_hits = np.asarray(payload["region_hits"], dtype=np.int64)
        store._capacities = np.asarray(payload["capacities"], dtype=np.int64)
        store._region_forgotten = np.asarray(
            payload.get("region_forgotten", np.zeros(store.num_regions)), dtype=np.int64
        )
        store.region_queries = int(payload["region_queries"])
        store.total_probes = int(payload["total_probes"])
        store.region_refits = int(payload["region_refits"])
        store.fit_steps = [int(step) for step in payload.get("fit_steps", [])]
        return store

