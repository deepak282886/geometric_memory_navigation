"""Hierarchical memory check: do regions route correctly, cheaply and adaptively?

Level 1 = region centroids, Level 2 = memories inside each region. This script
verifies the four properties the design claims, on the REAL WikiText latents from
the continuation encoder (``checkpoints/wikitext_val_predictor_vecs_continuation.npz``,
8,171 sentences) plus one synthetic large-scale run for the cost claim:

  1. routing      - every memory sits in its nearest region; centroids are unit-norm
  2. fidelity     - hierarchical top-k vs a flat FAISS top-k over the same vectors
                    (recall@10 and the mean cosine of the returned neighbours)
  3. adaptive detail - traffic concentrated in a few regions grows those regions'
                    capacity and retained size while cold regions stay sparse
  4. spillover    - a query landing in an empty/sparse region still gets k answers
  5. cost         - query time at 100k memories / 500 regions vs the flat store, and
                    how fidelity trades off against the number of regions probed
  6. persistence  - save/load round-trip

None of this re-trains anything: the latents are the frozen encoder's output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.dynamic_store import DynamicMemoryStore  # noqa: E402
from src.memory.region_store import RegionStore  # noqa: E402

LATENT_CACHE = PROJECT_ROOT / "checkpoints" / "wikitext_val_predictor_vecs_continuation.npz"
TMP_PATH = PROJECT_ROOT / "checkpoints" / "_test_region_store.pkl"

SEED = 42
NUM_REGIONS = 256
ROUTING_SAMPLES = 200
QUERIES = 200
#: queries used for the exactness invariant (probing every region is expensive)
EXACT_QUERIES = 20
TOP_K = 10
HOT_FRACTION = 0.1
TRAFFIC_QUERIES = 2000
#: total budget for the adaptive-detail experiment: 40 memories per region on
#: average, so per-region capacity binds and retention reflects traffic
TIGHT_MAX_SIZE = 10240
SCALE_REGIONS = 500
SCALE_MEMORIES = 100000
#: generous budget for the cost test: 2x the memories, so nothing is forgotten and
#: the measurement is purely about lookup cost
SCALE_MAX_SIZE = 2 * SCALE_MEMORIES
SCALE_QUERIES = 100
#: probe counts swept at scale to expose the fidelity/cost trade-off
SCALE_PROBE_SWEEP = (1, 4, 16)
SCALE_FIDELITY_QUERIES = 20


class Checks:
    """PASS/FAIL recorder (INFO = reported finding, does not fail the run)."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str, bool]] = []

    def add(self, name: str, passed: bool, detail: str = "", info_only: bool = False) -> bool:
        self.results.append((name, bool(passed), detail, bool(info_only)))
        label = "INFO" if info_only else ("PASS" if passed else "FAIL")
        print(f"  [{label}] {name}" + (f"  ({detail})" if detail else ""))
        return bool(passed)

    @property
    def failed(self) -> int:
        return sum(
            1 for _, passed, _, info_only in self.results if not passed and not info_only
        )


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def load_real_latents() -> np.ndarray:
    """The continuation encoder's WikiText val latents (already unit-norm)."""
    if not LATENT_CACHE.exists():
        raise FileNotFoundError(
            f"{LATENT_CACHE} not found - run scripts/train_latent_predictor.py with "
            "--vae checkpoints/vae_continuation.pt first"
        )
    with np.load(LATENT_CACHE, allow_pickle=False) as cached:
        vectors = cached["vectors"].astype(np.float32)
    return unit_rows(vectors)


def top_k_positions(store: DynamicMemoryStore, query: np.ndarray, k: int = TOP_K) -> List[int]:
    """Flat FAISS top-k positions, used as the ground truth for the hierarchy."""
    return [record["position"] for record in store.query(query, k=k, update_hits=False)]


def region_stats_table(store: RegionStore) -> Dict[str, Any]:
    """Compact view of the per-region detail (Level 1 x Level 2 summary)."""
    sizes = store.region_sizes
    capacities = store.capacities
    visits = store.visits
    return {
        "regions": store.num_regions,
        "sizes_min": int(sizes.min()),
        "sizes_max": int(sizes.max()),
        "sizes_mean": float(sizes.mean()),
        "empty": int((sizes == 0).sum()),
        "capacity_min": int(capacities.min()) if capacities.size else 0,
        "capacity_max": int(capacities.max()) if capacities.size else 0,
        "visits_total": int(visits.sum()),
    }



def in_region(store: RegionStore, source_index: int, region_index: int) -> bool:
    """Is the memory that came from ``source_index`` currently held in ``region_index``?"""
    if region_index >= len(store.regions) or region_index < 0:
        return False
    return bool((store.regions[region_index].sources == int(source_index)).any())


def part1_routing(checks: Checks, vectors: np.ndarray, rng: np.random.Generator) -> RegionStore:
    """Fit regions on the real latents and verify the Level-1 assignment."""
    print("\n1. regions + routing (real WikiText latents, continuation encoder)")
    store = RegionStore(
        num_regions=NUM_REGIONS,
        latent_dim=vectors.shape[1],
        max_size=100000,
        auto_fit_after=min(2048, len(vectors)),
        seed=SEED,
    )
    # source_index = the row index, so every memory can be traced back afterwards
    for index in range(vectors.shape[0]):
        store.add(vectors[index], 0.9 if index % 2 else 0.2, source_index=index)

    centroid_norms = np.linalg.norm(store.centroids, axis=1)
    checks.add(
        "regions fitted with unit-norm centroids",
        store.is_fitted and bool(np.allclose(centroid_norms, 1.0, atol=1e-5)),
        f"{store.num_regions} regions, |centroid| in "
        f"[{centroid_norms.min():.4f}, {centroid_norms.max():.4f}]",
    )

    # every RETENTION-checked memory must sit in its own nearest region. Memories a
    # full region had to forget are counted separately: being evicted is not being
    # misrouted, and conflating the two is what a naive check gets wrong.
    nearest = (vectors @ store.centroids.T).argmax(axis=1)
    check_positions = rng.choice(
        len(vectors), size=min(ROUTING_SAMPLES, len(vectors)), replace=False
    )
    retained = 0
    evicted = 0
    mismatched = 0
    for position in check_positions:
        region = int(nearest[position])
        if in_region(store, int(position), region):
            retained += 1
        else:
            # is it in some *other* region (a real routing bug), or simply gone?
            elsewhere = any(
                in_region(store, int(position), other)
                for other in range(store.num_regions)
                if other != region
            )
            if elsewhere:
                mismatched += 1
            else:
                evicted += 1
    checks.add(
        "every retained memory is stored in its nearest region",
        mismatched == 0,
        f"{retained} retained, {evicted} forgotten by a full region, "
        f"{mismatched} misrouted (of {len(check_positions)} sampled)",
    )
    total_forgotten = int(sum(region.forgotten for region in store.regions))
    print(
        f"    region detail: {region_stats_table(store)['regions']} regions | sizes "
        f"{region_stats_table(store)['sizes_min']}-{region_stats_table(store)['sizes_max']} "
        f"(mean {region_stats_table(store)['sizes_mean']:.1f}) | empty "
        f"{region_stats_table(store)['empty']} | capacity "
        f"{region_stats_table(store)['capacity_min']}-"
        f"{region_stats_table(store)['capacity_max']} | forgotten {total_forgotten}"
    )
    return store


def flat_reference(vectors: np.ndarray) -> DynamicMemoryStore:
    """One flat store with the same vectors: the ground truth for retrieval.

    ``refresh_index`` matters here: without it the flat store is allowed to lag by up
    to ``stale_fraction`` of its entries, and a stale reference makes the hierarchy
    look worse than it is (that is exactly what the exactness check found).
    """
    flat = DynamicMemoryStore(max_size=10 * len(vectors) + 1, latent_dim=vectors.shape[1])
    for index in range(vectors.shape[0]):
        flat.add(vectors[index], 0.9)
    flat.refresh_index()
    return flat


def top_k_similarities(store: DynamicMemoryStore, query: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Sorted top-k cosine values from a store (the flat ground truth)."""
    values = [record["similarity"] for record in store.query(query, k=k, update_hits=False)]
    return np.sort(np.asarray(values, dtype=np.float64))[::-1]


def profile_match(hierarchical: np.ndarray, flat: np.ndarray, tol: float = 1e-4) -> float:
    """Share of the flat top-k neighbours the hierarchical result recovers (recall@k).

    Both inputs are sorted descending cosine values. Matching is greedy with a
    tolerance: each flat value is paired with an as-yet-unused hierarchical value
    within ``tol``, so 10 identical values count as 10 matches and near-ties cannot
    mis-align the pairing.

    Comparing *similarities* rather than positions is deliberate: WikiText sentences
    repeat across pairs, so the same latent can be stored under two source indices,
    and then "the same position" is not well defined while "the same neighbour at the
    same distance" is.
    """
    if flat.size == 0:
        return 1.0
    used = np.zeros(hierarchical.size, dtype=bool)
    matched = 0
    for value in flat:
        candidates = np.flatnonzero(~used & (np.abs(hierarchical - value) <= tol))
        if candidates.size:
            used[candidates[0]] = True
            matched += 1
    return matched / flat.size


def part2_fidelity(
    checks: Checks,
    vectors: np.ndarray,
    flat: DynamicMemoryStore,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], RegionStore]:
    """Hierarchical top-k vs flat top-k, as a function of how many regions are probed.

    Two structural claims, no magic thresholds:

      * probing more regions must recover more of the flat top-10 (monotone curve);
      * probing ALL regions must reproduce the flat top-10 exactly - that is an
        invariant of the design (every memory is in exactly one region, so searching
        every region searches every memory), and it is the only place where the
        hierarchy can be checked for exactness.

    The store used here gets a budget that cannot bind, so the comparison measures
    ROUTING rather than the interaction between routing and local forgetting.
    """
    print("\n2. retrieval fidelity vs a flat store over the same vectors")
    store = RegionStore(
        num_regions=NUM_REGIONS,
        latent_dim=vectors.shape[1],
        max_size=NUM_REGIONS * len(vectors),  # base capacity 8171/region: never binds
        auto_fit_after=min(2048, len(vectors)),
        seed=SEED,
    )
    for index in range(vectors.shape[0]):
        store.add(vectors[index], 0.9, source_index=index)

    queries = rng.choice(len(vectors), size=QUERIES, replace=False)
    rows: List[Dict[str, Any]] = []
    for probes in (1, 2, 4, 8, 16, None):
        recall_sum = 0.0
        cos_hierarchical: List[float] = []
        cos_flat: List[float] = []
        probes_used: List[int] = []
        for index in queries:
            query = vectors[index]
            truth = top_k_similarities(flat, query)
            before = store.total_probes
            records = store.query(
                query, k=TOP_K, probe_regions=probes or 1, max_probes=probes
            )
            probes_used.append(store.total_probes - before)
            found = np.sort(
                np.asarray([record["similarity"] for record in records], dtype=np.float64)
            )[::-1]
            recall_sum += profile_match(found, truth)
            cos_hierarchical.extend(record["similarity"] for record in records)
            cos_flat.extend(truth.tolist())
        recall = recall_sum / len(queries)
        rows.append(
            {
                "probe_regions_setting": "spillover" if probes is None else probes,
                "mean_probes_per_query": float(np.mean(probes_used)),
                "recall_at_10": recall,
                "mean_cos_hierarchical": float(np.mean(cos_hierarchical)),
                "mean_cos_flat": float(np.mean(cos_flat)),
            }
        )
        print(
            f"    probes {str(rows[-1]['probe_regions_setting']):>9} | "
            f"mean probes/query {rows[-1]['mean_probes_per_query']:.2f} | "
            f"recall@10 {100 * recall:.1f}% | mean cos "
            f"{rows[-1]['mean_cos_hierarchical']:.4f} vs flat "
            f"{rows[-1]['mean_cos_flat']:.4f}"
        )

    ordered = [row for row in rows if isinstance(row["probe_regions_setting"], int)]
    checks.add(
        "probing more regions recovers more of the flat top-10",
        all(
            later["recall_at_10"] >= earlier["recall_at_10"] - 1e-9
            for earlier, later in zip(ordered, ordered[1:])
        ),
        "recall@10 " + " < ".join(f"{100 * row['recall_at_10']:.1f}%" for row in ordered),
    )

    # exactness: probe every region on a subsample (this is the invariant test)
    exact_queries = queries[:EXACT_QUERIES]
    exact_matches = 0
    for index in exact_queries:
        truth = top_k_similarities(flat, vectors[index])
        records = store.query(
            vectors[index], k=TOP_K, probe_regions=store.num_regions, max_probes=None
        )
        found = np.sort(
            np.asarray([record["similarity"] for record in records], dtype=np.float64)
        )[::-1]
        if profile_match(found, truth) >= 1.0:
            exact_matches += 1
    checks.add(
        "probing all regions reproduces the flat top-10 exactly",
        exact_matches == len(exact_queries),
        f"{exact_matches}/{len(exact_queries)} queries identical to the flat result",
    )
    print(
        f"    (practical settings: {100 * ordered[1]['recall_at_10']:.1f}% recall at 2 probes, "
        f"{100 * ordered[-1]['recall_at_10']:.1f}% at 16, cos gap "
        f"{ordered[-1]['mean_cos_hierarchical'] - ordered[-1]['mean_cos_flat']:+.4f})"
    )
    return rows, store



def part3_adaptive_detail(
    checks: Checks, vectors: np.ndarray, rng: np.random.Generator
) -> Dict[str, Any]:
    """Concentrate traffic in a few regions and see whether detail follows it.

    Uses its own store with a TIGHT total budget (``TIGHT_MAX_SIZE``), because
    capacity only shapes retention when it binds: with room for everything, every
    region keeps every memory and "visited regions get more detail" has nothing to
    do. This is the regime the design is meant to be used in - a budget smaller than
    the number of experiences.
    """
    print("\n3. adaptive detail (traffic concentrated in ~10% of regions)")
    store = RegionStore(
        num_regions=NUM_REGIONS,
        latent_dim=vectors.shape[1],
        max_size=TIGHT_MAX_SIZE,
        auto_fit_after=min(2048, len(vectors)),
        seed=SEED,
    )
    for index in range(vectors.shape[0]):
        store.add(vectors[index], 0.9 if index % 2 else 0.2, source_index=index)

    hot_count = max(1, int(HOT_FRACTION * store.num_regions))
    hot_regions = rng.choice(store.num_regions, size=hot_count, replace=False)
    for _ in range(TRAFFIC_QUERIES):
        region = int(rng.choice(hot_regions))
        # a query near that region's centroid (plus noise) - real traffic there
        store.query(store.centroids[region] + 0.1 * rng.normal(size=vectors.shape[1]), k=TOP_K)

    # keep feeding experiences so the per-region budgets bind
    extra = unit_rows(rng.normal(size=(TIGHT_MAX_SIZE, vectors.shape[1])))
    for index in range(extra.shape[0]):
        store.add(extra[index], 0.9, source_index=len(vectors) + index)

    visits = store.visit_stats()
    detail = store.detail_stats()
    checks.add(
        "traffic is concentrated (the premise of adaptive detail)",
        visits["top_10pct_visit_share"] > 0.5,
        f"top 10% of regions hold {100 * visits['top_10pct_visit_share']:.1f}% of "
        f"{visits['region_queries']} queries, {visits['visited_regions']}/"
        f"{visits['num_regions']} regions ever visited",
    )
    checks.add(
        "visited regions get more capacity and hold more memories",
        detail["pearson_visits_size"] > 0.5
        and detail["quartiles"][-1]["mean_size"] > detail["quartiles"][0]["mean_size"],
        f"corr(visits, size) {detail['pearson_visits_size']:+.3f}, cold size "
        f"{detail['quartiles'][0]['mean_size']:.1f} vs hot "
        f"{detail['quartiles'][-1]['mean_size']:.1f}",
    )
    print("    detail by visit quartile (cold -> hot)")
    print("    quartile | regions | mean visits | mean size | mean capacity")
    print("    " + "-" * 64)
    for row in detail["quartiles"]:
        print(
            f"    {row['quartile']:>8} | {row['regions']:>7} | {row['mean_visits']:>11.1f} | "
            f"{row['mean_size']:>9.1f} | {row['mean_capacity']:>13.0f}"
        )
    cold = detail["quartiles"][0]
    hot = detail["quartiles"][-1]
    checks.add(
        "rarely visited regions stay sparse",
        cold["mean_capacity"] < store.base_capacity() <= hot["mean_capacity"]
        and cold["mean_size"] < store.base_capacity(),
        f"cold capacity {cold['mean_capacity']:.0f} / size {cold['mean_size']:.1f} < "
        f"base {store.base_capacity()} <= hot capacity {hot['mean_capacity']:.0f} / "
        f"size {hot['mean_size']:.1f}",
    )
    print(
        f"    budget: max_size {store.max_size} = base {store.base_capacity()}/region | "
        f"memories {store.size} | retained per region "
        f"{int(store.region_sizes.min())}-{int(store.region_sizes.max())}"
    )
    return {"visit_stats": visits, "detail_stats": detail, "store": store}


def part4_spillover(
    checks: Checks, store: RegionStore, vectors: np.ndarray, rng: np.random.Generator
) -> None:
    """A query in an empty/sparse region must still get answers."""
    print("\n4. spillover from sparse regions")
    empty = np.flatnonzero(store.region_sizes == 0)
    if empty.size == 0:
        sizes = store.region_sizes
        target = int(np.argmin(sizes))
        detail = f"smallest region {target} holds {int(sizes[target])} memories"
    else:
        target = int(empty[0])
        detail = f"region {target} is empty"
    query = store.centroids[target] + 0.05 * rng.normal(size=vectors.shape[1])
    single = store.query(query, k=TOP_K, probe_regions=1, max_probes=1)
    spill = store.query(query, k=TOP_K, probe_regions=1, max_probes=None)
    checks.add(
        "sparse-region query still returns k neighbours",
        len(spill) == TOP_K,
        f"{detail}; strict 1-region lookup returned {len(single)}, "
        f"with spillover {len(spill)}",
    )

    # self-masking: an entry must never come back from its own query
    own_source = int(store.regions[0].sources[0]) if store.regions[0].size else 0
    own_vector = store.regions[0].vectors[0] if store.regions[0].size else vectors[0]
    masked = store.query(own_vector, k=TOP_K, max_probes=None, exclude_source=own_source)
    unmasked = store.query(own_vector, k=TOP_K, max_probes=None)
    checks.add(
        "exclude_source masks a memory out of its own retrieval",
        all(record.get("source") != own_source for record in masked)
        and any(record.get("source") == own_source for record in unmasked),
        f"source {own_source}: present without masking "
        f"{any(record.get('source') == own_source for record in unmasked)}, "
        f"absent with masking "
        f"{all(record.get('source') != own_source for record in masked)} "
        f"({len(masked)} records returned)",
    )
    outside = unit_rows(rng.normal(size=(1, vectors.shape[1])))[0]
    far = store.query(outside * -1.0, k=TOP_K, max_probes=None)
    checks.add(
        "query far outside the data still returns k neighbours",
        len(far) == TOP_K,
        f"{len(far)} records",
    )


def part5_scale(checks: Checks, rng: np.random.Generator) -> Dict[str, Any]:
    """Cost and fidelity at 100k memories: flat scan vs 500-region two-level lookup."""
    print(f"\n5. cost and fidelity at {SCALE_MEMORIES} memories / {SCALE_REGIONS} regions")
    vectors = unit_rows(rng.normal(size=(SCALE_MEMORIES, 256)))

    flat = DynamicMemoryStore(max_size=SCALE_MEMORIES + 1, latent_dim=256)
    started = time.perf_counter()
    for index in range(SCALE_MEMORIES):
        flat.add(vectors[index], 0.9 if index % 2 else 0.2)
    flat.refresh_index()  # exact ground truth for the fidelity comparison
    flat_build = time.perf_counter() - started

    started = time.perf_counter()
    store = RegionStore(
        num_regions=SCALE_REGIONS,
        latent_dim=256,
        max_size=SCALE_MAX_SIZE,
        auto_fit_after=2048,
        seed=SEED,
    )
    for index in range(SCALE_MEMORIES):
        store.add(vectors[index], 0.9 if index % 2 else 0.2, source_index=index)
    region_build = time.perf_counter() - started

    checks.add(
        "all experiences retained when the budget allows it",
        store.size == SCALE_MEMORIES and flat.size == SCALE_MEMORIES,
        f"region store {store.size}, flat {flat.size} (budget {SCALE_MAX_SIZE}, "
        f"base {store.base_capacity()}/region), built in {region_build:.2f}s vs "
        f"{flat_build:.2f}s",
    )

    queries = unit_rows(rng.normal(size=(SCALE_QUERIES, 256)))
    flat_times: List[float] = []
    for index in range(SCALE_QUERIES):
        flat.query(queries[index], k=TOP_K, update_hits=False)
        flat_times.append(flat.last_query_ms)

    sweep: List[Dict[str, Any]] = []
    for probes in (*SCALE_PROBE_SWEEP, None):
        times: List[float] = []
        probe_times: List[float] = []
        for index in range(SCALE_QUERIES):
            store.query(
                queries[index], k=TOP_K, probe_regions=probes or 1, max_probes=probes
            )
            times.append(store.last_query_ms)
            probe_times.append(store.last_probe_ms)
        # fidelity on a subsample (a flat query per example is cheap)
        recall = 0.0
        cos_hierarchical: List[float] = []
        cos_flat: List[float] = []
        for index in range(SCALE_FIDELITY_QUERIES):
            truth = top_k_similarities(flat, queries[index])
            records = store.query(
                queries[index], k=TOP_K, probe_regions=probes or 1, max_probes=probes
            )
            found = np.sort(
                np.asarray([record["similarity"] for record in records], dtype=np.float64)
            )[::-1]
            recall += profile_match(found, truth)
            cos_hierarchical.extend(record["similarity"] for record in records)
            cos_flat.extend(truth.tolist())
        sweep.append(
            {
                "probe_regions_setting": "spillover" if probes is None else probes,
                "mean_ms": float(np.mean(times)),
                "probe_ms": float(np.mean(probe_times)),
                "recall_at_10": recall / SCALE_FIDELITY_QUERIES,
                "mean_cos_hierarchical": float(np.mean(cos_hierarchical)),
                "mean_cos_flat": float(np.mean(cos_flat)),
                "mean_probes": float(
                    store.total_probes / max(1, store.region_queries)
                ),
            }
        )
        row = sweep[-1]
        print(
            f"    probes {str(row['probe_regions_setting']):>9} | "
            f"{row['mean_ms']:.3f} ms ({row['probe_ms']:.3f} ms centroid scan) | "
            f"recall@10 {100 * row['recall_at_10']:.1f}% | mean cos "
            f"{row['mean_cos_hierarchical']:.4f} vs flat {row['mean_cos_flat']:.4f}"
        )

    flat_mean = float(np.mean(flat_times))
    fastest = sweep[0]
    checks.add(
        "two-level lookup is much faster than the flat scan at 100k",
        fastest["mean_ms"] < flat_mean / 5,
        f"{fastest['mean_ms']:.3f} ms vs {flat_mean:.3f} ms "
        f"({flat_mean / fastest['mean_ms']:.1f}x), centroid scan "
        f"{fastest['probe_ms']:.3f} ms of it",
    )
    ordered = [row for row in sweep if isinstance(row["probe_regions_setting"], int)]
    checks.add(
        "more probes recover more of the flat top-10 at scale too",
        ordered[-1]["recall_at_10"] > ordered[0]["recall_at_10"],
        "recall@10 "
        + " < ".join(
            f"{100 * row['recall_at_10']:.1f}% ({row['mean_ms']:.3f} ms)" for row in ordered
        ),
    )
    checks.add(
        "every probe setting still beats the flat scan",
        max(row["mean_ms"] for row in sweep) < flat_mean,
        f"slowest setting {max(row['mean_ms'] for row in sweep):.3f} ms vs flat "
        f"{flat_mean:.3f} ms",
        info_only=False,
    )
    checks.add(
        "absolute recall at scale (uniform random latents = worst case)",
        True,
        f"recall@10 {100 * ordered[0]['recall_at_10']:.1f}% at "
        f"{ordered[0]['probe_regions_setting']} probe(s), mean cos "
        f"{ordered[0]['mean_cos_hierarchical']:.4f} vs flat {ordered[0]['mean_cos_flat']:.4f} - "
        "random vectors in 256-d have no neighbourhood structure, so this is a lower "
        "bound on routing fidelity, not a property of the design",
        info_only=True,
    )
    return {
        "flat_mean_ms": flat_mean,
        "sweep": sweep,
        "region_build_seconds": region_build,
        "flat_build_seconds": flat_build,
        "region_summary": store.summary(),
    }


def part6_persistence(checks: Checks, store: RegionStore) -> None:
    """Save/load must preserve centroids, counters and every region."""
    print("\n6. persistence")
    store.save(TMP_PATH)
    reloaded = RegionStore.load(TMP_PATH)
    checks.add(
        "save/load round-trip",
        reloaded.size == store.size
        and reloaded.num_regions == store.num_regions
        and np.allclose(reloaded.centroids, store.centroids)
        and np.array_equal(reloaded.visits, store.visits)
        and reloaded.score_mode == store.score_mode,
        f"size {reloaded.size}, regions {reloaded.num_regions}, "
        f"queries {reloaded.region_queries}, score_mode {reloaded.score_mode}",
    )
    TMP_PATH.unlink(missing_ok=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    checks = Checks()

    print("=" * 88)
    print("HIERARCHICAL MEMORY - REGIONS (Level 1) over MEMORIES (Level 2)")
    print("=" * 88)

    vectors = load_real_latents()
    print(
        f"\nreal latents: {vectors.shape} from {LATENT_CACHE.name} "
        f"(continuation encoder, |v| in "
        f"[{np.linalg.norm(vectors, axis=1).min():.4f}, "
        f"{np.linalg.norm(vectors, axis=1).max():.4f}])"
    )

    store = part1_routing(checks, vectors, rng)
    flat = flat_reference(vectors)
    fidelity, fidelity_store = part2_fidelity(checks, vectors, flat, rng)
    detail = part3_adaptive_detail(checks, vectors, rng)
    part4_spillover(checks, store, vectors, rng)
    scale = part5_scale(checks, rng)
    part6_persistence(checks, store)

    print("\n" + "=" * 88)
    print("REGION STORE CHECK SUMMARY")
    print("=" * 88)
    for name, passed, detail_text, info_only in checks.results:
        label = "INFO" if info_only else ("PASS" if passed else "FAIL")
        print(f"  {label}  {name}" + (f"  ({detail_text})" if detail_text else ""))
    print("=" * 88)
    print(
        f"  hierarchy summary: {store.num_regions} regions | memories {store.size} | "
        f"mean probes/query "
        f"{store.total_probes / max(1, store.region_queries):.2f} | "
        f"visits top-10% share "
        f"{100 * detail['visit_stats']['top_10pct_visit_share']:.1f}%"
    )
    print(
        "  fidelity at 8,171 real latents (recall@10 vs flat): "
        + " | ".join(
            f"probes {row['probe_regions_setting']} {100 * row['recall_at_10']:.1f}%"
            for row in fidelity
        )
    )
    print(
        f"  at {SCALE_MEMORIES} memories / {SCALE_REGIONS} regions: "
        f"{scale['sweep'][0]['mean_ms']:.3f} ms vs flat {scale['flat_mean_ms']:.3f} ms "
        f"({scale['flat_mean_ms'] / scale['sweep'][0]['mean_ms']:.1f}x at 1 probe; "
        f"recall@10 {100 * scale['sweep'][0]['recall_at_10']:.1f}% -> "
        f"{100 * scale['sweep'][-1]['recall_at_10']:.1f}% with spillover)"
    )
    print("=" * 88)
    if checks.failed:
        print(f"{checks.failed} check(s) FAILED")
        raise SystemExit(1)
    print("all checks passed")
    print("=" * 88)


if __name__ == "__main__":
    main()

