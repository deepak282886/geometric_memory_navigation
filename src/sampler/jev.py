"""Parallel Jev sampler: turn each retrieved memory into (weight, direction).

For every memory in the kNN result the sampler produces

    weight     = cosine_similarity(query, TARGET) * outcome_weight
    direction  = TARGET - query_vector      (NOT normalised here)

where ``TARGET`` is, by default, the memory's **known-correct continuation**
(``memory["correct_ending_vector"]``) rather than the neighbouring context. This
is the correction from the "continuation" step: pointing at similar *contexts*
carries no information about which ending is correct, whereas pointing at the
correct *continuations* of similar contexts does. Pass ``target="context"`` to
recover the original context-target behaviour.

``weight`` says *how much to trust* the memory, ``direction`` says *where it
points*. Fusing those directions is TASK 6's job, which is why the direction is
deliberately left un-normalised.

Vectors coming out of the FAISS memory store are already L2-normalised (and
``MemoryStore.query`` normalises the query the same way), so the cosine
similarity is simply the dot product - see ``cosine_similarity``.

``parallel_sample`` processes every memory in a single vectorised numpy pass.
An earlier ``ThreadPoolExecutor`` implementation was measurably worse: the pool
cost ~0.70 ms per call while the actual work was ~0.03 ms, because pool
spin-up dominates and numpy's dot holds the GIL on such small arrays. The
vectorised version keeps the same signature and the same output ordering.

``sample_memory`` and ``cosine_similarity`` are kept as the readable
single-memory utilities, and ``sample_sequentially`` is retained as a reference
implementation for equivalence checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: floor applied to non-positive weights so downstream fusion never sees <= 0
MIN_WEIGHT = 1e-6

#: navigate toward the memory's known-correct continuation (the corrected model)
TARGET_CORRECT_ENDING = "correct_ending"
#: navigate toward the neighbouring context (the original, chord/tangent model)
TARGET_CONTEXT = "context"
TARGETS = (TARGET_CORRECT_ENDING, TARGET_CONTEXT)

WeightedDirection = Tuple[float, np.ndarray]


def target_matrix(
    memories: Sequence[Dict[str, Any]],
    target: str = TARGET_CORRECT_ENDING,
) -> np.ndarray:
    """Stack the vectors navigation should point at: ``(N, 256)``.

    ``target="correct_ending"`` uses ``memory["correct_ending_vector"]`` and fails
    loudly if the store was built without them (silently falling back would make
    an experiment report the wrong thing).
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    if target == TARGET_CORRECT_ENDING:
        missing = [
            position
            for position, memory in enumerate(memories)
            if memory.get("correct_ending_vector") is None
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(memories)} memories have no "
                "'correct_ending_vector' (first at position {missing[0]}). Rebuild the "
                "store including correct endings "
                "(e.g. checkpoints/memory_store_full_v2.pkl), or call with "
                "target='context' to use the original context-target model."
            )
        return np.stack(
            [
                np.asarray(memory["correct_ending_vector"], dtype=np.float64).ravel()
                for memory in memories
            ]
        )
    return np.stack(
        [np.asarray(memory["vector"], dtype=np.float64).ravel() for memory in memories]
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors.

    The memory store stores unit-norm vectors and ``MemoryStore.query``
    normalises the query vector before searching, so both arguments are already
    unit length and the cosine similarity reduces to ``np.dot(a, b)``. Vectors
    must therefore be normalised by the caller.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"vector shape mismatch: {a.shape} vs {b.shape}")
    # Inputs are unit-norm, so the dot product IS the cosine similarity.
    return float(np.dot(a, b))


def sample_memory(
    query_vector: np.ndarray,
    memory: Dict[str, Any],
    outcome_weight: float,
    target: str = TARGET_CORRECT_ENDING,
) -> WeightedDirection:
    """Return ``(weight, direction)`` for a single memory.

    ``weight`` is clamped to :data:`MIN_WEIGHT` when the similarity (or the
    outcome weight) is zero or negative, so fusion never receives a
    non-positive weight. The direction is left un-normalised on purpose.
    """
    memory_vector = target_matrix([memory], target)[0]
    query = np.asarray(query_vector, dtype=np.float64).ravel()

    similarity = cosine_similarity(query, memory_vector)
    weight = similarity * float(outcome_weight)
    if weight <= 0.0:
        weight = MIN_WEIGHT

    direction = memory_vector - query
    return float(weight), direction


def parallel_sample(
    query_vector: np.ndarray,
    memories: Sequence[Dict[str, Any]],
    outcome_weights: Optional[Sequence[float]] = None,
    max_workers: Optional[int] = None,
    target: str = TARGET_CORRECT_ENDING,
) -> List[WeightedDirection]:
    """Sample every memory in one vectorised numpy pass.

    ``outcome_weights`` must have one entry per memory (pass all ``1.0`` at
    inference time, since the query label is unknown). If omitted, every memory
    gets a weight of ``1.0``. Results keep the input order, so fusion is
    deterministic.

    ``target`` selects what navigation points at: ``"correct_ending"`` (default)
    uses each memory's known-correct continuation; ``"context"`` reproduces the
    original context-target sampler. Either way the weight is the relevance of
    that target to the query, i.e. ``target . query_vector``.

    ``max_workers`` is accepted only for backwards compatibility with the old
    thread-pool version and is ignored - there is nothing left to parallelise
    once the work is a single ``(N, 256) @ (256,)`` product.

    NOTE: this is named after the "parallel Jev sampler" it implements. Compute
    stays in float64 (like ``sample_memory``) so both paths agree numerically.
    """
    memories = list(memories)
    if outcome_weights is None:
        outcome = np.ones(len(memories), dtype=np.float64)
    else:
        outcome = np.asarray([float(value) for value in outcome_weights], dtype=np.float64)
        if outcome.shape[0] != len(memories):
            raise ValueError(
                f"got {outcome.shape[0]} outcome_weights for {len(memories)} memories"
            )

    query = np.asarray(query_vector, dtype=np.float64).ravel()
    if not memories:
        print("sampled 0 memories (vectorized)")
        return []

    # (N, 256) targets: correct continuations by default, contexts if requested
    matrix = target_matrix(memories, target)
    weights = matrix @ query                 # (N,) relevance of each target to the query
    weights = weights * outcome              # (N,) apply the outcome weights
    weights = np.maximum(weights, MIN_WEIGHT)  # (N,) clamp zero/negative weights

    directions = matrix - query[None, :]     # (N, 256) un-normalised directions

    print(f"sampled {len(memories)} memories (vectorized, target={target})")
    return list(zip(weights.tolist(), directions))


def sample_sequentially(
    query_vector: np.ndarray,
    memories: Sequence[Dict[str, Any]],
    outcome_weights: Optional[Sequence[float]] = None,
    target: str = TARGET_CORRECT_ENDING,
) -> List[WeightedDirection]:
    """Reference sequential implementation (same results, used for equivalence checks)."""
    memories = list(memories)
    weights = (
        [1.0] * len(memories)
        if outcome_weights is None
        else [float(value) for value in outcome_weights]
    )
    query = np.asarray(query_vector, dtype=np.float64).ravel()
    return [
        sample_memory(query, memory, weight, target=target)
        for memory, weight in zip(memories, weights)
    ]


def retrieval_confidence(
    query_vector: np.ndarray,
    neighbors: Sequence[Dict[str, Any]],
    threshold: float = 0.5,
    top_n: int = 3,
) -> Tuple[bool, float]:
    """Are the retrieved neighbours relevant enough to trust the memory context?

    TASK 29 measured the mechanism that makes this worth asking: the prediction delta
    of memory-guided prediction correlates +0.569 with whether the fused context points
    at the true next position, and the memory helps in-corpus (+0.0039 cosine) while it
    hurts out-of-corpus (-0.0049). A query whose neighbours sit in the wrong
    neighbourhood should therefore fall back to ``C = zeros`` instead of being dragged
    off by them.

    Metric: the mean cosine similarity of the ``top_n`` nearest neighbours to the query
    (``top_n = 3`` because fusion weights the top-3 in ``fuse_directions``). ``True``
    when that mean is ``>= threshold``, i.e. the neighbours really are around the query
    rather than merely the least-bad memories in a distant region.

    Returns ``(is_relevant, mean_similarity)``; an empty neighbour list gives
    ``(False, 0.0)``, so a cold memory is never trusted.

    Neighbour similarity keys
    -------------------------
    Every record in this project carries the query-to-neighbour cosine, but the two
    store generations name it differently: ``MemoryStore`` (src/memory/store.py) uses
    ``"distance"`` and ``DynamicMemoryStore``/``RegionStore`` use ``"similarity"``.
    Both are accepted, so one gate serves both.

    Calibration caveat
    ------------------
    The threshold's scale depends on ``top_n``: the TASK 29 numbers quoted for
    calibration (in-corpus 0.7148 vs out-of-corpus 0.6704) are TOP-10 means, whereas
    this statistic is a TOP-3 mean, which sits higher for the same query. The two are
    not interchangeable, so the meaningful threshold range has to be read off the
    actual distribution (``scripts/run_gated_sweep.py`` prints it). ``0.5`` is the
    conservative default this task asked for: a query is trusted only when its three
    nearest memories are all fairly close.
    """
    if not neighbors:
        return False, 0.0
    similarities: List[float] = []
    for neighbor in neighbors:
        value = neighbor.get("similarity", neighbor.get("distance"))
        if value is None:
            continue
        similarities.append(float(value))
    if not similarities:
        return False, 0.0
    keep = max(1, int(top_n))
    top = sorted(similarities, reverse=True)[:keep]
    mean_similarity = float(np.mean(top))
    return mean_similarity >= float(threshold), mean_similarity

