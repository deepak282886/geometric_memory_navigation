"""Fuse the sampled memory directions into a single navigation vector.

Each retrieved memory contributes ``(weight, direction)`` where the direction
points from the query toward that memory. Fusion keeps only the strongest
``top_k`` contributions (the "Jev" selection), turns the weights into a convex
combination (they sum to 1) and L2-normalises the result, so the navigation
vector is a unit direction that can be added to the query position.

Important correction (measured, see ``tangent_direction``): with unit-normalised
embeddings the raw chords point *away* from the query (``cos(ctx, chord)`` is
negative for any neighbour that is not identical), which made the fused
navigation vector sit ~127 degrees from the context and actively destroyed the
scoring signal. ``fuse_directions`` therefore projects each direction onto the
query's tangent space by default when a ``query_vector`` is supplied.

``navigate`` then stacks the current position and the navigation vector into the
512-d vector consumed by the classifier (TASK 7).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_TOP_K = 3
#: anything with an L2 norm below this is treated as the zero vector
EPS = 1e-12
#: a projected direction shorter than this carries no tangential information
TANGENT_EPS = 1e-8


def tangent_direction(
    direction: np.ndarray,
    query_vector: np.ndarray,
    eps: float = TANGENT_EPS,
) -> Optional[np.ndarray]:
    """Remove the radial component of ``direction``: ``d - (d.q) q``.

    ``query_vector`` must be L2-normalised (the memory store and every caller in
    this project keep contexts unit-norm), so ``(d.q) q`` is exactly the radial
    projection of ``d`` onto the query axis.

    This exists because of a measured geometric failure. With unit-normalised
    embeddings, the chord ``memory - ctx`` satisfies
    ``cos(ctx, chord) = (mu - 1) / sqrt(2 - 2 mu)`` for ``mu = cos(ctx, memory)``,
    which is negative whenever ``mu < 1``. Fusing chords therefore produced a
    navigation vector at ~127 degrees from the context and destroyed the scoring
    signal. Projecting it out leaves a step that is orthogonal to the context, so
    ``cos(ctx + nav_tangent)`` stays positive and the context alignment survives.

    Returns ``None`` when the projection is (numerically) zero, i.e. the
    direction is parallel to the query and carries no tangential information.
    """
    d = np.asarray(direction, dtype=np.float64).ravel()
    q = np.asarray(query_vector, dtype=np.float64).ravel()
    if d.shape != q.shape:
        raise ValueError(f"direction dim {d.shape} != query dim {q.shape}")
    projected = d - float(np.dot(d, q)) * q
    norm = float(np.linalg.norm(projected))
    if norm < eps:
        return None
    return projected / norm


def fuse_directions(
    weighted_directions: Sequence[Tuple[float, np.ndarray]],
    top_k: int = DEFAULT_TOP_K,
    dim: Optional[int] = None,
    use_tangent: bool = True,
    query_vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted sum of the ``top_k`` strongest directions, L2-normalised.

    ``weighted_directions`` is the output of ``parallel_sample``: a list of
    ``(weight, direction)`` pairs. The steps are:

    1. (optionally) project every direction onto the tangent space of the query
    2. sort by weight descending (stable, so equal weights keep retrieval order)
    3. keep the ``top_k``
    4. rescale those weights so they sum to 1.0
    5. take the weighted sum of their directions
    6. L2-normalise the result

    ``use_tangent=True`` with a ``query_vector`` switches on the radial-component
    removal described in :func:`tangent_direction`; directions whose tangent part
    vanishes are dropped. When ``use_tangent=False`` or ``query_vector is None``
    the behaviour is exactly the original chord-based fusion.

    If every selected weight is zero (or their total is not positive) the zero
    vector is returned. ``dim`` is only consulted when ``weighted_directions`` is
    empty, since the dimensionality is otherwise inferred from the directions.
    """
    pairs: List[Tuple[float, np.ndarray]] = [
        (float(weight), np.asarray(direction, dtype=np.float64).ravel())
        for weight, direction in weighted_directions
    ]
    if not pairs:
        return np.zeros(int(dim) if dim else 0, dtype=np.float64)

    length = pairs[0][1].shape[0]
    for _, direction in pairs:
        if direction.shape[0] != length:
            raise ValueError(
                f"inconsistent direction lengths: expected {length}, "
                f"got {direction.shape[0]}"
            )

    if use_tangent and query_vector is not None:
        tangential: List[Tuple[float, np.ndarray]] = []
        for weight, direction in pairs:
            tangent = tangent_direction(direction, query_vector)
            if tangent is None:
                continue  # parallel to the query: nothing tangential to add
            tangential.append((weight, tangent))
        pairs = tangential
        if not pairs:
            return np.zeros(length, dtype=np.float64)

    # 1. strongest contribution first (stable for ties)
    pairs.sort(key=lambda item: item[0], reverse=True)
    # 2. keep the top_k
    keep = max(1, int(top_k))
    selected = pairs[:keep]

    weights = np.asarray([weight for weight, _ in selected], dtype=np.float64)
    total = float(weights.sum())
    # 3. guard: no usable weight at all -> zero vector
    if total <= EPS:
        return np.zeros(length, dtype=np.float64)

    # normalise the weights so they form a convex combination
    weights = weights / total

    # 4. weighted sum of the selected directions
    stacked = np.stack([direction for _, direction in selected])
    navigation_vector = weights @ stacked

    # 5. unit length
    norm = float(np.linalg.norm(navigation_vector))
    if norm <= EPS:
        return np.zeros(length, dtype=np.float64)
    return navigation_vector / norm


def navigate(
    query_vector: np.ndarray,
    navigation_vector: np.ndarray,
) -> np.ndarray:
    """Concatenate ``[position ; navigation]`` -> ``(2 * latent_dim,)``.

    Both inputs are expected to be ``(256,)`` for this project, giving the
    512-d vector that the classifier head consumes.
    """
    query = np.asarray(query_vector, dtype=np.float64).ravel()
    navigation = np.asarray(navigation_vector, dtype=np.float64).ravel()
    if query.shape[0] != navigation.shape[0]:
        raise ValueError(
            f"query dim {query.shape[0]} != navigation dim {navigation.shape[0]}"
        )
    return np.concatenate([query, navigation], axis=0)
