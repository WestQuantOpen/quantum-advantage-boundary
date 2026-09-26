"""MWIS (Maximum Weight Independent Set) problem generation.

This module provides reproducible generation of MWIS instances over random
graphs, together with small helper utilities used throughout the experiment.

The two graph families supported are:

* ``erdos_renyi`` -- an Erdos-Renyi graph where the edge probability equals the
  requested ``density``.
* ``random_regular`` -- a random regular graph generated via
  :func:`networkx.random_regular_graph` with degree ``int(density * n)``.

All randomness is driven by :class:`numpy.random.RandomState` seeded explicitly
so that every instance is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx


@dataclass
class MWISInstance:
    """A single MWIS problem instance.

    Attributes
    ----------
    instance_id:
        Human readable identifier for the instance.
    n:
        Number of vertices in the graph.
    density:
        Graph density parameter. For Erdos-Renyi this is the edge probability,
        for random regular graphs it is the fraction ``degree / n``.
    seed:
        Seed used to generate the instance, kept for reproducibility.
    adjacency:
        Symmetric ``n x n`` binary adjacency matrix with zeros on the diagonal.
    weights:
        Length-``n`` array of positive vertex weights.
    optimal_solution:
        Binary array of length ``n`` describing the optimal independent set,
        or ``None`` if it has not been computed yet.
    optimal_value:
        Objective value of ``optimal_solution`` or ``None``.
    graph_family:
        Name of the graph family used to generate the instance.
    hardness:
        Optional free-form hardness label assigned by the caller.
    """

    instance_id: str
    n: int
    density: float
    seed: int
    adjacency: np.ndarray
    weights: np.ndarray
    optimal_solution: Optional[np.ndarray] = None
    optimal_value: Optional[float] = None
    graph_family: str = "erdos_renyi"
    hardness: Optional[str] = None

    def __post_init__(self) -> None:
        self.adjacency = np.asarray(self.adjacency)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.adjacency.shape != (self.n, self.n):
            raise ValueError(
                f"adjacency must have shape ({self.n}, {self.n}), "
                f"got {self.adjacency.shape}"
            )
        if self.weights.shape != (self.n,):
            raise ValueError(
                f"weights must have shape ({self.n},), got {self.weights.shape}"
            )


def _erdos_renyi_adjacency(n: int, density: float, rng: np.random.RandomState) -> np.ndarray:
    """Build a symmetric Erdos-Renyi adjacency matrix.

    Only the strict upper triangle is sampled and then mirrored to the lower
    triangle, which guarantees a symmetric matrix with zero diagonal.
    """
    adj = np.zeros((n, n), dtype=np.int8)
    if n < 2:
        return adj
    upper = rng.random((n, n)) < density
    upper = np.triu(upper, k=1)
    adj = (upper | upper.T).astype(np.int8)
    return adj


def _random_regular_adjacency(n: int, density: float, rng: np.random.RandomState) -> np.ndarray:
    """Build a random regular graph adjacency matrix.

    The degree is ``int(density * n)``. If the requested degree is impossible
    (e.g. ``degree >= n`` or ``degree * n`` odd) we fall back to the closest
    feasible degree.
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=np.int8)
    degree = int(round(density * n))
    degree = max(0, min(degree, n - 1))
    # n * degree must be even for a regular graph to exist.
    if (n * degree) % 2 != 0:
        degree = max(0, degree - 1)
    if degree == 0:
        return np.zeros((n, n), dtype=np.int8)
    # networkx.random_regular_graph accepts a seed; we pass our RandomState so
    # that the generation is deterministic and tied to ``rng``.
    graph = nx.random_regular_graph(degree, n, seed=rng)
    adj = nx.to_numpy_array(graph, dtype=np.int8)
    # Ensure exact symmetry and zero diagonal (networkx already does this, but
    # we enforce it defensively).
    adj = np.triu(adj, k=1)
    adj = (adj | adj.T).astype(np.int8)
    return adj


def generate_mwis_instance(
    n: int,
    density: float,
    seed: int,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    graph_family: str = "erdos_renyi",
) -> MWISInstance:
    """Generate a reproducible random MWIS instance.

    Parameters
    ----------
    n:
        Number of vertices.
    density:
        Edge probability for Erdos-Renyi, or ``degree / n`` for random regular
        graphs.
    seed:
        Seed for :class:`numpy.random.RandomState`.
    weight_range:
        Inclusive range for the uniform vertex weights.
    graph_family:
        Either ``"erdos_renyi"`` or ``"random_regular"``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must lie in [0, 1]")
    if weight_range[0] > weight_range[1]:
        raise ValueError("weight_range must be (low, high) with low <= high")

    rng = np.random.RandomState(seed)

    if graph_family == "erdos_renyi":
        adjacency = _erdos_renyi_adjacency(n, density, rng)
    elif graph_family == "random_regular":
        adjacency = _random_regular_adjacency(n, density, rng)
    else:
        raise ValueError(f"unknown graph_family: {graph_family!r}")

    low, high = weight_range
    if n > 0:
        weights = rng.uniform(low, high, size=n).astype(float)
    else:
        weights = np.zeros(0, dtype=float)

    instance_id = f"mwis_{graph_family}_n{n}_d{density:.4f}_s{seed}"

    return MWISInstance(
        instance_id=instance_id,
        n=n,
        density=density,
        seed=seed,
        adjacency=adjacency,
        weights=weights,
        graph_family=graph_family,
    )


def is_independent(subset: np.ndarray, adjacency: np.ndarray) -> bool:
    """Return ``True`` if ``subset`` is an independent set of ``adjacency``.

    ``subset`` is interpreted as a binary array where ``subset[i] == 1`` means
    vertex ``i`` is selected. An independent set has no edge between any two
    selected vertices.
    """
    subset = np.asarray(subset)
    adjacency = np.asarray(adjacency)
    if subset.ndim != 1:
        raise ValueError("subset must be a 1D binary array")
    if subset.shape[0] != adjacency.shape[0]:
        raise ValueError("subset and adjacency have mismatched dimensions")
    selected = subset.astype(bool)
    # Submatrix of selected vertices; any non-zero entry means an internal edge.
    sub = adjacency[np.ix_(selected, selected)]
    # The diagonal of ``adjacency`` is zero, so any positive entry off the
    # diagonal indicates a conflict.
    return not np.any(sub.astype(bool))


def mwis_value(subset: np.ndarray, weights: np.ndarray) -> float:
    """Compute the MWIS objective value ``sum(weights[i] * subset[i])``."""
    subset = np.asarray(subset)
    weights = np.asarray(weights, dtype=float)
    if subset.shape != weights.shape:
        raise ValueError("subset and weights must have the same shape")
    return float(np.dot(weights, subset.astype(float)))


def generate_instance_set(
    sizes: Sequence[int],
    densities: Sequence[float],
    n_instances: int,
    base_seed: int,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    graph_family: str = "erdos_renyi",
) -> List[MWISInstance]:
    """Generate a list of :class:`MWISInstance` objects.

    For every ``(size, density)`` combination, ``n_instances`` instances are
    generated with seeds ``base_seed, base_seed + 1, ...``. Seeds are allocated
    deterministically across the grid so that re-running with the same
    ``base_seed`` reproduces the exact same set.
    """
    instances: List[MWISInstance] = []
    seed_counter = int(base_seed)
    for n in sizes:
        for density in densities:
            for _ in range(n_instances):
                instances.append(
                    generate_mwis_instance(
                        n=n,
                        density=density,
                        seed=seed_counter,
                        weight_range=weight_range,
                        graph_family=graph_family,
                    )
                )
                seed_counter += 1
    return instances


def edges_from_adjacency(adjacency: np.ndarray) -> List[Tuple[int, int]]:
    """Return the list of edges ``(u, v)`` with ``u < v`` from an adjacency matrix."""
    adjacency = np.asarray(adjacency)
    rows, cols = np.where(np.triu(adjacency.astype(bool), k=1))
    return list(zip(rows.tolist(), cols.tolist()))


if __name__ == "__main__":
    # Quick smoke test when run directly.
    inst = generate_mwis_instance(n=8, density=0.5, seed=42)
    print(f"instance_id: {inst.instance_id}")
    print(f"adjacency shape: {inst.adjacency.shape}, edges: {int(np.sum(inst.adjacency) // 2)}")
    print(f"weights: {np.round(inst.weights, 2)}")
    subset = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    print(f"is_independent({subset}): {is_independent(subset, inst.adjacency)}")
    print(f"mwis_value: {mwis_value(subset, inst.weights):.3f}")

    inst2 = generate_mwis_instance(n=10, density=0.3, seed=7, graph_family="random_regular")
    degrees = inst2.adjacency.sum(axis=1)
    print(f"random_regular degrees: {np.unique(degrees)}")

    batch = generate_instance_set(sizes=[6, 8], densities=[0.3, 0.5], n_instances=2, base_seed=100)
    print(f"generated {len(batch)} instances, ids: {[i.instance_id for i in batch]}")
