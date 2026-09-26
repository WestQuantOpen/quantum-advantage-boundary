"""Exact solvers for the Maximum Weight Independent Set (MWIS) problem.

Two solvers are provided:

* :func:`solve_mwis_exact` -- an ILP solved with :func:`scipy.optimize.milp`.
* :func:`solve_mwis_bruteforce` -- exhaustive enumeration over all bitstrings,
  used as a fallback / verification for small instances (``n <= 20``).

The ILP formulation is the standard MWIS formulation:

    maximize    sum_i w_i * x_i
    subject to  x_u + x_v <= 1   for every edge (u, v)
                x_i in {0, 1}

``scipy.optimize.milp`` minimizes by default, so we negate the objective.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from src.problems import MWISInstance, edges_from_adjacency, is_independent, mwis_value


def _edge_constraint_matrix(instance: MWISInstance) -> csr_matrix:
    """Build the sparse edge constraint matrix ``x_u + x_v <= 1``.

    Returns a sparse matrix with one row per edge and ``n`` columns.
    """
    edges = edges_from_adjacency(instance.adjacency)
    n = instance.n
    if not edges:
        return csr_matrix((0, n), dtype=float)
    rows = []
    cols = []
    data = []
    for row_idx, (u, v) in enumerate(edges):
        rows.append(row_idx)
        cols.append(u)
        data.append(1.0)
        rows.append(row_idx)
        cols.append(v)
        data.append(1.0)
    return csr_matrix((data, (rows, cols)), shape=(len(edges), n), dtype=float)


def solve_mwis_exact(instance: MWISInstance) -> Tuple[np.ndarray, float, float]:
    """Solve an MWIS instance exactly via integer linear programming.

    Parameters
    ----------
    instance:
        The :class:`~problems.MWISInstance` to solve.

    Returns
    -------
    (optimal_subset, optimal_value, runtime_seconds):
        ``optimal_subset`` is a binary ``np.ndarray`` of length ``n``,
        ``optimal_value`` is the objective value, and ``runtime_seconds`` is
        the wall-clock time spent solving.
    """
    n = instance.n
    if n == 0:
        return np.zeros(0, dtype=int), 0.0, 0.0

    start = time.perf_counter()

    weights = instance.weights.astype(float)
    # milp minimizes, so we minimize -sum(w_i * x_i).
    c = -weights

    # Edge constraints: A_ub @ x <= 1.
    A_ub = _edge_constraint_matrix(instance)
    if A_ub.shape[0] > 0:
        b_ub = np.ones(A_ub.shape[0], dtype=float)
        constraints = [LinearConstraint(A_ub, ub=b_ub)]
    else:
        constraints = []

    integrality = np.ones(n, dtype=int)  # all variables are integers
    bounds = Bounds(lb=0, ub=1)  # x_i in {0, 1}

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False},
    )

    runtime = time.perf_counter() - start

    if result.x is None or not result.success:
        # No solution found -- return the empty set as a safe fallback.
        return np.zeros(n, dtype=int), 0.0, runtime

    x = np.round(result.x).astype(int)
    # Defensive: clip to {0, 1} in case of tiny numerical drift.
    x = np.clip(x, 0, 1)
    value = mwis_value(x, instance.weights)
    return x, value, runtime


def solve_mwis_bruteforce(instance: MWISInstance) -> Tuple[np.ndarray, float, float]:
    """Solve an MWIS instance by exhaustive enumeration.

    Only suitable for ``n <= 20`` (otherwise the search is exponential). For
    larger ``n`` a :class:`ValueError` is raised.
    """
    n = instance.n
    if n > 20:
        raise ValueError(
            f"brute force is only supported for n <= 20, got n={n}"
        )
    if n == 0:
        return np.zeros(0, dtype=int), 0.0, 0.0

    start = time.perf_counter()

    best_value = -np.inf
    best_subset = np.zeros(n, dtype=int)
    adjacency = instance.adjacency
    weights = instance.weights.astype(float)

    # Iterate over all 2^n bitstrings. We use a Gray-code-free integer loop;
    # for n <= 20 this is at most ~1M iterations which is fast enough.
    for code in range(1 << n):
        subset = np.array([(code >> i) & 1 for i in range(n)], dtype=np.int8)
        if not is_independent(subset, adjacency):
            continue
        value = float(np.dot(weights, subset.astype(float)))
        if value > best_value:
            best_value = value
            best_subset = subset.copy()

    runtime = time.perf_counter() - start
    if best_value == -np.inf:
        best_value = 0.0
        best_subset = np.zeros(n, dtype=int)
    return best_subset.astype(int), float(best_value), runtime


def _main() -> None:
    """Compare the ILP and brute force solvers on small instances."""
    from problems import generate_mwis_instance

    print("Comparing exact ILP solver vs brute force on small instances:")
    print("-" * 70)
    for n in [5, 8, 10, 12]:
        for density in [0.3, 0.5, 0.7]:
            inst = generate_mwis_instance(n=n, density=density, seed=1234)
            x_ilp, val_ilp, t_ilp = solve_mwis_exact(inst)
            x_bf, val_bf, t_bf = solve_mwis_bruteforce(inst)
            match = np.isclose(val_ilp, val_bf)
            print(
                f"n={n:2d} d={density:.1f} | ILP val={val_ilp:7.3f} ({t_ilp*1e3:7.3f} ms) | "
                f"BF val={val_bf:7.3f} ({t_bf*1e3:7.3f} ms) | match={match}"
            )
    print("-" * 70)
    print("All comparisons complete.")


if __name__ == "__main__":
    _main()
