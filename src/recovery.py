"""Classical recovery pipeline for MWIS candidate bitstrings.

The recovery pipeline takes a set of raw candidate bitstrings (from a quantum
or classical sampler) and tries to improve them into feasible, high-value
independent sets through a sequence of classical post-processing stages:

1. **Repair** -- fix constraint violations by removing lower-weight conflicting
   vertices, then greedily add non-conflicting vertices.
2. **1-flip local search** -- add or remove a single vertex if it improves the
   objective while maintaining feasibility.
3. **2-flip local search** -- swap a selected vertex for an unselected one (or
   add two / remove two) when it improves the objective.
4. **Restricted solve** -- solve a small ILP over the union of vertices touched
   by the candidates (plus a small neighbourhood) to find the best independent
   set within that restricted subspace.

A global ``max_candidates`` budget caps the total number of candidate
subsets processed across all stages; once the budget is reached the pipeline
stops and returns the best result found so far.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.problems import MWISInstance, edges_from_adjacency, is_independent, mwis_value


def _repair(subset: np.ndarray, instance: MWISInstance) -> np.ndarray:
    """Repair a bitstring into a feasible independent set.

    Conflicting vertices are resolved greedily: for each violated edge we drop
    the lower-weight endpoint (keeping the more valuable vertex). After all
    conflicts are removed we greedily add any remaining vertex that does not
    conflict with the current set, in descending weight order.
    """
    n = instance.n
    if n == 0:
        return np.zeros(0, dtype=int)
    adjacency = instance.adjacency
    weights = instance.weights

    current = subset.astype(int).copy()
    selected = current.astype(bool).copy()

    # Resolve conflicts: repeatedly find a violated edge and drop the
    # lower-weight endpoint. We iterate until no conflicts remain.
    changed = True
    while changed:
        changed = False
        for u, v in edges_from_adjacency(adjacency):
            if selected[u] and selected[v]:
                # Drop the lower-weight endpoint; ties broken by index.
                if weights[u] < weights[v] or (weights[u] == weights[v] and u > v):
                    selected[u] = False
                else:
                    selected[v] = False
                changed = True
        # ``edges_from_adjacency`` is recomputed each pass; break early if
        # nothing changed to avoid an infinite loop.
        if not changed:
            break

    # Greedy fill: add vertices in descending weight order if they do not
    # conflict with the current set.
    order = np.argsort(-weights)
    for v in order:
        if selected[v]:
            continue
        neighbours = adjacency[v].astype(bool)
        if np.any(selected & neighbours):
            continue
        selected[v] = True

    return selected.astype(int)


def _one_flip(
    subset: np.ndarray, instance: MWISInstance
) -> Tuple[np.ndarray, float]:
    """Apply 1-flip local search (hill climb) until no improvement is found."""
    n = instance.n
    adjacency = instance.adjacency
    weights = instance.weights
    current = subset.astype(int).copy()
    current_val = mwis_value(current, weights)

    improved = True
    while improved:
        improved = False
        for v in range(n):
            candidate = current.copy()
            candidate[v] = 1 - candidate[v]
            if not is_independent(candidate, adjacency):
                continue
            cand_val = mwis_value(candidate, weights)
            if cand_val > current_val + 1e-12:
                current = candidate
                current_val = cand_val
                improved = True
    return current, current_val


def _two_flip(
    subset: np.ndarray, instance: MWISInstance
) -> Tuple[np.ndarray, float]:
    """Apply 2-flip local search (swap pairs) until no improvement is found."""
    n = instance.n
    adjacency = instance.adjacency
    weights = instance.weights
    current = subset.astype(int).copy()
    current_val = mwis_value(current, weights)

    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                candidate = current.copy()
                # Swap bits i and j (toggle both).
                candidate[i] = 1 - candidate[i]
                candidate[j] = 1 - candidate[j]
                if not is_independent(candidate, adjacency):
                    continue
                cand_val = mwis_value(candidate, weights)
                if cand_val > current_val + 1e-12:
                    current = candidate
                    current_val = cand_val
                    improved = True
    return current, current_val


def _restricted_solve(
    instance: MWISInstance, allowed_vertices: Set[int]
) -> Tuple[np.ndarray, float]:
    """Solve MWIS exactly restricted to ``allowed_vertices``.

    Vertices outside ``allowed_vertices`` are forced to 0. This is much cheaper
    than solving the full instance when ``allowed_vertices`` is small.
    """
    n = instance.n
    if n == 0 or not allowed_vertices:
        return np.zeros(n, dtype=int), 0.0

    # Build a sub-instance over the allowed vertices.
    allowed = sorted(allowed_vertices)
    idx_map = {v: i for i, v in enumerate(allowed)}
    sub_n = len(allowed)
    sub_adj = np.zeros((sub_n, sub_n), dtype=np.int8)
    for u, v in edges_from_adjacency(instance.adjacency):
        if u in idx_map and v in idx_map:
            i, j = idx_map[u], idx_map[v]
            sub_adj[i, j] = 1
            sub_adj[j, i] = 1
    sub_weights = instance.weights[allowed].copy()

    # Solve the sub-instance with brute force (sub_n is small by construction).
    best_val = -np.inf
    best_sub = np.zeros(sub_n, dtype=int)
    if sub_n <= 22:
        for code in range(1 << sub_n):
            sub = np.array([(code >> i) & 1 for i in range(sub_n)], dtype=np.int8)
            if not is_independent(sub, sub_adj):
                continue
            val = float(np.dot(sub_weights, sub.astype(float)))
            if val > best_val:
                best_val = val
                best_sub = sub.copy()
    else:
        # Fall back to greedy + local search if the subspace is too large.
        best_sub = _repair(np.ones(sub_n, dtype=int), type("S", (), {
            "n": sub_n, "adjacency": sub_adj, "weights": sub_weights
        })())
        best_val = float(np.dot(sub_weights, best_sub.astype(float)))

    full = np.zeros(n, dtype=int)
    for i, v in enumerate(allowed):
        full[v] = best_sub[i]
    if best_val == -np.inf:
        best_val = 0.0
    return full, float(best_val)


def recover(
    samples: List[np.ndarray],
    instance: MWISInstance,
    max_expansion: int = 2,
    max_candidates: int = 50000,
) -> Dict[str, Any]:
    """Apply the full recovery pipeline to a list of candidate bitstrings.

    Parameters
    ----------
    samples:
        List of raw binary candidate bitstrings (length ``instance.n`` each).
    instance:
        The MWIS instance the candidates refer to.
    max_expansion:
        Neighbourhood expansion radius used to build the restricted subspace
        for the final restricted-solve stage.
    max_candidates:
        Hard cap on the total number of candidate subsets processed across all
        stages. Once reached the pipeline stops and returns the best result.

    Returns
    -------
    dict
        Dictionary with keys: ``pre_recovery_best``, ``post_recovery_best``,
        ``optimality_gap``, ``n_candidates``, ``n_repaired``,
        ``recovery_time``, ``best_subset``.
    """
    start = time.perf_counter()
    n = instance.n
    weights = instance.weights
    adjacency = instance.adjacency

    if not samples:
        return {
            "pre_recovery_best": 0.0,
            "post_recovery_best": 0.0,
            "optimality_gap": 1.0,
            "n_candidates": 0,
            "n_repaired": 0,
            "recovery_time": 0.0,
            "best_subset": np.zeros(n, dtype=int),
        }

    candidates_processed = 0
    n_repaired = 0

    # Pre-recovery best (raw objective, ignoring feasibility).
    pre_values = [mwis_value(s, weights) for s in samples]
    pre_recovery_best = max(pre_values) if pre_values else 0.0

    best_subset = np.zeros(n, dtype=int)
    best_value = -np.inf

    # Stage 1: repair every sample.
    repaired: List[np.ndarray] = []
    for s in samples:
        if candidates_processed >= max_candidates:
            break
        r = _repair(s, instance)
        repaired.append(r)
        n_repaired += 1
        candidates_processed += 1
        val = mwis_value(r, weights)
        if val > best_value:
            best_value = val
            best_subset = r.copy()

    # Stage 2 & 3: 1-flip then 2-flip local search on each repaired candidate.
    locally_optimized: List[np.ndarray] = []
    for r in repaired:
        if candidates_processed >= max_candidates:
            break
        s1, v1 = _one_flip(r, instance)
        candidates_processed += 1
        s2, v2 = _two_flip(s1, instance)
        candidates_processed += 1
        locally_optimized.append(s2)
        if v2 > best_value:
            best_value = v2
            best_subset = s2.copy()

    # Stage 4: restricted solve over the union of vertices appearing in the
    # locally optimized candidates, expanded by ``max_expansion`` hops.
    touched: Set[int] = set()
    for s in locally_optimized:
        for v in range(n):
            if s[v] == 1:
                touched.add(v)
    # Expand by neighbourhood hops.
    frontier = set(touched)
    for _ in range(max_expansion):
        next_frontier = set(frontier)
        for v in frontier:
            neighbours = np.where(adjacency[v].astype(bool))[0]
            for nb in neighbours:
                next_frontier.add(int(nb))
        if next_frontier == frontier:
            break
        frontier = next_frontier
    allowed_vertices = frontier

    if candidates_processed < max_candidates and allowed_vertices:
        rs_subset, rs_val = _restricted_solve(instance, allowed_vertices)
        candidates_processed += 1
        if rs_val > best_value:
            best_value = rs_val
            best_subset = rs_subset.copy()

    recovery_time = time.perf_counter() - start

    post_recovery_best = float(best_value) if best_value != -np.inf else 0.0

    # Optimality gap relative to the known optimum if available.
    if instance.optimal_value is not None and instance.optimal_value > 0:
        optimality_gap = 1.0 - post_recovery_best / instance.optimal_value
    else:
        optimality_gap = float("nan")

    return {
        "pre_recovery_best": float(pre_recovery_best),
        "post_recovery_best": post_recovery_best,
        "optimality_gap": float(optimality_gap),
        "n_candidates": int(candidates_processed),
        "n_repaired": int(n_repaired),
        "recovery_time": float(recovery_time),
        "best_subset": best_subset.astype(int),
    }


if __name__ == "__main__":
    from problems import generate_mwis_instance
    from classical_samplers import UniformRandomSampler, GreedySampler
    from exact_solver import solve_mwis_exact

    inst = generate_mwis_instance(n=14, density=0.4, seed=55)
    opt_subset, opt_val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt_subset
    inst.optimal_value = opt_val
    print(f"instance: {inst.instance_id}, optimal value: {opt_val:.3f}")

    for sampler in [UniformRandomSampler(), GreedySampler()]:
        samples, _ = sampler.sample(inst, n_shots=50, seed=1)
        result = recover(samples, inst, max_expansion=2, max_candidates=10000)
        print(
            f"{sampler.name:18s} | pre={result['pre_recovery_best']:7.3f} | "
            f"post={result['post_recovery_best']:7.3f} | "
            f"gap={result['optimality_gap']:.4f} | "
            f"cands={result['n_candidates']} | repaired={result['n_repaired']} | "
            f"time={result['recovery_time']*1e3:.2f} ms"
        )
