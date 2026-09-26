"""QCSC: Classical decomposition + quantum hard-core solving.

Instead of solving the full problem on quantum, decompose:
  G (N_global=500) -> classical decomposition -> H_hard (32) -> quantum sampler
  -> reconstruction

Pipeline:
  1. Classical preprocessing fixes easy variables (greedy, LP relaxation)
  2. AI identifies uncertain hard core
  3. Quantum works only on H_hard
  4. Result recombined with fixed variables
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.problems import (
    MWISInstance,
    edges_from_adjacency,
    is_independent,
    mwis_value,
)
from src.exact_solver import solve_mwis_exact
from src.recovery import recover


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecompositionResult:
    """Result of decomposing a large MWIS instance into a fixed part and a core.

    Attributes
    ----------
    fixed_vars:
        Dict mapping vertex index -> 0 or 1 for variables fixed by the
        decomposition.
    core_instance:
        The sub-instance (``MWISInstance``) over the uncertain variables.
    core_vertices:
        List of original vertex indices that form the core.
    mapping:
        Dict mapping core vertex index (in ``core_instance``) -> original
        vertex index.
    fixed_value:
        Objective contribution of the fixed variables.
    """

    fixed_vars: Dict[int, int]
    core_instance: Optional[MWISInstance]
    core_vertices: List[int]
    mapping: Dict[int, int]
    fixed_value: float


@dataclass
class QCSCResult:
    """Result of the full QCSC pipeline.

    Attributes
    ----------
    best_subset:
        Full binary solution (length ``n_global``).
    best_value:
        Objective value of ``best_subset``.
    core_samples:
        Samples produced for the core subproblem.
    core_recovery:
        Recovery result for the core.
    decomposition:
        The :class:`DecompositionResult`.
    metadata:
        Additional diagnostics.
    """

    best_subset: np.ndarray
    best_value: float
    core_samples: List[np.ndarray] = field(default_factory=list)
    core_recovery: Dict[str, Any] = field(default_factory=dict)
    decomposition: Optional[DecompositionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _greedy_fix(
    instance: MWISInstance, fixed: Dict[int, int]
) -> Tuple[Dict[int, int], float]:
    """Greedily fix variables that are clearly 0 or 1.

    A vertex is fixed to 1 (selected) if it has the highest weight among its
    neighbourhood and none of its neighbours are already fixed to 1.  A vertex
    is fixed to 0 if one of its neighbours is fixed to 1.

    Returns the updated ``fixed`` dict and the objective value of fixed-to-1
    variables.
    """
    n = instance.n
    adjacency = instance.adjacency
    weights = instance.weights

    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in fixed:
                continue
            neighbours = np.where(adjacency[v].astype(bool))[0]
            # If any neighbour is fixed to 1, fix v to 0.
            if any(int(fixed.get(int(u), -1)) == 1 for u in neighbours):
                fixed[v] = 0
                changed = True
                continue
            # If v has the strictly highest weight among unfixed neighbours and
            # all neighbours are unfixed, tentatively fix it to 1.
            unfixed_neighbours = [int(u) for u in neighbours if u not in fixed]
            if not unfixed_neighbours:
                # All neighbours fixed to 0 -> fix v to 1.
                if all(int(fixed.get(int(u), -1)) == 0 for u in neighbours):
                    fixed[v] = 1
                    changed = True
                continue
            # Fix v to 1 only if it strictly dominates all unfixed neighbours
            # by a margin AND has no fixed-to-1 neighbours (already checked).
            if weights[v] > np.max(weights[unfixed_neighbours]) + 1e-9:
                # Only fix if v's weight is much higher (strong signal).
                fixed[v] = 1
                # Fix all neighbours to 0.
                for u in neighbours:
                    if int(u) not in fixed:
                        fixed[int(u)] = 0
                changed = True

    fixed_value = float(
        sum(instance.weights[v] for v, val in fixed.items() if val == 1)
    )
    return fixed, fixed_value


def _lp_decompose(
    instance: MWISInstance
) -> Tuple[Dict[int, int], float]:
    """Use the LP relaxation to identify fixable variables.

    Solves the LP relaxation of the MWIS ILP.  Variables with LP value close
    to 0 or 1 are fixed to the corresponding integer value.

    Returns the ``fixed`` dict and the objective value of fixed-to-1 variables.
    """
    n = instance.n
    if n == 0:
        return {}, 0.0

    weights = instance.weights.astype(float)
    edges = edges_from_adjacency(instance.adjacency)

    # LP: maximize sum(w_i x_i)  =>  minimize -sum(w_i x_i)
    c = -weights

    # Edge constraints: x_u + x_v <= 1.
    if edges:
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
        A_ub = csr_matrix((data, (rows, cols)), shape=(len(edges), n), dtype=float)
        b_ub = np.ones(len(edges), dtype=float)
    else:
        A_ub = csr_matrix((0, n), dtype=float)
        b_ub = np.zeros(0, dtype=float)

    bounds = [(0.0, 1.0) for _ in range(n)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    fixed: Dict[int, int] = {}
    if result.x is None:
        return fixed, 0.0

    x = result.x
    tol = 0.01  # fix variables that are within tol of 0 or 1
    for i in range(n):
        if x[i] <= tol:
            fixed[i] = 0
        elif x[i] >= 1.0 - tol:
            fixed[i] = 1

    # Propagate: if a vertex is fixed to 1, all its neighbours must be 0.
    adjacency = instance.adjacency
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if fixed.get(v, -1) == 1:
                neighbours = np.where(adjacency[v].astype(bool))[0]
                for u in neighbours:
                    if int(u) not in fixed:
                        fixed[int(u)] = 0
                        changed = True

    fixed_value = float(
        sum(instance.weights[v] for v, val in fixed.items() if val == 1)
    )
    return fixed, fixed_value


def _build_core_instance(
    instance: MWISInstance,
    fixed: Dict[int, int],
) -> Tuple[Optional[MWISInstance], List[int], Dict[int, int], float]:
    """Build the core sub-instance over unfixed variables.

    Returns
    -------
    (core_instance, core_vertices, mapping, fixed_value)
        ``mapping`` maps core index -> original vertex index.
    """
    n = instance.n
    core_vertices = sorted(v for v in range(n) if v not in fixed)
    if not core_vertices:
        return None, [], {}, float(
            sum(instance.weights[v] for v, val in fixed.items() if val == 1)
        )

    idx_map = {v: i for i, v in enumerate(core_vertices)}
    sub_n = len(core_vertices)
    sub_adj = np.zeros((sub_n, sub_n), dtype=np.int8)
    for u, v in edges_from_adjacency(instance.adjacency):
        if u in idx_map and v in idx_map:
            i, j = idx_map[u], idx_map[v]
            sub_adj[i, j] = 1
            sub_adj[j, i] = 1
    sub_weights = instance.weights[core_vertices].copy()

    core_instance = MWISInstance(
        instance_id=instance.instance_id + "_core",
        n=sub_n,
        density=instance.density,
        seed=instance.seed,
        adjacency=sub_adj,
        weights=sub_weights,
        optimal_solution=None,
        optimal_value=None,
        graph_family=instance.graph_family,
        hardness=instance.hardness,
    )

    fixed_value = float(
        sum(instance.weights[v] for v, val in fixed.items() if val == 1)
    )
    mapping = {i: core_vertices[i] for i in range(sub_n)}
    return core_instance, core_vertices, mapping, fixed_value


def decompose_instance(
    instance: MWISInstance,
    target_core_size: int = 32,
) -> DecompositionResult:
    """Decompose an MWIS instance using LP relaxation + greedy fixing.

    Variables that are clearly 0 or 1 (from the LP relaxation and greedy
    propagation) are fixed.  The remaining uncertain variables form the
    ``core`` sub-instance, which is small enough to be solved on a quantum
    device.

    If the core is larger than ``target_core_size``, additional greedy fixing
    is applied (fixing the highest-confidence variables) until the core fits.

    Parameters
    ----------
    instance:
        The MWIS instance to decompose.
    target_core_size:
        Maximum number of variables in the core sub-instance.

    Returns
    -------
    DecompositionResult
    """
    n = instance.n
    if n == 0:
        return DecompositionResult(
            fixed_vars={}, core_instance=None, core_vertices=[],
            mapping={}, fixed_value=0.0,
        )

    # Step 1: LP relaxation.
    fixed, fixed_value = _lp_decompose(instance)

    # Step 2: Greedy propagation.
    fixed, fixed_value = _greedy_fix(instance, fixed)

    # Step 3: Build core and check size.
    core_instance, core_vertices, mapping, fixed_value = _build_core_instance(
        instance, fixed
    )

    # If core is still too large, fix more variables greedily by weight.
    if core_instance is not None and core_instance.n > target_core_size:
        # Fix the highest-weight unfixed vertices to 1 (and their neighbours to 0)
        # until the core is small enough.
        unfixed = sorted(
            (v for v in range(n) if v not in fixed),
            key=lambda v: -instance.weights[v],
        )
        for v in unfixed:
            if core_instance is None or core_instance.n <= target_core_size:
                break
            if v in fixed:
                continue
            fixed[v] = 1
            neighbours = np.where(instance.adjacency[v].astype(bool))[0]
            for u in neighbours:
                if int(u) not in fixed:
                    fixed[int(u)] = 0
            core_instance, core_vertices, mapping, fixed_value = _build_core_instance(
                instance, fixed
            )

    return DecompositionResult(
        fixed_vars=fixed,
        core_instance=core_instance,
        core_vertices=core_vertices,
        mapping=mapping,
        fixed_value=fixed_value,
    )


def reconstruct_solution(
    core_solution: np.ndarray,
    fixed_vars: Dict[int, int],
    mapping: Dict[int, int],
    original_instance: MWISInstance,
) -> np.ndarray:
    """Combine a core solution with fixed variables into a full solution.

    Parameters
    ----------
    core_solution:
        Binary array over the core variables.
    fixed_vars:
        Dict mapping original vertex -> 0 or 1.
    mapping:
        Dict mapping core index -> original vertex index.
    original_instance:
        The original MWIS instance (used for ``n``).

    Returns
    -------
    np.ndarray
        Full binary solution of length ``original_instance.n``.
    """
    n = original_instance.n
    full = np.zeros(n, dtype=int)
    for v, val in fixed_vars.items():
        full[v] = int(val)
    for i, val in enumerate(core_solution):
        original_v = mapping.get(i, i)
        full[original_v] = int(val)
    return full


# ---------------------------------------------------------------------------
# QCSC pipeline
# ---------------------------------------------------------------------------

class QCSCPipeline:
    """Full QCSC pipeline: decompose -> sample core -> reconstruct -> recover.

    Parameters
    ----------
    n_global:
        Expected global problem size (for bookkeeping).
    target_core:
        Target core sub-instance size (number of qubits).
    """

    def __init__(self, n_global: int = 500, target_core: int = 32) -> None:
        if target_core < 1:
            raise ValueError("target_core must be >= 1")
        self.n_global = n_global
        self.target_core = target_core

    def _lp_decompose(self, instance: MWISInstance) -> Tuple[Dict[int, int], float]:
        """LP-relaxation based decomposition."""
        return _lp_decompose(instance)

    def _greedy_decompose(self, instance: MWISInstance) -> Tuple[Dict[int, int], float]:
        """Greedy fixing of easy variables."""
        fixed: Dict[int, int] = {}
        fixed, fixed_value = _greedy_fix(instance, fixed)
        return fixed, fixed_value

    def _identify_hard_core(
        self, instance: MWISInstance, fixed_vars: Dict[int, int]
    ) -> Tuple[Optional[MWISInstance], List[int], Dict[int, int]]:
        """Find the uncertain subproblem (core) given fixed variables.

        Returns
        -------
        (core_instance, core_vertices, mapping)
        """
        core_instance, core_vertices, mapping, _ = _build_core_instance(
            instance, fixed_vars
        )
        return core_instance, core_vertices, mapping

    def solve(
        self,
        instance: MWISInstance,
        quantum_sampler: Optional[Any] = None,
        n_shots: int = 128,
    ) -> QCSCResult:
        """Run the full QCSC pipeline.

        Parameters
        ----------
        instance:
            The (potentially large) MWIS instance to solve.
        quantum_sampler:
            An object with a ``sample(instance, n_shots, seed)`` method that
            returns ``(samples, metadata)``.  If ``None``, the exact solver is
            used on the core (classical fallback).
        n_shots:
            Number of shots for the quantum sampler.

        Returns
        -------
        QCSCResult
        """
        start = time.perf_counter()

        # Step 1: decompose.
        decomp = decompose_instance(instance, target_core_size=self.target_core)

        # Step 2: solve the core.
        core_samples: List[np.ndarray] = []
        core_recovery: Dict[str, Any] = {}
        core_best_subset: np.ndarray = np.zeros(0, dtype=int)
        core_best_value = 0.0

        if decomp.core_instance is not None and decomp.core_instance.n > 0:
            if quantum_sampler is not None:
                core_samples, q_meta = quantum_sampler.sample(
                    decomp.core_instance, n_shots=n_shots, seed=instance.seed
                )
            else:
                # Classical fallback: exact solve on the core.
                opt_subset, opt_val, _ = solve_mwis_exact(decomp.core_instance)
                core_samples = [opt_subset]
                decomp.core_instance.optimal_solution = opt_subset
                decomp.core_instance.optimal_value = opt_val

            # Recover the core samples.
            core_recovery = recover(
                core_samples, decomp.core_instance,
                max_expansion=2, max_candidates=50000,
            )
            core_best_subset = np.asarray(core_recovery.get("best_subset", np.zeros(
                decomp.core_instance.n, dtype=int))).astype(int)
            core_best_value = float(core_recovery.get("post_recovery_best", 0.0))
        else:
            # No core: all variables were fixed.
            core_best_subset = np.zeros(0, dtype=int)

        # Step 3: reconstruct.
        full_solution = reconstruct_solution(
            core_best_subset, decomp.fixed_vars, decomp.mapping, instance
        )

        # Ensure feasibility: if the reconstructed solution has conflicts
        # (possible if fixed variables interact with core), repair.
        if not is_independent(full_solution, instance.adjacency):
            # Simple repair: drop conflicting core variables.
            repaired = full_solution.copy()
            changed = True
            while changed:
                changed = False
                for u, v in edges_from_adjacency(instance.adjacency):
                    if repaired[u] == 1 and repaired[v] == 1:
                        # Drop the lower-weight one, preferring to keep fixed vars.
                        if u in decomp.fixed_vars and v not in decomp.fixed_vars:
                            repaired[v] = 0
                        elif v in decomp.fixed_vars and u not in decomp.fixed_vars:
                            repaired[u] = 0
                        elif instance.weights[u] < instance.weights[v]:
                            repaired[u] = 0
                        else:
                            repaired[v] = 0
                        changed = True

            full_solution = repaired

        best_value = mwis_value(full_solution, instance.weights)

        elapsed = time.perf_counter() - start

        metadata: Dict[str, Any] = {
            "time": elapsed,
            "n_global": instance.n,
            "n_core": decomp.core_instance.n if decomp.core_instance else 0,
            "n_fixed": len(decomp.fixed_vars),
            "n_fixed_to_1": sum(1 for v in decomp.fixed_vars.values() if v == 1),
            "fixed_value": decomp.fixed_value,
            "core_best_value": core_best_value,
            "target_core": self.target_core,
        }

        return QCSCResult(
            best_subset=full_solution,
            best_value=best_value,
            core_samples=core_samples,
            core_recovery=core_recovery,
            decomposition=decomp,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.classical_samplers import GreedySampler

    inst = generate_mwis_instance(n=30, density=0.3, seed=42)
    opt, val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt
    inst.optimal_value = val
    print(f"instance: n={inst.n}, optimal: {val:.3f}")

    # Decompose.
    decomp = decompose_instance(inst, target_core_size=12)
    print(f"decomposition: fixed={len(decomp.fixed_vars)}, "
          f"core={decomp.core_instance.n if decomp.core_instance else 0}, "
          f"fixed_value={decomp.fixed_value:.3f}")

    # Full pipeline with classical fallback.
    pipeline = QCSCPipeline(n_global=30, target_core=12)
    result = pipeline.solve(inst, quantum_sampler=None, n_shots=64)
    print(f"QCSC result: value={result.best_value:.3f}, "
          f"gap={1 - result.best_value / val:.4f}")
    print(f"metadata: {result.metadata}")

    # Full pipeline with greedy quantum sampler.
    greedy_sampler = GreedySampler(n_restarts=3)
    result2 = pipeline.solve(inst, quantum_sampler=greedy_sampler, n_shots=64)
    print(f"QCSC (greedy sampler) result: value={result2.best_value:.3f}, "
          f"gap={1 - result2.best_value / val:.4f}")
