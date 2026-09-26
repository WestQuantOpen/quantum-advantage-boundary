"""Representation search for the quantum-advantage boundary hunter.

A *representation* is a particular way of encoding a problem instance into a
quantum circuit: choice of penalty coefficient, variable ordering, QAOA depth,
parameter initialisation, mixer family, and compilation options.

The :class:`RepresentationSearch` explores the space of representations,
evaluates each candidate (optionally on BlueQubit hardware), and returns the
Pareto frontier trading off solution quality against circuit cost.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import structlog

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

__all__ = ["Representation", "RepresentationSearch"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Representation:
    """A single problem-to-circuit encoding candidate."""

    rep_id: str
    parent_id: str | None
    transform_name: str
    penalty_coeff: float
    variable_order: list[int]
    qaoa_depth: int
    gammas: list[float]
    betas: list[float]
    mixer_type: str
    compilation_opts: dict[str, Any]
    mathematical_equivalence: str
    circuit_metrics: dict[str, Any] = field(default_factory=dict)
    simulation_metrics: dict[str, Any] | None = None
    recovery_metrics: dict[str, Any] | None = None

    # Convenience accessors used by the Pareto logic -----------------------
    @property
    def approximation_ratio(self) -> float:
        sm = self.simulation_metrics or {}
        return float(sm.get("approximation_ratio", 0.0))

    @property
    def depth(self) -> int:
        return int(self.circuit_metrics.get("depth", 0))

    @property
    def cost(self) -> float:
        sm = self.simulation_metrics or {}
        return float(sm.get("cost", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_graph(instance: Any) -> Any:
    """Return a networkx Graph from a flexible ``instance`` argument."""
    if nx is not None and isinstance(instance, nx.Graph):
        return instance
    if isinstance(instance, dict) and "graph" in instance:
        return instance["graph"]
    if isinstance(instance, dict) and "edges" in instance:
        g = nx.Graph() if nx is not None else None
        if g is not None:
            g.add_edges_from(instance["edges"])
            return g
    # Last resort: assume it is graph-like (has .edges).
    return instance


def _maxcut_value(bitstring: str, graph: Any) -> int:
    """Cut value (number of crossing edges) for a bitstring on ``graph``."""
    edges = _graph_edges(graph)
    val = 0
    n = len(bitstring)
    for u, v in edges:
        if u < n and v < n and bitstring[u] != bitstring[v]:
            val += 1
    return val


def _graph_edges(graph: Any) -> list[tuple[int, int]]:
    """Normalise edges from a graph-like object into a list of int pairs."""
    if graph is None:
        return []
    if nx is not None and isinstance(graph, nx.Graph):
        return [(int(u), int(v)) for u, v in graph.edges()]
    if hasattr(graph, "edges"):
        return [(int(u), int(v)) for u, v in graph.edges()]
    return []


def _graph_num_nodes(graph: Any) -> int:
    if graph is None:
        return 0
    if nx is not None and isinstance(graph, nx.Graph):
        return graph.number_of_nodes()
    if hasattr(graph, "number_of_nodes"):
        return graph.number_of_nodes()
    edges = _graph_edges(graph)
    return max((max(u, v) for u, v in edges), default=-1) + 1


def _brute_force_maxcut(graph: Any) -> int:
    """Exact MaxCut value by enumeration (feasible for n <= ~22)."""
    n = _graph_num_nodes(graph)
    edges = _graph_edges(graph)
    if n == 0 or not edges:
        return 0
    if n > 22:
        # Greedy upper-bound fallback for large graphs.
        return len(edges)
    best = 0
    for mask in range(1 << n):
        bs = format(mask, f"0{n}b")
        val = _maxcut_value(bs, graph)
        if val > best:
            best = val
    return best


def _build_qaoa_circuit(
    graph: Any,
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    variable_order: Sequence[int],
    mixer_type: str,
) -> Any:
    """Build a MaxCut QAOA circuit for ``graph``.

    Parameters
    ----------
    graph:
        Graph-like object with integer-labelled nodes.
    p:
        QAOA depth (number of layers).
    gammas, betas:
        Length-``p`` parameter vectors.
    variable_order:
        Permutation mapping logical qubits to physical qubits.
    mixer_type:
        ``"x_mixer"`` (standard RX) or ``"xy_mixer"`` (pairwise XY).
    """
    from qiskit import QuantumCircuit

    n = _graph_num_nodes(graph)
    edges = _graph_edges(graph)

    # Map logical -> physical via variable_order.
    order = list(variable_order) if variable_order else list(range(n))
    if len(order) < n:
        order = list(range(n))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for layer in range(p):
        gamma = float(gammas[layer]) if layer < len(gammas) else 0.0
        beta = float(betas[layer]) if layer < len(betas) else 0.0

        # Cost unitary: exp(-i * gamma * sum Z_i Z_j)  ->  RZZ(2*gamma) per edge
        for u, v in edges:
            pu, pv = order[u] % n, order[v] % n
            if pu == pv:
                continue
            qc.rzz(2.0 * gamma, pu, pv)

        # Mixer unitary.
        if mixer_type == "xy_mixer":
            # Pairwise XY mixer: alternate RX and RXX/RYY on adjacent pairs.
            for i in range(n - 1):
                qc.rxx(2.0 * beta, i, i + 1)
                qc.ryy(2.0 * beta, i, i + 1)
            qc.rx(2.0 * beta, n - 1)
        else:  # "x_mixer" (default)
            for i in range(n):
                qc.rx(2.0 * beta, i)

    qc.measure(range(n), range(n))
    return qc


def _run_local(circuit: Any, n_shots: int) -> dict[str, float]:
    """Run a circuit on a local simulator and return bitstring counts."""
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        from qiskit import transpile

        tqc = transpile(circuit, backend)
        job = backend.run(tqc, shots=n_shots)
        result = job.result()
        counts = result.get_counts()
    except Exception:
        # Fallback: qiskit primitives.
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler()
        job = sampler.run([circuit], shots=n_shots)
        counts = job.result()[0].data.c.get_counts()

    # Normalise keys (strip spaces) and convert to float probabilities.
    normalised: dict[str, float] = {}
    total = sum(counts.values())
    if total == 0:
        return normalised
    for k, v in counts.items():
        key = k.replace(" ", "")
        normalised[key] = float(v) / total
    return normalised


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class RepresentationSearch:
    """Explore and evaluate representation candidates for a problem instance."""

    def __init__(self, instance: Any, max_candidates: int = 50) -> None:
        self.instance = instance
        self.graph = _extract_graph(instance)
        self.max_candidates = max_candidates
        self._candidates: list[Representation] = []
        self._evaluated: list[Representation] = []
        self._max_cut: int | None = None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _generate_candidates(self) -> list[Representation]:
        """Generate representation candidates by varying key knobs."""
        n = _graph_num_nodes(self.graph)
        candidates: list[Representation] = []

        penalty_coeffs = [1.0, 1.5, 2.0, 2.5, 3.0]
        depths = [1, 2, 3]
        mixer_types = ["x_mixer", "xy_mixer"]

        # Variable orderings: identity, reverse, and a couple of seeded shuffles.
        orderings: list[list[int]] = [list(range(n)), list(reversed(range(n)))]
        rng = np.random.default_rng(42)
        for _ in range(2):
            perm = list(range(n))
            rng.shuffle(perm)
            orderings.append(perm)

        # Parameter initialisations (per depth).
        param_seeds = [0, 1, 2, 3]
        counter = 0
        for p, coeff, mixer, order in itertools.product(
            depths, penalty_coeffs, mixer_types, orderings
        ):
            if counter >= self.max_candidates:
                break
            rng = np.random.default_rng(param_seeds[counter % len(param_seeds)])
            gammas = [float(rng.uniform(0, math.pi)) for _ in range(p)]
            betas = [float(rng.uniform(0, math.pi / 2)) for _ in range(p)]

            rep = Representation(
                rep_id=f"rep-{counter:04d}",
                parent_id=None,
                transform_name=f"qaoa_p{p}_{mixer}_c{coeff}",
                penalty_coeff=coeff,
                variable_order=list(order),
                qaoa_depth=p,
                gammas=gammas,
                betas=betas,
                mixer_type=mixer,
                compilation_opts={"optimization_level": 1},
                mathematical_equivalence="maxcut_qaoa",
                circuit_metrics={},
                simulation_metrics=None,
                recovery_metrics=None,
            )
            candidates.append(rep)
            counter += 1

        logger.info("representation.candidates_generated", n=len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_candidate(
        self,
        rep: Representation,
        adapter: Any | None,
        n_shots: int,
    ) -> Representation:
        """Evaluate a single representation and populate its metrics."""
        # Build circuit.
        circuit = _build_qaoa_circuit(
            self.graph,
            rep.qaoa_depth,
            rep.gammas,
            rep.betas,
            rep.variable_order,
            rep.mixer_type,
        )

        # Circuit metrics.
        try:
            rep.circuit_metrics = {
                "depth": circuit.depth(),
                "gate_count": circuit.size(),
                "n_qubits": circuit.num_qubits,
            }
        except Exception:
            rep.circuit_metrics = {"depth": 0, "gate_count": 0, "n_qubits": 0}

        # Run.
        counts: dict[str, float] | None = None
        cost = 0.0
        device = "local"
        if adapter is not None:
            try:
                result = adapter.run(
                    circuit,
                    device=adapter.select_device(rep.circuit_metrics["n_qubits"]),
                    shots=n_shots,
                    job_name=rep.rep_id,
                )
                counts = result.get("counts")
                cost = float(result.get("cost") or 0.0)
                device = result.get("device", "bluequbit")
            except Exception as exc:
                logger.warning("representation.adapter_failed", rep_id=rep.rep_id, error=repr(exc))
                counts = None

        if counts is None:
            counts = _run_local(circuit, n_shots)
            device = "local"

        # Compute solution-quality metrics.
        max_cut = self._get_max_cut()
        expected_cut = 0.0
        best_cut = 0
        optimal_count = 0.0
        for bitstring, prob in counts.items():
            cv = _maxcut_value(bitstring, self.graph)
            expected_cut += prob * cv
            if cv > best_cut:
                best_cut = cv
            if max_cut > 0 and cv >= max_cut:
                optimal_count += prob

        approx_ratio = (expected_cut / max_cut) if max_cut > 0 else 0.0
        success_prob = optimal_count if max_cut > 0 else 0.0

        rep.simulation_metrics = {
            "approximation_ratio": approx_ratio,
            "expected_cut": expected_cut,
            "best_cut": best_cut,
            "max_cut": max_cut,
            "success_probability": success_prob,
            "cost": cost,
            "device": device,
            "n_shots": n_shots,
            "counts": counts,
        }

        logger.info(
            "representation.evaluated",
            rep_id=rep.rep_id,
            approx_ratio=round(approx_ratio, 4),
            depth=rep.circuit_metrics.get("depth"),
            cost=cost,
        )
        return rep

    def _get_max_cut(self) -> int:
        if self._max_cut is None:
            self._max_cut = _brute_force_maxcut(self.graph)
        return self._max_cut

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------
    def _is_dominated(self, rep: Representation, others: Sequence[Representation]) -> bool:
        """Return True if ``rep`` is Pareto-dominated by any rep in ``others``.

        Objectives (minimise depth & cost, maximise approximation_ratio):
        ``rep`` is dominated when some other candidate is no worse on every
        objective and strictly better on at least one.
        """
        for other in others:
            if other.rep_id == rep.rep_id:
                continue
            # other >= rep on all (min-depth, min-cost, max-ratio)
            no_worse = (
                other.depth <= rep.depth
                and other.cost <= rep.cost
                and other.approximation_ratio >= rep.approximation_ratio
            )
            strictly_better = (
                other.depth < rep.depth
                or other.cost < rep.cost
                or other.approximation_ratio > rep.approximation_ratio
            )
            if no_worse and strictly_better:
                return True
        return False

    @property
    def pareto_frontier(self) -> list[Representation]:
        """Return the non-dominated evaluated representations."""
        evaluated = [r for r in self._evaluated if r.simulation_metrics is not None]
        if not evaluated:
            return []
        return [r for r in evaluated if not self._is_dominated(r, evaluated)]

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    def search(
        self,
        bluequbit_adapter: Any | None = None,
        n_shots: int = 128,
    ) -> list[Representation]:
        """Run the full search and return the Pareto frontier.

        Parameters
        ----------
        bluequbit_adapter:
            Optional :class:`BlueQubitAdapter` for hardware evaluation.  When
            ``None`` a local simulator is used.
        n_shots:
            Number of measurement shots per candidate.
        """
        self._candidates = self._generate_candidates()
        self._evaluated = []

        for rep in self._candidates:
            try:
                evaluated = self._evaluate_candidate(rep, bluequbit_adapter, n_shots)
                self._evaluated.append(evaluated)
            except Exception as exc:
                logger.error("representation.evaluation_failed", rep_id=rep.rep_id, error=repr(exc))

        frontier = self.pareto_frontier
        logger.info(
            "representation.search_complete",
            candidates=len(self._candidates),
            evaluated=len(self._evaluated),
            frontier=len(frontier),
        )
        return frontier
