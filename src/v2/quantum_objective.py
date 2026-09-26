"""Changed quantum objective: optimize recovery utility, not energy.

Instead of optimizing <H>, optimize:

    R(theta) = U(Recovery(S_theta, B_C))

The quantum search learns to produce bitstrings that a bounded classical
solver has the most use for.  Not: produce low average energy.

Also support:
    P_theta[d_H(x, X*) <= k]  — probability of being within Hamming distance k
    P_theta[E(x) <= E_threshold]  — probability of low energy
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.metrics import compute_metrics
from src.problems import MWISInstance, is_independent, mwis_value
from src.quantum_circuits import (
    build_mwis_qaoa_circuit,
    get_circuit_metrics,
)
from src.recovery import recover

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

__all__ = ["RecoveryObjective", "HammingObjective"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _counts_to_samples(counts: Dict[str, int], n_qubits: int) -> List[np.ndarray]:
    """Expand a counts dict into a list of binary sample arrays.

    Qiskit bitstrings are little-endian (rightmost char = qubit 0), so we
    reverse the string before converting to an array indexed by qubit.
    """
    samples: List[np.ndarray] = []
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        subset = np.array([int(b) for b in reversed(bits)], dtype=int)
        if len(subset) < n_qubits:
            subset = np.pad(subset, (0, n_qubits - len(subset)))
        for _ in range(int(count)):
            samples.append(subset[:n_qubits])
    return samples


def _run_circuit(
    circuit: Any,
    n_shots: int,
    adapter: Any | None = None,
) -> Dict[str, int]:
    """Run a circuit and return counts, using the adapter or local sim."""
    if adapter is not None and _adapter_usable(adapter):
        try:
            device = adapter.select_device(circuit.num_qubits, prefer_gpu=True)
            result = adapter.run(circuit, device=device, shots=n_shots)
            counts = result.get("counts", {})
            if counts:
                return counts
        except Exception as exc:
            logger.warning("objective.run_failed", error=repr(exc))

    # Local fallback.
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    t_circ = transpile(circuit, sim)
    sim_result = sim.run(t_circ, shots=n_shots).result()
    return sim_result.get_counts()


def _adapter_usable(adapter: Any) -> bool:
    """Check whether the adapter has a live BlueQubit client."""
    client = getattr(adapter, "_client", None)
    return client is not None


# ---------------------------------------------------------------------------
# Recovery objective
# ---------------------------------------------------------------------------
class RecoveryObjective:
    """Objective function that optimises post-recovery utility, not energy.

    The core idea: the quantum circuit should produce bitstrings that are
    *most useful* to a bounded classical recovery solver, not bitstrings
    with the lowest average Hamiltonian energy.

    The objective is::

        R(theta) = -post_recovery_gap + p_hit + 0.1 * diversity - 0.01 * depth

    where ``post_recovery_gap`` is the optimality gap after recovery (lower
    is better, so we negate it), ``p_hit`` is the probability of producing a
    near-optimal feasible solution, ``diversity`` is the normalised sample
    entropy, and ``depth`` penalises deep circuits.
    """

    def __init__(
        self,
        instance: MWISInstance,
        recovery_budget: int = 5000,
        n_shots: int = 128,
    ) -> None:
        self.instance = instance
        self.recovery_budget = recovery_budget
        self.n_shots = n_shots
        self._history: List[Dict[str, Any]] = []

    def evaluate(
        self,
        circuit: Any,
        adapter: Any | None = None,
    ) -> Dict[str, Any]:
        """Run the circuit, recover, and return a metrics dict.

        The returned dict contains all the raw metrics used by
        :meth:`score`, plus the circuit depth and 2-qubit gate count.
        """
        n = self.instance.n
        counts = _run_circuit(circuit, self.n_shots, adapter)
        samples = _counts_to_samples(counts, n)

        # Pad if we got fewer shots than requested.
        while len(samples) < self.n_shots:
            samples.append(np.zeros(n, dtype=int))

        rec = recover(
            samples,
            self.instance,
            max_expansion=2,
            max_candidates=self.recovery_budget,
        )
        metrics = compute_metrics(samples, self.instance, recovery_result=rec)

        circ_metrics = get_circuit_metrics(circuit)
        metrics["circuit_depth"] = circ_metrics["depth"]
        metrics["2q_gate_count"] = circ_metrics["2q_gate_count"]
        metrics["post_recovery_best"] = float(
            rec.get("post_recovery_best", metrics.get("pre_recovery_best", 0.0))
        )
        metrics["post_recovery_gap"] = float(rec.get("optimality_gap", float("nan")))
        metrics["recovery_time"] = float(rec.get("recovery_time", 0.0))

        self._history.append(metrics)
        return metrics

    def score(self, metrics: Dict[str, Any]) -> float:
        """Compute the composite recovery objective.

        R = -post_recovery_gap + p_hit + 0.1 * diversity - 0.01 * circuit_depth

        A higher score is better.  ``post_recovery_gap`` is negated because a
        smaller gap is better.  ``circuit_depth`` is lightly penalised to
        favour shallower circuits at equal quality.
        """
        gap = metrics.get("post_recovery_gap", 1.0)
        if gap is None or (isinstance(gap, float) and np.isnan(gap)):
            gap = 1.0
        gap = float(gap)

        p_hit = metrics.get("p_hit", 0.0)
        if p_hit is None or (isinstance(p_hit, float) and np.isnan(p_hit)):
            p_hit = 0.0
        p_hit = float(p_hit)

        diversity = float(metrics.get("diversity", 0.0))
        depth = float(metrics.get("circuit_depth", 0))

        return -gap + p_hit + 0.1 * diversity - 0.01 * depth

    def optimize_parameters(
        self,
        instance: MWISInstance,
        p: int = 1,
        n_iterations: int = 100,
        adapter: Any | None = None,
    ) -> Dict[str, Any]:
        """Simple random search over gamma/beta optimising R.

        Performs ``n_iterations`` random samples of the QAOA parameter space
        and returns the best-scoring set of parameters.  This is intentionally
        simple — the focus is on the *objective* (recovery utility), not the
        optimiser.
        """
        rng = np.random.RandomState(42)
        penalty_coeff = float(2.0 * np.max(instance.weights)) if instance.n > 0 else 1.0

        best_score = -np.inf
        best_params: Dict[str, Any] = {
            "gammas": [],
            "betas": [],
            "penalty_coeff": penalty_coeff,
            "score": best_score,
            "metrics": {},
        }

        for i in range(n_iterations):
            gammas = rng.uniform(0.0, np.pi, size=p).tolist()
            betas = rng.uniform(0.0, np.pi / 2, size=p).tolist()

            circuit = build_mwis_qaoa_circuit(
                instance, p, gammas, betas, penalty_coeff
            )
            metrics = self.evaluate(circuit, adapter=adapter)
            score = self.score(metrics)

            if score > best_score:
                best_score = score
                best_params = {
                    "gammas": gammas,
                    "betas": betas,
                    "penalty_coeff": penalty_coeff,
                    "score": score,
                    "metrics": metrics,
                    "iteration": i,
                }
                logger.info(
                    "objective.improved",
                    iteration=i,
                    score=score,
                    gap=metrics.get("post_recovery_gap"),
                )

        best_params["n_iterations"] = n_iterations
        best_params["p"] = p
        return best_params


# ---------------------------------------------------------------------------
# Hamming objective
# ---------------------------------------------------------------------------
class HammingObjective:
    """Objective measuring the probability of being within Hamming distance k.

    Given the known optimal solution ``X*`` (if available), this objective
    computes::

        P_theta[d_H(x, X*) <= k]

    i.e. the fraction of samples that differ from the optimum in at most
    ``k`` bit positions.  This is a softer proximity measure than exact
    hit probability.
    """

    def __init__(self, instance: MWISInstance, k: int = 3) -> None:
        self.instance = instance
        self.k = k

    def evaluate(self, samples: List[np.ndarray]) -> Dict[str, Any]:
        """Compute Hamming-distance metrics for a set of samples.

        Returns a dict with ``n_within_k``, ``n_shots``, ``p_within_k``,
        ``mean_hamming``, and ``min_hamming``.
        """
        n_shots = len(samples)
        if n_shots == 0:
            return {
                "n_within_k": 0,
                "n_shots": 0,
                "p_within_k": 0.0,
                "mean_hamming": float("nan"),
                "min_hamming": float("nan"),
            }

        optimal = self.instance.optimal_solution
        if optimal is None:
            # Without a known optimum, fall back to the best sample.
            weights = self.instance.weights
            values = [mwis_value(s, weights) for s in samples]
            best_idx = int(np.argmax(values))
            optimal = samples[best_idx]

        optimal = np.asarray(optimal, dtype=int)
        hamming_distances = [
            int(np.sum(s.astype(int) != optimal)) for s in samples
        ]
        n_within_k = sum(1 for d in hamming_distances if d <= self.k)

        return {
            "n_within_k": n_within_k,
            "n_shots": n_shots,
            "p_within_k": n_within_k / n_shots,
            "mean_hamming": float(np.mean(hamming_distances)),
            "min_hamming": float(np.min(hamming_distances)),
            "hamming_distances": hamming_distances,
        }

    def score(self, metrics: Dict[str, Any]) -> float:
        """Return the fraction of samples within Hamming distance k."""
        return float(metrics.get("p_within_k", 0.0))


if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.exact_solver import solve_mwis_exact

    inst = generate_mwis_instance(n=10, density=0.4, seed=99)
    opt, opt_val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt
    inst.optimal_value = opt_val

    obj = RecoveryObjective(inst, recovery_budget=5000, n_shots=128)
    best = obj.optimize_parameters(inst, p=1, n_iterations=20)
    print(f"Best score: {best['score']:.4f}")
    print(f"Best gammas: {best['gammas']}")
    print(f"Best betas: {best['betas']}")
    print(f"Post-recovery gap: {best['metrics'].get('post_recovery_gap', 'nan')}")

    # Hamming objective
    from src.classical_samplers import GreedySampler

    sampler = GreedySampler()
    samples, _ = sampler.sample(inst, n_shots=50, seed=1)
    ham_obj = HammingObjective(inst, k=3)
    ham_metrics = ham_obj.evaluate(samples)
    print(f"P(d_H <= 3) = {ham_obj.score(ham_metrics):.3f}")
    print(f"Mean Hamming distance: {ham_metrics['mean_hamming']:.2f}")
