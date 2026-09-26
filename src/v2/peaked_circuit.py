"""Peaked circuit track: search for circuits with strong verifiable peaks,
high entanglement, rapid Pauli growth, and poor low-chi MPS approximation.

This is a secondary 10-15% sideline, not a replacement for the optimization
project.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

__all__ = ["PeakedCircuitSearch"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _counts_to_distribution(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    """Convert a counts dict to a length-2^n probability vector."""
    total = sum(counts.values())
    if total == 0:
        return np.zeros(1 << n_qubits, dtype=float)
    dist = np.zeros(1 << n_qubits, dtype=float)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        idx = int(bits, 2)
        dist[idx] = count / total
    return dist


def _adapter_usable(adapter: Any) -> bool:
    """Check whether the adapter has a live BlueQubit client."""
    client = getattr(adapter, "_client", None)
    return client is not None


def _run_circuit(
    circuit: Any,
    n_shots: int,
    adapter: Any | None = None,
) -> Dict[str, int]:
    """Run a circuit and return counts."""
    if adapter is not None and _adapter_usable(adapter):
        try:
            device = adapter.select_device(circuit.num_qubits, prefer_gpu=True)
            result = adapter.run(circuit, device=device, shots=n_shots)
            counts = result.get("counts", {})
            if counts:
                return counts
        except Exception as exc:
            logger.warning("peaked.run_failed", error=repr(exc))

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    t_circ = transpile(circuit, sim)
    sim_result = sim.run(t_circ, shots=n_shots).result()
    return sim_result.get_counts()


def _get_statevector(circuit: Any) -> np.ndarray:
    """Get the statevector of a circuit (without measurements)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")
    circ_no_meas = circuit.copy()
    circ_no_meas.remove_final_measurements()
    circ_no_meas.save_statevector()
    t_circ = transpile(circ_no_meas, sim)
    result = sim.run(t_circ).result()
    return np.asarray(result.data(0)["statevector"])


def _truncate_bond_dimension(
    dist: np.ndarray, n_qubits: int, chi: int
) -> np.ndarray:
    """Approximate a distribution by truncating its MPS bond dimension to chi.

    Sequential SVD truncation across each bipartition of the qubit chain.
    """
    amps = np.sqrt(np.maximum(dist, 0.0)).reshape([2] * n_qubits)
    current = amps
    for i in range(n_qubits - 1):
        shape = current.shape
        left_dim = int(np.prod(shape[: i + 1]))
        right_dim = int(np.prod(shape[i + 1 :]))
        mat = current.reshape(left_dim, right_dim)
        u_mat, svals, vh = np.linalg.svd(mat, full_matrices=False)
        k = min(chi, len(svals))
        u_mat = u_mat[:, :k]
        svals = svals[:k]
        vh = vh[:k, :]
        norm = np.sqrt(np.sum(svals**2))
        if norm > 1e-15:
            svals = svals / norm
        else:
            svals = np.zeros_like(svals)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            truncated = (u_mat * svals) @ vh
        truncated = np.nan_to_num(truncated, nan=0.0, posinf=0.0, neginf=0.0)
        current = truncated.reshape(shape)

    result_amps = current.reshape(1 << n_qubits)
    result_probs = np.abs(result_amps) ** 2
    total = result_probs.sum()
    if total > 0:
        result_probs = result_probs / total
    return result_probs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class PeakedCircuitSearch:
    """Search for circuits with strong verifiable peaks.

    A "peaked" circuit is one whose output distribution has a dominant peak
    (high peak strength), high entanglement, and poor low-chi MPS
    approximation.  Such circuits are interesting because:

    - The peak is *verifiable* (you can check the top bitstring).
    - High entanglement suggests genuine quantum structure.
    - Poor MPS approximability suggests the peak is not a classical artifact.

    This is a secondary 10-15% sideline alongside the main optimization track.
    """

    def __init__(
        self,
        n_qubits: int = 20,
        depth_range: Tuple[int, int] = (2, 10),
    ) -> None:
        self.n_qubits = n_qubits
        self.depth_range = depth_range
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Circuit generation
    # ------------------------------------------------------------------
    def generate_random_circuit(
        self,
        n_qubits: int,
        depth: int,
        seed: int = 0,
    ) -> QuantumCircuit:
        """Generate a random parameterised circuit.

        Uses a layered architecture: alternating rounds of single-qubit
        rotations (RZ, RY) and nearest-neighbour CNOT entangling layers.
        """
        rng = np.random.RandomState(seed)
        circuit = QuantumCircuit(n_qubits)

        for layer in range(depth):
            # Single-qubit rotation layer.
            for q in range(n_qubits):
                theta = rng.uniform(0, 2 * np.pi)
                phi = rng.uniform(0, 2 * np.pi)
                circuit.rz(theta, q)
                circuit.ry(phi, q)

            # Entangling layer: nearest-neighbour CNOTs.
            for q in range(n_qubits - 1):
                if rng.random() < 0.8:
                    circuit.cx(q, q + 1)

            # Occasional long-range CNOT for more entanglement.
            if n_qubits > 4 and rng.random() < 0.3:
                q1 = rng.randint(n_qubits)
                q2 = rng.randint(n_qubits)
                if q1 != q2:
                    circuit.cx(q1, q2)

        circuit.measure_all()
        return circuit

    # ------------------------------------------------------------------
    # Peak strength
    # ------------------------------------------------------------------
    def compute_peak_strength(self, counts: Dict[str, int]) -> float:
        """Measure how peaked the distribution is.

        Peak strength = max_prob / mean_prob.

        A uniform distribution has peak strength 1.0.  A distribution
        concentrated on a single bitstring has peak strength = 2^n.
        """
        if not counts:
            return 0.0
        total = sum(counts.values())
        n_outcomes = len(counts)
        if n_outcomes == 0 or total == 0:
            return 0.0

        max_prob = max(counts.values()) / total
        mean_prob = 1.0 / n_outcomes  # uniform over observed outcomes

        if mean_prob <= 0:
            return 0.0
        return max_prob / mean_prob

    # ------------------------------------------------------------------
    # Entanglement proxy
    # ------------------------------------------------------------------
    def compute_entanglement_proxy(self, circuit: Any) -> Dict[str, float]:
        """Proxy for entanglement.

        Computes several proxies:
        - ``circuit_depth``: deeper circuits tend to be more entangling.
        - ``two_qubit_gate_count``: more 2q gates -> more entanglement.
        - ``mean_bipartite_entropy``: average von Neumann entropy across
          all single-qubit cuts of the output statevector.  This is a
          direct (but expensive) entanglement measure.
        - ``max_bipartite_entropy``: maximum single-qubit-cut entropy.
        """
        two_qubit_names = {"cx", "cz", "rzz", "cp", "swap", "ecr", "iswap"}
        depth = circuit.depth()
        total_gates = circuit.size()
        two_qubit_count = sum(
            1
            for inst in circuit.data
            if inst.operation.num_qubits == 2
            or inst.operation.name in two_qubit_names
        )

        # Compute bipartite entropy from the statevector.
        mean_entropy = 0.0
        max_entropy = 0.0
        try:
            sv = _get_statevector(circuit)
            n = circuit.num_qubits
            entropies: List[float] = []
            for q in range(n):
                # Reshape to separate qubit q from the rest.
                # Qiskit statevector is indexed as |q_{n-1} ... q_0>.
                # We need to move qubit q to the first axis.
                sv_matrix = sv.reshape([2] * n)
                # Move axis q to position 0.
                sv_matrix = np.moveaxis(sv_matrix, q, 0)
                sv_matrix = sv_matrix.reshape(2, -1)
                # SVD to get Schmidt coefficients.
                u, svals, vh = np.linalg.svd(sv_matrix, full_matrices=False)
                probs = np.abs(svals) ** 2
                probs = probs / max(probs.sum(), 1e-15)
                # Von Neumann entropy (natural log).
                mask = probs > 1e-15
                entropy = -float(np.sum(probs[mask] * np.log(probs[mask])))
                entropies.append(entropy)
            if entropies:
                mean_entropy = float(np.mean(entropies))
                max_entropy = float(np.max(entropies))
        except Exception as exc:
            logger.warning("entanglement.computation_failed", error=repr(exc))

        return {
            "circuit_depth": float(depth),
            "two_qubit_gate_count": float(two_qubit_count),
            "total_gates": float(total_gates),
            "mean_bipartite_entropy": mean_entropy,
            "max_bipartite_entropy": max_entropy,
        }

    # ------------------------------------------------------------------
    # MPS approximability
    # ------------------------------------------------------------------
    def compute_mps_approximability(
        self,
        circuit: Any,
        bond_dims: Sequence[int] = (8, 16, 32, 64),
        n_shots: int = 1024,
    ) -> Dict[str, Any]:
        """Measure how poorly low-chi MPS approximates the circuit.

        Returns a dict with ``js_per_chi`` (JS divergence between exact and
        MPS-truncated at each bond dimension) and ``approximability_score``
        (higher = harder to approximate).
        """
        n_qubits = circuit.num_qubits

        # Get exact distribution.
        try:
            sv = _get_statevector(circuit)
            exact_dist = np.abs(sv) ** 2
            exact_dist = np.real(exact_dist).astype(float)
            exact_dist = exact_dist / max(exact_dist.sum(), 1e-15)
        except Exception:
            # Fallback: use sampled counts.
            counts = _run_circuit(circuit, n_shots, adapter=None)
            exact_dist = _counts_to_distribution(counts, n_qubits)

        js_per_chi: Dict[int, float] = {}
        for chi in bond_dims:
            chi = int(chi)
            truncated = _truncate_bond_dimension(exact_dist, n_qubits, chi)
            js = self._js_divergence(exact_dist, truncated)
            js_per_chi[chi] = js

        # Approximability score: average JS divergence across bond dims.
        # Higher score = harder to approximate with MPS.
        score = float(np.mean(list(js_per_chi.values())))

        return {
            "js_per_chi": js_per_chi,
            "approximability_score": score,
            "exact_peak_prob": float(np.max(exact_dist)),
        }

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence (natural log)."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if len(p) != len(q):
            max_len = max(len(p), len(q))
            p = np.pad(p, (0, max_len - len(p)))
            q = np.pad(q, (0, max_len - len(q)))
        p = p / max(p.sum(), 1e-15)
        q = q / max(q.sum(), 1e-15)
        m = 0.5 * (p + q)

        def _kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-15))))

        return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))

    # ------------------------------------------------------------------
    # Evaluate a single circuit
    # ------------------------------------------------------------------
    def evaluate_circuit(
        self,
        circuit: Any,
        adapter: Any | None = None,
        n_shots: int = 1024,
    ) -> Dict[str, Any]:
        """Compute peak strength, entanglement proxy, and MPS approximability.

        Returns a dict with all metrics plus a composite ``peaked_score``.
        """
        n_qubits = circuit.num_qubits

        # Run the circuit to get counts.
        counts = _run_circuit(circuit, n_shots, adapter=adapter)

        peak_strength = self.compute_peak_strength(counts)
        entanglement = self.compute_entanglement_proxy(circuit)
        mps_approx = self.compute_mps_approximability(circuit)

        # Composite peaked score: high peak + high entanglement + poor MPS.
        # Normalise peak strength by 2^n to get it in a comparable range.
        norm_peak = peak_strength / max(1 << n_qubits, 1)
        ent_score = entanglement["mean_bipartite_entropy"] / max(math.log(2), 1e-9)
        mps_score = min(mps_approx["approximability_score"], 1.0)

        peaked_score = (
            0.4 * norm_peak
            + 0.3 * ent_score
            + 0.3 * mps_score
        )

        return {
            "n_qubits": n_qubits,
            "peak_strength": float(peak_strength),
            "normalized_peak": float(norm_peak),
            "entanglement": entanglement,
            "mps_approximability": mps_approx,
            "peaked_score": float(peaked_score),
            "counts": counts,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        n_candidates: int = 100,
        n_qubits: int = 20,
        adapter: Any | None = None,
        n_shots: int = 1024,
        depth_range: Tuple[int, int] | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for circuits with strong peaks + high entanglement + poor MPS.

        Generates ``n_candidates`` random circuits, evaluates each, and
        returns the top candidates sorted by ``peaked_score``.
        """
        if depth_range is None:
            depth_range = self.depth_range

        results: List[Dict[str, Any]] = []

        for i in range(n_candidates):
            rng = np.random.RandomState(i)
            depth = int(rng.randint(depth_range[0], depth_range[1] + 1))
            circuit = self.generate_random_circuit(n_qubits, depth, seed=i)

            try:
                evaluation = self.evaluate_circuit(circuit, adapter=adapter, n_shots=n_shots)
                evaluation["candidate_id"] = i
                evaluation["depth"] = depth
                evaluation["seed"] = i
                results.append(evaluation)
            except Exception as exc:
                logger.warning("peaked.evaluate_failed", candidate=i, error=repr(exc))

        # Sort by peaked score (descending).
        results.sort(key=lambda r: r.get("peaked_score", 0.0), reverse=True)
        self._results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Pauli growth (rapid Pauli growth heuristic)
    # ------------------------------------------------------------------
    def compute_pauli_growth(
        self,
        circuit: Any,
    ) -> Dict[str, Any]:
        """Heuristic for rapid Pauli operator growth.

        Pauli growth is a key indicator of classical simulability hardness.
        Circuits where the number of non-Clifford gates and the depth cause
        rapid growth of the Pauli operator support are harder to simulate
        classically via Pauli-path methods.

        We approximate this by counting non-Clifford gates and estimating
        the Pauli support growth rate.
        """
        clifford_gates = {"h", "s", "sdg", "cx", "cz", "x", "y", "z", "sx", "sxdg"}
        non_clifford_count = 0
        total_gates = 0
        for inst in circuit.data:
            name = inst.operation.name
            total_gates += 1
            if name not in clifford_gates:
                non_clifford_count += 1

        n_qubits = circuit.num_qubits
        depth = circuit.depth()

        # Heuristic: Pauli support grows roughly as (non_clifford_fraction * depth).
        non_clifford_fraction = non_clifford_count / max(total_gates, 1)
        growth_rate = non_clifford_fraction * depth

        # Estimate the Pauli support after the circuit.
        # Upper bound: 4^n (all Paulis), but in practice grows as ~exp(growth_rate).
        estimated_support = min(4**n_qubits, int(2 ** min(growth_rate, n_qubits * 2)))

        return {
            "non_clifford_gate_count": non_clifford_count,
            "total_gates": total_gates,
            "non_clifford_fraction": float(non_clifford_fraction),
            "depth": depth,
            "growth_rate": float(growth_rate),
            "estimated_pauli_support": estimated_support,
            "rapid_growth": growth_rate > n_qubits,
        }


if __name__ == "__main__":
    search = PeakedCircuitSearch(n_qubits=8, depth_range=(2, 6))

    circuit = search.generate_random_circuit(8, depth=4, seed=42)
    evaluation = search.evaluate_circuit(circuit, n_shots=1024)
    print(f"Peak strength: {evaluation['peak_strength']:.2f}")
    print(f"Normalized peak: {evaluation['normalized_peak']:.6f}")
    print(f"Mean entropy: {evaluation['entanglement']['mean_bipartite_entropy']:.4f}")
    print(f"MPS approx score: {evaluation['mps_approximability']['approximability_score']:.4f}")
    print(f"Peaked score: {evaluation['peaked_score']:.4f}")

    pauli = search.compute_pauli_growth(circuit)
    print(f"Pauli growth rate: {pauli['growth_rate']:.2f} (rapid: {pauli['rapid_growth']})")

    results = search.search(n_candidates=10, n_qubits=8, n_shots=512)
    print(f"\nTop 3 candidates:")
    for r in results[:3]:
        print(f"  #{r['candidate_id']}: peaked={r['peaked_score']:.4f} "
              f"peak={r['peak_strength']:.1f} "
              f"ent={r['entanglement']['mean_bipartite_entropy']:.3f}")
