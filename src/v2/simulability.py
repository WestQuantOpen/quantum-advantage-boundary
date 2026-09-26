"""Classical simulability attack using MPS bond dimension sweep.

For each promising quantum circuit:
  1. Exact GPU gives reference distribution P_exact(x)
  2. MPS with bond dimensions chi=8,16,32,64,128 gives P_MPS,chi(x)
  3. Compute D_JS(P_exact, P_MPS,chi) and Delta_U_chi = U(P_exact) - U(P_MPS,chi)

If chi=16 reproduces all downstream utility, there's no interesting quantum-hardness.
But if U_quantum >> U_MPS,chi up to large chi, we've identified a classical
simulability limit.

Also use Pauli-path as red-team:
  "Can I classically explain this supposedly hard quantum circuit?"
  Pauli-path with truncation threshold 10^-5 as adversarial verification.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.metrics import compute_metrics
from src.problems import MWISInstance, mwis_value
from src.recovery import recover

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

__all__ = ["SimulabilityAttack"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _counts_to_distribution(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    """Convert a counts dict to a length-2^n probability vector.

    Bitstring keys are interpreted with the rightmost character as qubit 0
    (matching Qiskit's little-endian convention).
    """
    total = sum(counts.values())
    if total == 0:
        return np.zeros(1 << n_qubits, dtype=float)
    dist = np.zeros(1 << n_qubits, dtype=float)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        # Qiskit uses little-endian: the rightmost bit is qubit 0.
        idx = int(bits, 2)
        dist[idx] = count / total
    return dist


def _distribution_to_counts(dist: np.ndarray, n_shots: int) -> Dict[str, int]:
    """Convert a probability vector to integer counts (multinomial rounding)."""
    n_qubits = int(round(math.log2(len(dist)))) if len(dist) > 1 else 0
    probs = np.maximum(dist, 0.0)
    total = probs.sum()
    if total <= 0:
        return {}
    probs = probs / total
    counts_arr = np.random.multinomial(n_shots, probs)
    counts: Dict[str, int] = {}
    for idx, c in enumerate(counts_arr):
        if c > 0:
            bitstring = format(idx, f"0{n_qubits}b")
            counts[bitstring] = int(c)
    return counts


def _counts_to_samples(counts: Dict[str, int], n_qubits: int) -> List[np.ndarray]:
    """Expand counts into a list of binary sample arrays."""
    samples: List[np.ndarray] = []
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        subset = np.array([int(b) for b in reversed(bits)], dtype=int)
        if len(subset) < n_qubits:
            subset = np.pad(subset, (0, n_qubits - len(subset)))
        for _ in range(int(count)):
            samples.append(subset[:n_qubits])
    return samples


def _local_exact_counts(circuit: Any, n_shots: int) -> Dict[str, int]:
    """Run exact statevector simulation locally via Qiskit Aer."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")
    t_circ = transpile(circuit, sim)
    result = sim.run(t_circ, shots=n_shots).result()
    return result.get_counts()


def _local_exact_distribution(circuit: Any) -> np.ndarray:
    """Get the exact probability distribution via local statevector sim."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")
    # Remove measurements for statevector simulation.
    circ_no_meas = circuit.copy()
    circ_no_meas.remove_final_measurements()
    # Explicitly request the statevector to be saved.
    circ_no_meas.save_statevector()
    t_circ = transpile(circ_no_meas, sim)
    result = sim.run(t_circ).result()
    statevector = result.data(0)["statevector"]
    probs = np.abs(statevector) ** 2
    return np.real(probs).astype(float)


def _truncate_bond_dimension(
    dist: np.ndarray, n_qubits: int, chi: int
) -> np.ndarray:
    """Approximate a distribution by truncating its MPS bond dimension to chi.

    This performs a sequential SVD truncation across each bipartition of the
    qubit chain.  At each cut the Schmidt spectrum is truncated to the top
    ``chi`` singular values and the state is renormalised.  This is a proxy
    for a full MPS simulation with bond dimension ``chi`` — it captures the
    leading-order entanglement truncation error but is not identical to a
    gate-by-gate MPS simulation.
    """
    # Reshape the statevector into a sequence of qubit tensors and sweep
    # left-to-right, truncating at each bond.
    state = dist.reshape([2] * n_qubits).astype(complex)
    # We work with the amplitude representation for truncation, then convert
    # back to probabilities.  Since ``dist`` is real and non-negative we can
    # treat sqrt(dist) as the "amplitude".
    amps = np.sqrt(np.maximum(dist, 0.0)).reshape([2] * n_qubits)

    # Left-to-right sweep: merge qubits one at a time, SVD, truncate.
    current = amps
    for i in range(n_qubits - 1):
        shape = current.shape
        # Reshape into a matrix: left part (2^(i+1)) x right part (rest).
        left_dim = int(np.prod(shape[: i + 1]))
        right_dim = int(np.prod(shape[i + 1 :]))
        mat = current.reshape(left_dim, right_dim)
        u_mat, svals, vh = np.linalg.svd(mat, full_matrices=False)
        # Truncate to chi singular values.
        k = min(chi, len(svals))
        u_mat = u_mat[:, :k]
        svals = svals[:k]
        vh = vh[:k, :]
        # Renormalise.
        norm = np.sqrt(np.sum(svals**2))
        if norm > 1e-15:
            svals = svals / norm
        else:
            svals = np.zeros_like(svals)
        # Reconstruct the truncated tensor.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            truncated = (u_mat * svals) @ vh
        truncated = np.nan_to_num(truncated, nan=0.0, posinf=0.0, neginf=0.0)
        current = truncated.reshape(shape)

    # Flatten back to a probability distribution.
    result_amps = current.reshape(1 << n_qubits)
    result_probs = np.abs(result_amps) ** 2
    total = result_probs.sum()
    if total > 0:
        result_probs = result_probs / total
    return result_probs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class SimulabilityAttack:
    """Classical simulability attack via MPS bond-dimension sweep.

    For each quantum circuit the attack:
      1. Obtains the exact (statevector) distribution.
      2. Runs MPS simulations at a sweep of bond dimensions.
      3. Compares distributions (Jensen-Shannon divergence) and downstream
         utility (post-recovery objective) to determine whether a bounded
         classical MPS simulator can reproduce the quantum advantage.

    A secondary Pauli-path verification acts as an adversarial red-team.
    """

    def __init__(self, bluequbit_adapter: Any | None = None) -> None:
        self.adapter = bluequbit_adapter
        self.results_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # MPS sweep
    # ------------------------------------------------------------------
    def run_mps_sweep(
        self,
        circuit: Any,
        n_shots: int = 256,
        bond_dims: Sequence[int] = (8, 16, 32, 64, 128),
    ) -> Dict[int, Dict[str, int]]:
        """Run the same circuit on MPS with different bond dimensions.

        Returns a dict mapping ``chi -> counts`` (bitstring -> integer counts).
        """
        n_qubits = circuit.num_qubits
        results: Dict[int, Dict[str, int]] = {}

        for chi in bond_dims:
            chi = int(chi)
            counts = self._run_mps(circuit, chi=chi, n_shots=n_shots, n_qubits=n_qubits)
            results[chi] = counts

        return results

    def _run_mps(
        self,
        circuit: Any,
        chi: int,
        n_shots: int,
        n_qubits: int,
    ) -> Dict[str, int]:
        """Run a single MPS simulation at bond dimension ``chi``."""
        # Try BlueQubit MPS backend with explicit bond dimension.
        if self.adapter is not None and self._adapter_usable():
            try:
                return self._run_mps_bluequbit(circuit, chi, n_shots)
            except Exception as exc:
                logger.warning("mps.bluequbit_failed", chi=chi, error=repr(exc))

        # Local fallback: exact statevector + bond-dimension truncation.
        return self._run_mps_local(circuit, chi, n_shots, n_qubits)

    def _adapter_usable(self) -> bool:
        """Check whether the adapter has a live BlueQubit client."""
        if self.adapter is None:
            return False
        client = getattr(self.adapter, "_client", None)
        return client is not None

    def _run_mps_bluequbit(
        self, circuit: Any, chi: int, n_shots: int
    ) -> Dict[str, int]:
        """Run MPS on BlueQubit with a specific bond dimension.

        Reaches into the adapter's underlying client to pass the
        ``max_bond_dimension`` option, which the public ``run`` method does
        not expose.
        """
        client = self.adapter._require_client()  # type: ignore[attr-defined]
        result = client.run(
            circuit,
            device="mps.cpu",
            shots=n_shots,
            options={"max_bond_dimension": chi},
        )
        return result.get_counts()

    def _run_mps_local(
        self, circuit: Any, chi: int, n_shots: int, n_qubits: int
    ) -> Dict[str, int]:
        """Local fallback: exact statevector + bond-dimension truncation."""
        exact_dist = _local_exact_distribution(circuit)
        truncated = _truncate_bond_dimension(exact_dist, n_qubits, chi)
        return _distribution_to_counts(truncated, n_shots)

    # ------------------------------------------------------------------
    # Exact reference
    # ------------------------------------------------------------------
    def get_exact_counts(
        self, circuit: Any, n_shots: int = 4096
    ) -> Dict[str, int]:
        """Get exact (statevector) counts for reference."""
        n_qubits = circuit.num_qubits

        if self.adapter is not None and self._adapter_usable():
            try:
                device = self.adapter.select_device(n_qubits, prefer_gpu=True)
                bq_result = self.adapter.run(
                    circuit, device=device, shots=n_shots,
                    job_name="simulability_exact",
                )
                counts = bq_result.get("counts", {})
                if counts:
                    return counts
            except Exception as exc:
                logger.warning("exact.bluequbit_failed", error=repr(exc))

        return _local_exact_counts(circuit, n_shots)

    # ------------------------------------------------------------------
    # Divergence metrics
    # ------------------------------------------------------------------
    def compute_js_divergence(
        self,
        p_exact: np.ndarray | Dict[str, int],
        p_approx: np.ndarray | Dict[str, int],
    ) -> float:
        """Jensen-Shannon divergence between two distributions.

        Accepts either probability vectors or counts dicts.  Returns a
        value in ``[0, ln(2)]`` (natural log) — 0 means identical, ln(2)
        means maximally different.
        """
        p = self._to_prob_vector(p_exact)
        q = self._to_prob_vector(p_approx)

        # Align lengths (pad the shorter with zeros).
        if len(p) != len(q):
            max_len = max(len(p), len(q))
            p = np.pad(p, (0, max_len - len(p)))
            q = np.pad(q, (0, max_len - len(q)))

        # Normalise defensively.
        p = p / max(p.sum(), 1e-15)
        q = q / max(q.sum(), 1e-15)

        m = 0.5 * (p + q)

        def _kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-15))))

        js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        return max(0.0, js)

    @staticmethod
    def _to_prob_vector(
        dist: np.ndarray | Dict[str, int]
    ) -> np.ndarray:
        """Convert a counts dict or array into a probability vector."""
        if isinstance(dist, dict):
            n_qubits = max(len(k.replace(" ", "")) for k in dist) if dist else 0
            return _counts_to_distribution(dist, n_qubits)
        return np.asarray(dist, dtype=float)

    # ------------------------------------------------------------------
    # Utility loss
    # ------------------------------------------------------------------
    def compute_utility_loss(
        self,
        exact_counts: Dict[str, int],
        mps_counts: Dict[str, int],
        instance: MWISInstance,
        recovery_budget: int = 5000,
    ) -> Dict[str, float]:
        """Compute U(exact) - U(mps) after recovery.

        Returns a dict with ``u_exact``, ``u_mps``, ``delta_u`` (utility
        loss), and the JS divergence between the two count distributions.
        """
        n = instance.n

        exact_samples = _counts_to_samples(exact_counts, n)
        mps_samples = _counts_to_samples(mps_counts, n)

        if not exact_samples:
            return {"u_exact": 0.0, "u_mps": 0.0, "delta_u": 0.0, "js": 0.0}
        if not mps_samples:
            u_exact = self._recovery_utility(exact_samples, instance, recovery_budget)
            return {
                "u_exact": u_exact,
                "u_mps": 0.0,
                "delta_u": u_exact,
                "js": float(self.compute_js_divergence(exact_counts, {})),
            }

        u_exact = self._recovery_utility(exact_samples, instance, recovery_budget)
        u_mps = self._recovery_utility(mps_samples, instance, recovery_budget)
        js = self.compute_js_divergence(exact_counts, mps_counts)

        return {
            "u_exact": float(u_exact),
            "u_mps": float(u_mps),
            "delta_u": float(u_exact - u_mps),
            "js": float(js),
        }

    @staticmethod
    def _recovery_utility(
        samples: List[np.ndarray],
        instance: MWISInstance,
        recovery_budget: int,
    ) -> float:
        """Run recovery and return the post-recovery best value."""
        rec = recover(
            samples,
            instance,
            max_expansion=2,
            max_candidates=recovery_budget,
        )
        return float(rec["post_recovery_best"])

    # ------------------------------------------------------------------
    # Simulability limit
    # ------------------------------------------------------------------
    def find_simulability_limit(
        self,
        circuit: Any,
        instance: MWISInstance,
        bond_dims: Sequence[int] = (8, 16, 32, 64, 128),
        n_shots: int = 256,
        utility_threshold: float = 0.01,
        recovery_budget: int = 5000,
    ) -> Dict[str, Any]:
        """Find the chi where utility loss drops below threshold.

        Returns a dict with:
          - ``simulability_limit``: the smallest chi where Delta_U < threshold
            (or None if never reached).
          - ``per_chi``: detailed results for each bond dimension.
          - ``hardness``: classification label.
        """
        exact_counts = self.get_exact_counts(circuit, n_shots=max(n_shots, 1024))
        mps_results = self.run_mps_sweep(circuit, n_shots=n_shots, bond_dims=bond_dims)

        per_chi: List[Dict[str, Any]] = []
        limit: int | None = None

        for chi in bond_dims:
            chi = int(chi)
            mps_counts = mps_results[chi]
            util = self.compute_utility_loss(
                exact_counts, mps_counts, instance, recovery_budget
            )
            entry = {
                "chi": chi,
                "u_exact": util["u_exact"],
                "u_mps": util["u_mps"],
                "delta_u": util["delta_u"],
                "js": util["js"],
            }
            per_chi.append(entry)

            if limit is None and util["delta_u"] < utility_threshold:
                limit = chi

        result = {
            "simulability_limit": limit,
            "per_chi": per_chi,
            "hardness": self.classify_hardness(per_chi),
            "u_exact": per_chi[0]["u_exact"] if per_chi else 0.0,
        }
        self.results_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Pauli-path verification
    # ------------------------------------------------------------------
    def pauli_path_verify(
        self,
        circuit: Any,
        n_shots: int = 256,
        truncation_threshold: float = 1e-5,
    ) -> Dict[str, Any]:
        """Run Pauli-path simulation with truncation and compare to exact.

        Pauli-path is a classical simulation method that decomposes the
        circuit into Pauli transfer paths and truncates low-weight paths.
        If the truncated Pauli-path reproduces the exact distribution, the
        circuit is classically easy via this method.

        Returns a dict with ``js_divergence``, ``pauli_path_counts``,
        ``exact_counts``, and ``verdict`` ("easy" or "hard").
        """
        n_qubits = circuit.num_qubits
        exact_counts = self.get_exact_counts(circuit, n_shots=max(n_shots, 1024))

        pauli_counts = self._run_pauli_path(
            circuit, n_shots=n_shots, truncation_threshold=truncation_threshold
        )

        js = self.compute_js_divergence(exact_counts, pauli_counts)

        verdict = "easy" if js < 0.01 else "hard"

        return {
            "js_divergence": float(js),
            "pauli_path_counts": pauli_counts,
            "exact_counts": exact_counts,
            "truncation_threshold": truncation_threshold,
            "verdict": verdict,
        }

    def _run_pauli_path(
        self,
        circuit: Any,
        n_shots: int,
        truncation_threshold: float,
    ) -> Dict[str, int]:
        """Run Pauli-path simulation.

        When a BlueQubit adapter is available, uses the ``pauli-path`` device.
        Otherwise falls back to a local approximation: exact statevector with
        a noise model whose strength is tied to the truncation threshold
        (paths below threshold are dropped, approximated as depolarising
        noise on the output distribution).
        """
        if self.adapter is not None and self._adapter_usable():
            try:
                client = self.adapter._require_client()  # type: ignore[attr-defined]
                result = client.run(
                    circuit,
                    device="pauli-path",
                    shots=n_shots,
                    options={"truncation_threshold": truncation_threshold},
                )
                return result.get_counts()
            except Exception as exc:
                logger.warning("pauli_path.bluequbit_failed", error=repr(exc))

        # Local fallback: exact distribution + depolarising noise proportional
        # to the truncation threshold.  A higher threshold (more aggressive
        # truncation) maps to more noise.
        n_qubits = circuit.num_qubits
        exact_dist = _local_exact_distribution(circuit)

        # Map truncation threshold to a depolarising parameter.
        # threshold=1e-5 -> minimal noise; threshold=1e-1 -> strong noise.
        log_thresh = math.log10(max(truncation_threshold, 1e-10))
        # log_thresh ranges from about -5 (tiny threshold) to -1 (large).
        # Map to noise level in [0.0, 0.3].
        noise_level = max(0.0, min(0.3, (log_thresh + 5.0) / 4.0 * 0.3))

        uniform = np.ones_like(exact_dist) / len(exact_dist)
        noisy = (1.0 - noise_level) * exact_dist + noise_level * uniform
        return _distribution_to_counts(noisy, n_shots)

    # ------------------------------------------------------------------
    # Hardness classification
    # ------------------------------------------------------------------
    def classify_hardness(
        self, results: List[Dict[str, Any]]
    ) -> str:
        """Classify a circuit as classically easy, borderline, or hard.

        Decision rule based on the utility loss (Delta_U) across bond
        dimensions:

        - **classically easy**: Delta_U < 0.01 already at chi=16 (a small
          bond dimension suffices to reproduce the quantum utility).
        - **borderline**: Delta_U drops below 0.01 only at chi >= 32, or
          never drops below 0.01 but the maximum Delta_U is < 0.05.
        - **classically hard**: Delta_U remains above 0.05 even at the
          largest tested bond dimension.
        """
        if not results:
            return "classically easy"

        # Sort by chi.
        sorted_results = sorted(results, key=lambda r: r["chi"])

        # Find Delta_U at chi=16 and at the largest chi.
        delta_at_16 = None
        max_delta = 0.0
        min_delta = float("inf")
        largest_chi_delta = 0.0

        for r in sorted_results:
            delta = r.get("delta_u", 0.0)
            max_delta = max(max_delta, delta)
            min_delta = min(min_delta, delta)
            if r["chi"] <= 16 and delta < 0.01:
                delta_at_16 = delta
            largest_chi_delta = delta

        if min_delta < 0.01 and delta_at_16 is not None:
            return "classically easy"
        if min_delta < 0.01 or max_delta < 0.05:
            return "borderline"
        return "classically hard"


if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.quantum_circuits import build_random_qaoa

    inst = generate_mwis_instance(n=8, density=0.5, seed=42)
    circuit, _, _, _ = build_random_qaoa(inst, p=1, seed=7)

    attack = SimulabilityAttack()
    sweep = attack.run_mps_sweep(circuit, n_shots=128, bond_dims=[8, 16, 32])
    exact = attack.get_exact_counts(circuit, n_shots=1024)

    for chi, counts in sweep.items():
        js = attack.compute_js_divergence(exact, counts)
        print(f"chi={chi:3d}  JS={js:.6f}")

    limit = attack.find_simulability_limit(circuit, inst, bond_dims=[8, 16, 32])
    print(f"Simulability limit: {limit['simulability_limit']}")
    print(f"Hardness: {limit['hardness']}")

    pp = attack.pauli_path_verify(circuit, n_shots=128)
    print(f"Pauli-path verdict: {pp['verdict']} (JS={pp['js_divergence']:.6f})")
