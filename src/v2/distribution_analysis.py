"""Full distribution analysis using BlueQubit's ability to return
the complete probability distribution (shots=None).

Separate two questions:
  1. Is the quantum distribution better at all?
  2. Can we observe the difference with 200 shots?

For each circuit, compute ideal P_theta(x), then analytically resample
at N_shots = 16, 32, 64, 128, 256, 512, 1000 without re-simulating.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.metrics import compute_metrics
from src.recovery import recover

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

__all__ = ["DistributionAnalysis"]


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


def _distribution_to_counts(dist: np.ndarray, n_shots: int) -> Dict[str, int]:
    """Convert a probability vector to integer counts (multinomial)."""
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


def _adapter_usable(adapter: Any) -> bool:
    """Check whether the adapter has a live BlueQubit client."""
    client = getattr(adapter, "_client", None)
    return client is not None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class DistributionAnalysis:
    """Full distribution analysis with analytic resampling.

    This class separates two questions:

    1. **Ideal advantage**: Is the quantum distribution *better at all*?
       Answered by comparing the full ideal distribution P_theta(x) against
       a classical baseline — no shot noise.

    2. **Observable advantage**: Can we *observe* the difference with a
       realistic number of shots?  Answered by resampling the ideal
       distribution at various shot counts and measuring the variability
       of the downstream utility.
    """

    def __init__(self, bluequbit_adapter: Any | None = None) -> None:
        self.adapter = bluequbit_adapter
        self._cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Exact distribution
    # ------------------------------------------------------------------
    def get_exact_distribution(self, circuit: Any) -> np.ndarray:
        """Return the exact probability distribution P(x) as a vector.

        Uses BlueQubit with ``shots=None`` (or a very high shot count) when
        available, otherwise falls back to a local statevector simulation.
        """
        n_qubits = circuit.num_qubits
        cache_key = id(circuit)
        if cache_key in self._cache:
            return self._cache[cache_key]

        dist: np.ndarray | None = None

        if self.adapter is not None and _adapter_usable(self.adapter):
            try:
                dist = self._get_distribution_bluequbit(circuit, n_qubits)
            except Exception as exc:
                logger.warning("distribution.bluequbit_failed", error=repr(exc))

        if dist is None:
            dist = self._get_distribution_local(circuit)

        self._cache[cache_key] = dist
        return dist

    def _get_distribution_bluequbit(
        self, circuit: Any, n_qubits: int
    ) -> np.ndarray | None:
        """Get the exact distribution from BlueQubit.

        Tries ``shots=None`` first (returns the full distribution).  If that
        fails (some devices require integer shots), falls back to a very high
        shot count and normalises.
        """
        client = self.adapter._require_client()  # type: ignore[attr-defined]
        device = self.adapter.select_device(n_qubits, prefer_gpu=True)

        # Try shots=None for the exact distribution.
        try:
            result = client.run(circuit, device=device, shots=None)
            counts = result.get_counts()
            if counts:
                return _counts_to_distribution(counts, n_qubits)
        except Exception as exc:
            logger.warning("distribution.shots_none_failed", error=repr(exc))

        # Fallback: high shot count.
        high_shots = max(1 << n_qubits, 8192)
        result = client.run(circuit, device=device, shots=high_shots)
        counts = result.get_counts()
        if counts:
            return _counts_to_distribution(counts, n_qubits)
        return None

    @staticmethod
    def _get_distribution_local(circuit: Any) -> np.ndarray:
        """Get the exact distribution via local statevector simulation."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        sim = AerSimulator(method="statevector")
        circ_no_meas = circuit.copy()
        circ_no_meas.remove_final_measurements()
        circ_no_meas.save_statevector()
        t_circ = transpile(circ_no_meas, sim)
        result = sim.run(t_circ).result()
        statevector = result.data(0)["statevector"]
        probs = np.abs(statevector) ** 2
        return np.real(probs).astype(float)

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    def resample(
        self,
        distribution: np.ndarray,
        n_shots: int,
        seed: int = 0,
    ) -> Dict[str, int]:
        """Sample ``n_shots`` from ``distribution`` (multinomial).

        This is purely analytic — no re-simulation is needed.  The seed
        controls reproducibility.
        """
        rng = np.random.RandomState(seed)
        n_qubits = int(round(math.log2(len(distribution)))) if len(distribution) > 1 else 0

        probs = np.maximum(distribution, 0.0)
        total = probs.sum()
        if total <= 0:
            # Uniform fallback.
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total

        # Use the seeded RNG for the multinomial draw.
        # numpy's multinomial doesn't accept a RandomState directly, so we
        # set the global seed via the RandomState's internal state.
        np.random.set_state(rng.get_state())
        counts_arr = np.random.multinomial(n_shots, probs)

        counts: Dict[str, int] = {}
        for idx, c in enumerate(counts_arr):
            if c > 0:
                bitstring = format(idx, f"0{n_qubits}b")
                counts[bitstring] = int(c)
        return counts

    # ------------------------------------------------------------------
    # Shots sweep
    # ------------------------------------------------------------------
    def shots_sweep(
        self,
        distribution: np.ndarray,
        instance: Any,
        n_shots_list: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024),
        n_trials: int = 10,
        recovery_budget: int = 5000,
    ) -> Dict[int, Dict[str, float]]:
        """For each n_shots, resample n_trials times and compute utility.

        Returns a dict mapping ``n_shots -> {mean, std, min, max, trials}``
        of the post-recovery utility.
        """
        n = instance.n
        results: Dict[int, Dict[str, float]] = {}

        for n_shots in n_shots_list:
            utilities: List[float] = []
            for trial in range(n_trials):
                counts = self.resample(distribution, n_shots, seed=trial)
                samples = _counts_to_samples(counts, n)

                if not samples:
                    utilities.append(0.0)
                    continue

                rec = recover(
                    samples,
                    instance,
                    max_expansion=2,
                    max_candidates=recovery_budget,
                )
                utilities.append(float(rec["post_recovery_best"]))

            arr = np.array(utilities, dtype=float)
            results[n_shots] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n_trials": n_trials,
                "utilities": utilities,
            }

        return results

    # ------------------------------------------------------------------
    # Minimum shots for signal
    # ------------------------------------------------------------------
    def min_shots_for_signal(
        self,
        distribution: np.ndarray,
        classical_baseline: float,
        instance: Any,
        threshold: float = 0.05,
        n_shots_list: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024),
        n_trials: int = 20,
        recovery_budget: int = 5000,
    ) -> Dict[str, Any]:
        """Find the minimum shots where the quantum advantage signal is
        detectable.

        The "signal" is the quantum post-recovery utility exceeding the
        classical baseline by at least ``threshold`` (relative), with the
        lower bound of the trial distribution still above the baseline.

        Returns a dict with ``min_shots`` (or None), ``per_shots`` detail,
        and ``classical_baseline``.
        """
        sweep = self.shots_sweep(
            distribution,
            instance,
            n_shots_list=n_shots_list,
            n_trials=n_trials,
            recovery_budget=recovery_budget,
        )

        per_shots: List[Dict[str, Any]] = []
        min_shots: int | None = None

        for n_shots in n_shots_list:
            stats = sweep[n_shots]
            mean_util = stats["mean"]
            std_util = stats["std"]

            # Signal: mean exceeds baseline by threshold (relative).
            if classical_baseline > 0:
                relative_gain = (mean_util - classical_baseline) / classical_baseline
            else:
                relative_gain = mean_util / max(abs(mean_util), 1e-9)

            # Conservative: lower bound of mean still above baseline.
            lower_bound = mean_util - std_util
            detectable = relative_gain >= threshold and lower_bound > classical_baseline

            entry = {
                "n_shots": n_shots,
                "mean_utility": mean_util,
                "std_utility": std_util,
                "classical_baseline": classical_baseline,
                "relative_gain": relative_gain,
                "lower_bound": lower_bound,
                "detectable": detectable,
            }
            per_shots.append(entry)

            if min_shots is None and detectable:
                min_shots = n_shots

        return {
            "min_shots": min_shots,
            "per_shots": per_shots,
            "classical_baseline": classical_baseline,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Ideal advantage (no shot noise)
    # ------------------------------------------------------------------
    def ideal_advantage(
        self,
        distribution: np.ndarray,
        classical_baseline: float,
        instance: Any,
    ) -> Dict[str, float]:
        """Compute the ideal (shot-noise-free) quantum advantage.

        Compares the probability-weighted expected objective of the ideal
        quantum distribution against the classical baseline.
        """
        n_qubits = int(round(math.log2(len(distribution)))) if len(distribution) > 1 else 0
        weights = np.asarray(instance.weights, dtype=float)

        if n_qubits == 0:
            return {
                "ideal_quantum_utility": 0.0,
                "classical_baseline": classical_baseline,
                "ideal_advantage": 0.0,
            }

        # Expected objective under the quantum distribution.
        # E[obj(x)] = sum_x P(x) * obj(x)
        # For large n this is expensive; we approximate by sampling from
        # the high-probability region.
        top_k = min(1000, len(distribution))
        top_indices = np.argsort(distribution)[-top_k:]
        top_probs = distribution[top_indices]
        top_probs = top_probs / max(top_probs.sum(), 1e-15)

        # Expected objective over the top-k states.
        expected_obj = 0.0
        for idx, p in zip(top_indices, top_probs):
            bits = format(int(idx), f"0{n_qubits}b")
            subset = np.array([int(b) for b in reversed(bits)], dtype=int)
            obj = float(np.dot(weights, subset.astype(float)))
            expected_obj += p * obj

        advantage = expected_obj - classical_baseline

        return {
            "ideal_quantum_utility": float(expected_obj),
            "classical_baseline": float(classical_baseline),
            "ideal_advantage": float(advantage),
        }


if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.quantum_circuits import build_random_qaoa

    inst = generate_mwis_instance(n=8, density=0.5, seed=42)
    circuit, _, _, _ = build_random_qaoa(inst, p=1, seed=7)

    da = DistributionAnalysis()
    dist = da.get_exact_distribution(circuit)
    print(f"Distribution size: {len(dist)}, sum={dist.sum():.6f}")
    print(f"Top-5 probs: {np.sort(dist)[-5:]}")

    sweep = da.shots_sweep(dist, inst, n_shots_list=[16, 64, 256], n_trials=5)
    for n_shots, stats in sweep.items():
        print(f"  n_shots={n_shots:4d}  mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    min_shots = da.min_shots_for_signal(dist, classical_baseline=10.0, instance=inst)
    print(f"Min shots for signal: {min_shots['min_shots']}")
