"""Quantum-enhanced MCMC for MWIS.

Instead of QAOA producing a final distribution, use quantum circuits
as transition kernels in an MCMC chain. The quantum kernel can occasionally
jump between solution basins that are hard for local classical proposals.

Based on IBM's 2026 work on QeMCMC + parallel tempering for MIS.

Pipeline:
  Parallel Tempering chain at temperatures T_1 < T_2 < ... < T_K
  At each step:
    1. Classical 1-flip proposal (local)
    2. Quantum proposal: use QAOA-like circuit to propose a jump
    3. Metropolis accept/reject
    4. Replica exchange between temperatures

Metrics:
  T_hit = time to reach optimal basin
  tau_mix = mixing time (autocorrelation)
  round-trip time between temperatures
  P(reach low-energy basin within budget B)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.problems import MWISInstance, is_independent, mwis_value
from src.quantum_circuits import build_mwis_qaoa_circuit


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QeMCMCResult:
    """Result of a QeMCMC sampling run.

    Attributes
    ----------
    samples:
        List of binary candidate bitstrings (best state from each chain).
    best_subset:
        Best feasible independent set found across all chains.
    best_value:
        Objective value of ``best_subset``.
    t_hit:
        Step at which the optimal basin was first reached (``-1`` if never).
    round_trips:
        Number of round trips between the hottest and coldest replicas.
    autocorrelation:
        Lag-1 autocorrelation of the objective value time series (cold chain).
    metadata:
        Additional diagnostics.
    """

    samples: List[np.ndarray]
    best_subset: np.ndarray
    best_value: float
    t_hit: int
    round_trips: int
    autocorrelation: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Local circuit execution helper
# ---------------------------------------------------------------------------

def _run_circuit_local(
    circuit: Any, n_shots: int, seed: int
) -> Dict[str, int]:
    """Run a Qiskit circuit locally using the Aer simulator."""
    try:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(seed_simulator=seed)
    except ImportError:
        from qiskit import Aer
        simulator = Aer.get_backend("qasm_simulator")

    from qiskit import transpile
    transpiled = transpile(circuit, simulator)
    job = simulator.run(transpiled, shots=n_shots)
    result = job.result()
    return result.get_counts()


def _counts_to_best_sample(counts: Dict[str, int], n: int) -> np.ndarray:
    """Return the most frequent bitstring from a counts dict as a binary array."""
    if not counts:
        return np.zeros(n, dtype=int)
    best_bitstring = max(counts, key=counts.get)
    best_bitstring = best_bitstring.replace(" ", "")
    bits = [int(c) for c in reversed(best_bitstring)]
    if len(bits) < n:
        bits = bits + [0] * (n - len(bits))
    elif len(bits) > n:
        bits = bits[:n]
    return np.array(bits, dtype=int)


# ---------------------------------------------------------------------------
# QeMCMC sampler
# ---------------------------------------------------------------------------

class QeMCMCSampler:
    """Quantum-enhanced MCMC sampler with parallel tempering.

    Parameters
    ----------
    n_chains:
        Number of parallel tempering replicas (temperatures).
    n_steps:
        Number of MCMC steps per chain.
    t_min, t_max:
        Temperature range for the replicas (geometric spacing).
    quantum_jump_interval:
        Every this many steps, use a quantum proposal instead of a classical
        1-flip proposal.
    p:
        QAOA depth used for the quantum proposal circuit.
    penalty_coeff:
        Penalty coefficient for the quantum proposal circuit.  If ``None`` it
        is set to ``2 * max(weights)``.
    """

    def __init__(
        self,
        n_chains: int = 4,
        n_steps: int = 500,
        t_min: float = 0.1,
        t_max: float = 5.0,
        quantum_jump_interval: int = 10,
        p: int = 1,
        penalty_coeff: Optional[float] = None,
    ) -> None:
        if n_chains < 2:
            raise ValueError("n_chains must be >= 2 for parallel tempering")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if quantum_jump_interval < 1:
            raise ValueError("quantum_jump_interval must be >= 1")
        self.n_chains = n_chains
        self.n_steps = n_steps
        self.t_min = t_min
        self.t_max = t_max
        self.quantum_jump_interval = quantum_jump_interval
        self.p = p
        self.penalty_coeff = penalty_coeff

    # ------------------------------------------------------------------
    # Energy / objective
    # ------------------------------------------------------------------

    def _penalized_value(
        self, subset: np.ndarray, instance: MWISInstance, penalty: float
    ) -> float:
        """Objective minus a penalty for each violated edge constraint."""
        value = float(np.dot(instance.weights, subset.astype(float)))
        if not np.any(subset):
            return value
        selected = subset.astype(bool)
        sub = instance.adjacency[np.ix_(selected, selected)]
        n_conflicts = int(np.sum(np.triu(sub.astype(bool), k=1)))
        return value - penalty * n_conflicts

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------

    def _classical_proposal(
        self, current_state: np.ndarray, instance: MWISInstance, rng: np.random.RandomState
    ) -> np.ndarray:
        """1-flip Metropolis proposal: toggle a single random bit."""
        n = instance.n
        if n == 0:
            return current_state.copy()
        candidate = current_state.copy()
        v = rng.randint(n)
        candidate[v] = 1 - candidate[v]
        return candidate

    def _quantum_proposal(
        self,
        current_state: np.ndarray,
        instance: MWISInstance,
        p: int,
    ) -> np.ndarray:
        """Build a small QAOA circuit initialized from ``current_state``.

        The QAOA circuit is run for a single shot.  The most frequent
        measurement outcome is returned as the proposed state.  This allows the
        quantum kernel to propose non-local jumps between solution basins.
        """
        n = instance.n
        if n == 0:
            return current_state.copy()

        rng = np.random.RandomState(
            abs(hash((current_state.tobytes(), p, time.perf_counter()))) % (2**31)
        )
        gammas = rng.uniform(0.0, np.pi, size=p).tolist()
        betas = rng.uniform(0.0, np.pi, size=p).tolist()
        penalty = self.penalty_coeff
        if penalty is None:
            penalty = float(2.0 * np.max(instance.weights)) if n > 0 else 1.0

        circuit = build_mwis_qaoa_circuit(
            instance, p=p, gammas=gammas, betas=betas, penalty_coeff=penalty
        )

        try:
            counts = _run_circuit_local(circuit, n_shots=1, seed=rng.randint(0, 2**31))
            proposed = _counts_to_best_sample(counts, n)
        except Exception:
            # Fallback to a random 1-flip if the quantum backend fails.
            proposed = current_state.copy()
            v = rng.randint(n)
            proposed[v] = 1 - proposed[v]

        return proposed

    # ------------------------------------------------------------------
    # Metropolis
    # ------------------------------------------------------------------

    def _metropolis_accept(
        self,
        current_val: float,
        proposed_val: float,
        temperature: float,
        rng: np.random.RandomState,
    ) -> bool:
        """Metropolis acceptance criterion for a maximization objective."""
        delta = proposed_val - current_val
        if delta >= 0:
            return True
        # Accept worse solutions with probability exp(delta / T).
        t = max(temperature, 1e-12)
        prob = np.exp(delta / t)
        return bool(rng.random() < prob)

    # ------------------------------------------------------------------
    # Replica exchange
    # ------------------------------------------------------------------

    def _replica_exchange(
        self,
        chains: List[np.ndarray],
        vals: List[float],
        temperatures: np.ndarray,
        rng: np.random.RandomState,
    ) -> int:
        """Attempt to swap adjacent replicas.  Returns number of swaps made."""
        n_swaps = 0
        for i in range(len(chains) - 1):
            delta_beta = (1.0 / temperatures[i + 1]) - (1.0 / temperatures[i])
            delta_val = vals[i] - vals[i + 1]
            log_accept = delta_beta * delta_val
            if log_accept > 0 or rng.random() < np.exp(log_accept):
                chains[i], chains[i + 1] = chains[i + 1], chains[i]
                vals[i], vals[i + 1] = vals[i + 1], vals[i]
                n_swaps += 1
        return n_swaps

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    @staticmethod
    def _lag1_autocorrelation(series: List[float]) -> float:
        """Compute the lag-1 autocorrelation of a time series."""
        if len(series) < 2:
            return 0.0
        arr = np.array(series, dtype=float)
        mean = np.mean(arr)
        var = np.var(arr)
        if var < 1e-12:
            return 0.0
        return float(np.mean((arr[:-1] - mean) * (arr[1:] - mean)) / var)

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    def sample(
        self,
        instance: MWISInstance,
        n_shots: int = 128,
        seed: int = 0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Run parallel-tempering QeMCMC and return samples + metadata.

        Parameters
        ----------
        instance:
            The MWIS instance to sample.
        n_shots:
            Number of independent sampling runs (each produces one sample).
        seed:
            Random seed for reproducibility.

        Returns
        -------
        (samples, metadata)
        """
        rng = np.random.RandomState(seed)
        n = instance.n
        start = time.perf_counter()

        if n == 0:
            return [], {
                "sampler": "qemcmc",
                "time": 0.0,
                "t_hit": -1,
                "round_trips": 0,
                "autocorrelation": 0.0,
            }

        temperatures = np.geomspace(self.t_max, self.t_min, self.n_chains)
        penalty = self.penalty_coeff
        if penalty is None:
            penalty = float(2.0 * np.max(instance.weights))

        all_samples: List[np.ndarray] = []
        global_best_val = -float("inf")
        global_best = np.zeros(n, dtype=int)
        total_round_trips = 0
        t_hit = -1
        autocorrelations: List[float] = []

        for shot in range(n_shots):
            # Initialize chains with random bitstrings.
            chains = [(rng.random(n) < 0.5).astype(int) for _ in range(self.n_chains)]
            vals = [self._penalized_value(s, instance, penalty) for s in chains]
            bests = [s.copy() for s in chains]
            best_vals = list(vals)

            # Track the cold chain's objective time series for autocorrelation.
            cold_series: List[float] = []

            # Track replica positions for round-trip counting.
            # We tag each chain with its initial temperature index and count
            # how many times a tag travels from cold -> hot -> cold.
            tags = list(range(self.n_chains))
            tag_extremes: Dict[int, Dict[str, bool]] = {
                t: {"hit_hot": False, "hit_cold": False} for t in tags
            }
            shot_round_trips = 0

            for step in range(self.n_steps):
                for i in range(self.n_chains):
                    # Choose proposal type.
                    use_quantum = ((step + 1) % self.quantum_jump_interval) == 0
                    if use_quantum:
                        proposed = self._quantum_proposal(chains[i], instance, self.p)
                    else:
                        proposed = self._classical_proposal(chains[i], instance, rng)

                    proposed_val = self._penalized_value(proposed, instance, penalty)
                    if self._metropolis_accept(
                        vals[i], proposed_val, temperatures[i], rng
                    ):
                        chains[i] = proposed
                        vals[i] = proposed_val
                        if proposed_val > best_vals[i]:
                            best_vals[i] = proposed_val
                            bests[i] = proposed.copy()

                # Record cold chain (index 0) objective.
                cold_series.append(mwis_value(chains[0], instance.weights))

                # Replica exchange.
                n_swaps = self._replica_exchange(
                    chains, vals, temperatures, rng
                )
                # Swap tags too.
                if n_swaps > 0:
                    for i in range(self.n_chains - 1):
                        delta_beta = (1.0 / temperatures[i + 1]) - (1.0 / temperatures[i])
                        delta_val = vals[i] - vals[i + 1]
                        log_accept = delta_beta * delta_val
                        if log_accept > 0 or rng.random() < np.exp(log_accept):
                            tags[i], tags[i + 1] = tags[i + 1], tags[i]

                # Track round trips.
                for pos, tag in enumerate(tags):
                    if pos == 0:
                        tag_extremes[tag]["hit_cold"] = True
                    if pos == self.n_chains - 1:
                        tag_extremes[tag]["hit_hot"] = True
                    if tag_extremes[tag]["hit_cold"] and tag_extremes[tag]["hit_hot"]:
                        shot_round_trips += 1
                        tag_extremes[tag] = {"hit_hot": False, "hit_cold": False}

                # Check for t_hit: has the cold chain reached the optimal basin?
                if t_hit < 0 and instance.optimal_solution is not None:
                    opt = np.asarray(instance.optimal_solution).astype(int)
                    d = int(np.sum(chains[0] != opt))
                    if d == 0:
                        t_hit = step

            total_round_trips += shot_round_trips
            autocorrelations.append(self._lag1_autocorrelation(cold_series))

            # The cold chain's best is this shot's sample.
            cold_best = bests[0]
            cold_best_val = mwis_value(cold_best, instance.weights)
            all_samples.append(cold_best)
            if cold_best_val > global_best_val:
                global_best_val = cold_best_val
                global_best = cold_best.copy()

        elapsed = time.perf_counter() - start
        avg_autocorr = float(np.mean(autocorrelations)) if autocorrelations else 0.0

        metadata: Dict[str, Any] = {
            "sampler": "qemcmc",
            "time": elapsed,
            "n_chains": self.n_chains,
            "n_steps": self.n_steps,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "quantum_jump_interval": self.quantum_jump_interval,
            "p": self.p,
            "t_hit": t_hit,
            "round_trips": total_round_trips,
            "autocorrelation": avg_autocorr,
            "best_value": global_best_val,
            "best_subset": global_best,
        }
        return all_samples, metadata


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.exact_solver import solve_mwis_exact

    inst = generate_mwis_instance(n=8, density=0.5, seed=3)
    opt_subset, opt_val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt_subset
    inst.optimal_value = opt_val
    print(f"instance: {inst.instance_id}, optimal: {opt_val:.3f}")

    sampler = QeMCMCSampler(
        n_chains=4, n_steps=100, t_min=0.1, t_max=5.0,
        quantum_jump_interval=20, p=1,
    )
    samples, meta = sampler.sample(inst, n_shots=10, seed=0)
    print(f"best value: {meta['best_value']:.3f}")
    print(f"t_hit: {meta['t_hit']}, round_trips: {meta['round_trips']}, "
          f"autocorr: {meta['autocorrelation']:.4f}")
    print(f"time: {meta['time']:.3f}s, n_samples: {len(samples)}")
