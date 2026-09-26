"""Classical samplers for MWIS candidate generation.

Every sampler exposes the same interface:

    sample(instance, n_shots, seed) -> (list[np.ndarray], dict)

The returned list contains ``n_shots`` binary ``np.ndarray`` objects of length
``instance.n``. The metadata dict always contains the keys ``"sampler"`` (the
sampler name) and ``"time"`` (wall-clock seconds), and may contain additional
sampler-specific diagnostics.

The samplers do *not* enforce the independent-set constraint -- they produce
raw candidate bitstrings. Constraint repair / local search is handled
separately by the :mod:`recovery` module. This keeps the sampling stage
decoupled from the post-processing stage and mirrors the way quantum samples
are treated (raw bitstrings first, classical recovery afterwards).
"""

from __future__ import annotations

import abc
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.problems import MWISInstance, is_independent, mwis_value


class BaseSampler(abc.ABC):
    """Abstract base class for all MWIS samplers."""

    name: str = "base"

    @abc.abstractmethod
    def sample(
        self, instance: MWISInstance, n_shots: int, seed: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Generate ``n_shots`` candidate bitstrings.

        Returns
        -------
        (samples, metadata):
            ``samples`` is a list of ``n_shots`` binary ``np.ndarray`` objects.
            ``metadata`` is a dict containing at least ``"sampler"`` and
            ``"time"``.
        """
        raise NotImplementedError


class UniformRandomSampler(BaseSampler):
    """Sampler that draws uniform random bitstrings.

    Each bit is independently set to 1 with probability ``p`` (default 0.5).
    """

    name = "uniform_random"

    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must lie in [0, 1]")
        self.p = p

    def sample(
        self, instance: MWISInstance, n_shots: int, seed: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        start = time.perf_counter()
        n = instance.n
        samples: List[np.ndarray] = []
        for _ in range(n_shots):
            if n == 0:
                samples.append(np.zeros(0, dtype=int))
            else:
                bits = (rng.random(n) < self.p).astype(int)
                samples.append(bits)
        elapsed = time.perf_counter() - start
        meta: Dict[str, Any] = {
            "sampler": self.name,
            "time": elapsed,
            "p": self.p,
        }
        return samples, meta


class GreedySampler(BaseSampler):
    """Randomized greedy sampler with random restarts and local search.

    For each shot the sampler builds an independent set greedily: vertices are
    considered in a random order and added if they do not conflict with the
    already-selected set. A 1-flip local search (add any non-conflicting vertex
    that increases the value) is then applied as a hill-climb. Multiple
    restarts per shot are averaged and the best result is kept.
    """

    name = "greedy"

    def __init__(self, n_restarts: int = 5) -> None:
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        self.n_restarts = n_restarts

    def _greedy_build(
        self, order: np.ndarray, adjacency: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        n = len(order)
        subset = np.zeros(n, dtype=int)
        selected = np.zeros(n, dtype=bool)
        for v in order:
            # Check none of the neighbours are already selected.
            neighbours = adjacency[v].astype(bool)
            if np.any(selected & neighbours):
                continue
            subset[v] = 1
            selected[v] = True
        return subset

    def _local_search(
        self,
        subset: np.ndarray,
        adjacency: np.ndarray,
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        n = len(subset)
        improved = True
        current = subset.copy()
        while improved:
            improved = False
            order = rng.permutation(n)
            for v in order:
                if current[v] == 1:
                    continue
                neighbours = adjacency[v].astype(bool)
                if np.any(current.astype(bool) & neighbours):
                    continue
                # Adding v strictly increases the value (weights > 0).
                current[v] = 1
                improved = True
        return current

    def sample(
        self, instance: MWISInstance, n_shots: int, seed: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        start = time.perf_counter()
        n = instance.n
        adjacency = instance.adjacency
        weights = instance.weights
        samples: List[np.ndarray] = []
        for _ in range(n_shots):
            if n == 0:
                samples.append(np.zeros(0, dtype=int))
                continue
            best = np.zeros(n, dtype=int)
            best_val = -np.inf
            for _ in range(self.n_restarts):
                order = rng.permutation(n)
                cand = self._greedy_build(order, adjacency, weights)
                cand = self._local_search(cand, adjacency, weights, rng)
                val = mwis_value(cand, weights)
                if val > best_val:
                    best_val = val
                    best = cand
            samples.append(best)
        elapsed = time.perf_counter() - start
        meta: Dict[str, Any] = {
            "sampler": self.name,
            "time": elapsed,
            "n_restarts": self.n_restarts,
        }
        return samples, meta


class SimulatedAnnealingSampler(BaseSampler):
    """Simulated annealing sampler with 1-flip moves.

    The sampler runs ``n_chains`` independent SA chains, each starting from a
    random bitstring. Each chain performs ``n_steps`` single-bit flips using a
    Metropolis acceptance criterion on the MWIS objective with a soft penalty
    for constraint violations. The temperature follows a geometric schedule
    from ``t_initial`` to ``t_final``.
    """

    name = "simulated_annealing"

    def __init__(
        self,
        n_chains: int = 4,
        n_steps: int = 1000,
        t_initial: float = 2.0,
        t_final: float = 0.01,
        penalty: float = 2.0,
    ) -> None:
        if n_chains < 1:
            raise ValueError("n_chains must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.n_chains = n_chains
        self.n_steps = n_steps
        self.t_initial = t_initial
        self.t_final = t_final
        self.penalty = penalty

    def _penalized_value(
        self, subset: np.ndarray, adjacency: np.ndarray, weights: np.ndarray
    ) -> float:
        """Objective minus a penalty for each violated edge constraint."""
        value = float(np.dot(weights, subset.astype(float)))
        if not np.any(subset):
            return value
        selected = subset.astype(bool)
        sub = adjacency[np.ix_(selected, selected)]
        n_conflicts = int(np.sum(np.triu(sub.astype(bool), k=1)))
        return value - self.penalty * n_conflicts

    def _run_chain(
        self,
        adjacency: np.ndarray,
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        n = weights.shape[0]
        current = (rng.random(n) < 0.5).astype(int)
        current_val = self._penalized_value(current, adjacency, weights)
        best = current.copy()
        best_val = current_val
        temps = np.linspace(self.t_initial, self.t_final, self.n_steps)
        for t in temps:
            if n == 0:
                break
            v = rng.randint(n)
            candidate = current.copy()
            candidate[v] = 1 - candidate[v]
            cand_val = self._penalized_value(candidate, adjacency, weights)
            delta = cand_val - current_val
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-12)):
                current = candidate
                current_val = cand_val
                if current_val > best_val:
                    best_val = current_val
                    best = current.copy()
        return best

    def sample(
        self, instance: MWISInstance, n_shots: int, seed: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        start = time.perf_counter()
        n = instance.n
        adjacency = instance.adjacency
        weights = instance.weights
        samples: List[np.ndarray] = []
        for _ in range(n_shots):
            if n == 0:
                samples.append(np.zeros(0, dtype=int))
                continue
            best = np.zeros(n, dtype=int)
            best_val = -np.inf
            for _ in range(self.n_chains):
                cand = self._run_chain(adjacency, weights, rng)
                # Evaluate on the true (unpenalized) objective.
                val = mwis_value(cand, weights)
                if val > best_val:
                    best_val = val
                    best = cand
            samples.append(best)
        elapsed = time.perf_counter() - start
        meta: Dict[str, Any] = {
            "sampler": self.name,
            "time": elapsed,
            "n_chains": self.n_chains,
            "n_steps": self.n_steps,
            "t_initial": self.t_initial,
            "t_final": self.t_final,
            "penalty": self.penalty,
        }
        return samples, meta


class ParallelTemperingSampler(BaseSampler):
    """Parallel tempering sampler.

    Runs ``n_chains`` SA chains in parallel at geometrically spaced
    temperatures. After every ``swap_interval`` steps, adjacent chains attempt
    a replica-exchange swap using the standard Metropolis criterion. The
    lowest-temperature chain's best state is reported for each shot.
    """

    name = "parallel_tempering"

    def __init__(
        self,
        n_chains: int = 6,
        n_steps: int = 1000,
        t_min: float = 0.01,
        t_max: float = 4.0,
        penalty: float = 2.0,
        swap_interval: int = 10,
    ) -> None:
        if n_chains < 2:
            raise ValueError("n_chains must be >= 2 for parallel tempering")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if swap_interval < 1:
            raise ValueError("swap_interval must be >= 1")
        self.n_chains = n_chains
        self.n_steps = n_steps
        self.t_min = t_min
        self.t_max = t_max
        self.penalty = penalty
        self.swap_interval = swap_interval

    def _penalized_value(
        self, subset: np.ndarray, adjacency: np.ndarray, weights: np.ndarray
    ) -> float:
        value = float(np.dot(weights, subset.astype(float)))
        if not np.any(subset):
            return value
        selected = subset.astype(bool)
        sub = adjacency[np.ix_(selected, selected)]
        n_conflicts = int(np.sum(np.triu(sub.astype(bool), k=1)))
        return value - self.penalty * n_conflicts

    def _run(
        self,
        adjacency: np.ndarray,
        weights: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        n = weights.shape[0]
        if n == 0:
            return np.zeros(0, dtype=int)
        temps = np.geomspace(self.t_max, self.t_min, self.n_chains)
        states = [(rng.random(n) < 0.5).astype(int) for _ in range(self.n_chains)]
        vals = [self._penalized_value(s, adjacency, weights) for s in states]
        bests = [s.copy() for s in states]
        best_vals = list(vals)

        for step in range(self.n_steps):
            # 1-flip Metropolis update for each chain at its own temperature.
            for i in range(self.n_chains):
                v = rng.randint(n)
                candidate = states[i].copy()
                candidate[v] = 1 - candidate[v]
                cand_val = self._penalized_value(candidate, adjacency, weights)
                delta = cand_val - vals[i]
                t = temps[i]
                if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-12)):
                    states[i] = candidate
                    vals[i] = cand_val
                    if cand_val > best_vals[i]:
                        best_vals[i] = cand_val
                        bests[i] = candidate.copy()

            # Replica exchange between adjacent chains.
            if (step + 1) % self.swap_interval == 0:
                for i in range(self.n_chains - 1):
                    # Standard PT swap criterion.
                    delta_beta = (1.0 / temps[i + 1]) - (1.0 / temps[i])
                    delta_val = vals[i] - vals[i + 1]
                    log_accept = delta_beta * delta_val
                    if log_accept > 0 or rng.random() < np.exp(log_accept):
                        states[i], states[i + 1] = states[i + 1], states[i]
                        vals[i], vals[i + 1] = vals[i + 1], vals[i]

        # Return the best state found by the coldest chain.
        coldest = 0
        return bests[coldest]

    def sample(
        self, instance: MWISInstance, n_shots: int, seed: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        start = time.perf_counter()
        n = instance.n
        adjacency = instance.adjacency
        weights = instance.weights
        samples: List[np.ndarray] = []
        for _ in range(n_shots):
            if n == 0:
                samples.append(np.zeros(0, dtype=int))
                continue
            cand = self._run(adjacency, weights, rng)
            samples.append(cand)
        elapsed = time.perf_counter() - start
        meta: Dict[str, Any] = {
            "sampler": self.name,
            "time": elapsed,
            "n_chains": self.n_chains,
            "n_steps": self.n_steps,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "penalty": self.penalty,
            "swap_interval": self.swap_interval,
        }
        return samples, meta


if __name__ == "__main__":
    from problems import generate_mwis_instance

    inst = generate_mwis_instance(n=12, density=0.4, seed=99)
    print(f"instance: {inst.instance_id}, weights sum={np.sum(inst.weights):.2f}")
    for sampler in [
        UniformRandomSampler(),
        GreedySampler(n_restarts=5),
        SimulatedAnnealingSampler(n_chains=4, n_steps=500),
        ParallelTemperingSampler(n_chains=6, n_steps=500),
    ]:
        samples, meta = sampler.sample(inst, n_shots=20, seed=7)
        values = [mwis_value(s, inst.weights) for s in samples]
        n_indep = sum(is_independent(s, inst.adjacency) for s in samples)
        print(
            f"{sampler.name:22s} | best={max(values):7.3f} | "
            f"mean={np.mean(values):7.3f} | indep={n_indep}/{len(samples)} | "
            f"time={meta['time']*1e3:7.3f} ms"
        )
