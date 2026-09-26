"""Oracle Test: give quantum an unfair advantage.

For each graph, search over 10^3-10^4 quantum configurations
(gamma, beta, p, penalty, mixer, representation, ordering)
and select the BEST per-instance quantum config in hindsight.

Question: max_theta U_Q(G, theta) > U_best_classical(G)?

If NO even with oracle, there's no point training WestQuant on this family.
If YES with oracle but NO with general config, it's a perfect ML problem:
  Can WestQuant predict the oracle-like representation from graph structure?
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.problems import MWISInstance, mwis_value
from src.quantum_circuits import build_mwis_qaoa_circuit
from src.recovery import recover
from src.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OracleResult:
    """Result of an oracle quantum configuration search.

    Attributes
    ----------
    best_config:
        The best quantum configuration found (dict of parameters).
    best_metrics:
        Metrics computed for the best configuration's samples.
    best_recovery:
        Recovery result for the best configuration.
    n_configs_searched:
        Total number of configurations evaluated.
    beats_classical:
        Whether the best quantum config beats the best classical result.
    margin:
        ``U_Q(best) - U_best_classical`` (positive favours quantum).
    all_configs:
        Optional list of all evaluated configs with their metrics.
    """

    best_config: Dict[str, Any]
    best_metrics: Dict[str, Any]
    best_recovery: Dict[str, Any]
    n_configs_searched: int
    beats_classical: bool = False
    margin: float = 0.0
    all_configs: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Local Aer simulation fallback
# ---------------------------------------------------------------------------

def _run_circuit_local(
    circuit: Any, n_shots: int, seed: int
) -> Dict[str, int]:
    """Run a Qiskit circuit locally using the Aer simulator.

    Returns a counts dict mapping bitstring -> count.
    """
    try:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(seed_simulator=seed)
    except ImportError:
        # Fallback to the legacy Aer provider path.
        from qiskit import Aer
        simulator = Aer.get_backend("qasm_simulator")

    from qiskit import transpile
    transpiled = transpile(circuit, simulator)
    job = simulator.run(transpiled, shots=n_shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def _counts_to_samples(
    counts: Dict[str, int], n: int
) -> List[np.ndarray]:
    """Convert a Qiskit counts dict to a list of binary arrays.

    Qiskit bitstrings are big-endian (leftmost char = highest qubit index).
    We reverse so that index 0 corresponds to vertex 0.
    """
    samples: List[np.ndarray] = []
    for bitstring, count in counts.items():
        # Remove spaces if present.
        bitstring = bitstring.replace(" ", "")
        # Reverse to get qubit 0 on the left.
        bits = [int(c) for c in reversed(bitstring)]
        if len(bits) < n:
            bits = bits + [0] * (n - len(bits))
        elif len(bits) > n:
            bits = bits[:n]
        arr = np.array(bits, dtype=int)
        for _ in range(count):
            samples.append(arr.copy())
    return samples


# ---------------------------------------------------------------------------
# Oracle search
# ---------------------------------------------------------------------------

class OracleSearch:
    """Search over quantum configurations to find the best per-instance config.

    Parameters
    ----------
    instance:
        The MWIS instance to search over.
    n_configs:
        Number of random configurations to evaluate (default 1000).
    """

    def __init__(self, instance: MWISInstance, n_configs: int = 1000) -> None:
        if n_configs < 1:
            raise ValueError("n_configs must be >= 1")
        self.instance = instance
        self.n_configs = n_configs
        self._configs: List[Dict[str, Any]] = []

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """Generate ``n_configs`` random QAOA configurations.

        Each config varies:
        - ``p`` in {1, 2, 3}
        - ``gammas``: random in [0, pi)
        - ``betas``: random in [0, pi)
        - ``penalty_coeff``: 1x-4x max weight
        - ``mixer``: "x-mixer" (the standard QAOA mixer)
        - ``ordering``: identity, reverse, or random permutation of variables
        """
        rng = np.random.RandomState(42)
        max_weight = float(np.max(self.instance.weights)) if self.instance.n > 0 else 1.0
        configs: List[Dict[str, Any]] = []
        for _ in range(self.n_configs):
            p = int(rng.choice([1, 2, 3]))
            gammas = rng.uniform(0.0, np.pi, size=p).tolist()
            betas = rng.uniform(0.0, np.pi, size=p).tolist()
            penalty_mult = float(rng.uniform(1.0, 4.0))
            penalty_coeff = penalty_mult * max_weight
            ordering_choice = rng.choice(["identity", "reverse", "random"])
            if ordering_choice == "identity":
                ordering = list(range(self.instance.n))
            elif ordering_choice == "reverse":
                ordering = list(reversed(range(self.instance.n)))
            else:
                ordering = rng.permutation(self.instance.n).tolist()
            configs.append({
                "p": p,
                "gammas": gammas,
                "betas": betas,
                "penalty_coeff": penalty_coeff,
                "mixer": "x-mixer",
                "ordering": ordering,
            })
        self._configs = configs
        return configs

    def _reorder_instance(
        self, ordering: List[int]
    ) -> MWISInstance:
        """Build a permuted copy of the instance according to ``ordering``.

        ``ordering[i]`` gives the original vertex index that maps to position
        ``i`` in the permuted instance.
        """
        n = self.instance.n
        if ordering == list(range(n)):
            return self.instance
        idx = np.array(ordering, dtype=int)
        adj = self.instance.adjacency
        new_adj = adj[np.ix_(idx, idx)].copy()
        new_weights = self.instance.weights[idx].copy()
        return MWISInstance(
            instance_id=self.instance.instance_id + "_oracle",
            n=n,
            density=self.instance.density,
            seed=self.instance.seed,
            adjacency=new_adj,
            weights=new_weights,
            optimal_solution=None,
            optimal_value=self.instance.optimal_value,
            graph_family=self.instance.graph_family,
            hardness=self.instance.hardness,
        )

    def _unreorder_samples(
        self, samples: List[np.ndarray], ordering: List[int]
    ) -> List[np.ndarray]:
        """Map samples from the permuted ordering back to the original ordering."""
        n = self.instance.n
        if ordering == list(range(n)):
            return samples
        # inverse permutation
        inv = np.zeros(n, dtype=int)
        for i, v in enumerate(ordering):
            inv[v] = i
        return [np.asarray(s)[inv].copy() for s in samples]

    def _evaluate_config(
        self,
        config: Dict[str, Any],
        adapter: Optional[Any],
        n_shots: int,
    ) -> Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Build a circuit for ``config``, run it, recover, and compute metrics.

        Returns
        -------
        (samples, metrics, recovery_result)
        """
        ordering = config["ordering"]
        permuted_instance = self._reorder_instance(ordering)
        circuit = build_mwis_qaoa_circuit(
            permuted_instance,
            p=config["p"],
            gammas=config["gammas"],
            betas=config["betas"],
            penalty_coeff=config["penalty_coeff"],
        )

        seed = abs(hash((tuple(config["gammas"]), tuple(config["betas"])))) % (2**31)

        if adapter is not None:
            result_dict = adapter.run(
                circuit, device="cpu", shots=n_shots,
                job_name=f"oracle_{self.instance.instance_id}",
            )
            counts = result_dict.get("counts", {})
        else:
            counts = _run_circuit_local(circuit, n_shots, seed)

        permuted_samples = _counts_to_samples(counts, permuted_instance.n)
        samples = self._unreorder_samples(permuted_samples, ordering)

        recovery_result = recover(
            samples, self.instance, max_expansion=2, max_candidates=50000
        )
        metrics = compute_metrics(samples, self.instance, recovery_result=recovery_result)
        return samples, metrics, recovery_result

    def search(
        self,
        bluequbit_adapter: Optional[Any] = None,
        n_shots: int = 128,
    ) -> OracleResult:
        """Search over ``n_configs`` random QAOA configs and return the best.

        Parameters
        ----------
        bluequbit_adapter:
            Optional :class:`BlueQubitAdapter` for running circuits on
            BlueQubit.  If ``None``, a local Aer simulator is used.
        n_shots:
            Number of shots per configuration.

        Returns
        -------
        OracleResult
            The best configuration found and its metrics.
        """
        configs = self._generate_configs()
        best_config: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None
        best_recovery: Optional[Dict[str, Any]] = None
        best_value = -float("inf")
        all_configs: List[Dict[str, Any]] = []

        for i, config in enumerate(configs):
            try:
                samples, metrics, recovery = self._evaluate_config(
                    config, bluequbit_adapter, n_shots
                )
            except Exception:
                # Skip configs that fail (e.g. circuit too deep for backend).
                all_configs.append({"config": config, "post_recovery_best": 0.0,
                                    "error": True})
                continue

            post_val = float(metrics.get("post_recovery_best", 0.0))
            all_configs.append({
                "config": config,
                "post_recovery_best": post_val,
                "pre_recovery_best": metrics.get("pre_recovery_best", 0.0),
                "diversity": metrics.get("diversity", 0.0),
                "p_hit": metrics.get("p_hit", 0.0),
            })

            if post_val > best_value:
                best_value = post_val
                best_config = config
                best_metrics = metrics
                best_recovery = recovery

        if best_config is None:
            best_config = configs[0] if configs else {}
            best_metrics = {}
            best_recovery = {}

        return OracleResult(
            best_config=best_config,
            best_metrics=best_metrics if best_metrics is not None else {},
            best_recovery=best_recovery if best_recovery is not None else {},
            n_configs_searched=len(configs),
            all_configs=all_configs,
        )

    def oracle_vs_classical(
        self,
        classical_metrics: Dict[str, Any],
        bluequbit_adapter: Optional[Any] = None,
        n_shots: int = 128,
    ) -> OracleResult:
        """Search for the best quantum config and compare to classical.

        Parameters
        ----------
        classical_metrics:
            Metrics dict for the best classical sampler (must contain
            ``post_recovery_best`` or ``pre_recovery_best``).
        bluequbit_adapter:
            Optional BlueQubit adapter.
        n_shots:
            Shots per quantum config.

        Returns
        -------
        OracleResult
            Updated with ``beats_classical`` and ``margin``.
        """
        result = self.search(bluequbit_adapter=bluequbit_adapter, n_shots=n_shots)
        classical_val = float(
            classical_metrics.get(
                "post_recovery_best",
                classical_metrics.get("pre_recovery_best", 0.0),
            )
        )
        quantum_val = float(
            result.best_metrics.get(
                "post_recovery_best",
                result.best_metrics.get("pre_recovery_best", 0.0),
            )
        )
        result.margin = quantum_val - classical_val
        result.beats_classical = quantum_val > classical_val + 1e-9
        return result


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.exact_solver import solve_mwis_exact
    from src.classical_samplers import GreedySampler

    inst = generate_mwis_instance(n=8, density=0.5, seed=3)
    opt_subset, opt_val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt_subset
    inst.optimal_value = opt_val
    print(f"instance: {inst.instance_id}, optimal: {opt_val:.3f}")

    # Classical baseline.
    c_samples, _ = GreedySampler().sample(inst, n_shots=50, seed=5)
    c_recovery = recover(c_samples, inst, max_expansion=2, max_candidates=5000)
    c_metrics = compute_metrics(c_samples, inst, recovery_result=c_recovery)
    print(f"classical post-recovery best: {c_metrics['post_recovery_best']:.3f}")

    # Oracle search (small for smoke test).
    oracle = OracleSearch(inst, n_configs=20)
    result = oracle.oracle_vs_classical(c_metrics, n_shots=64)
    print(f"oracle best post-recovery: "
          f"{result.best_metrics.get('post_recovery_best', 0.0):.3f}")
    print(f"beats classical: {result.beats_classical}, margin: {result.margin:.4f}")
    print(f"best config: p={result.best_config.get('p')}, "
          f"penalty={result.best_config.get('penalty_coeff', 0):.3f}, "
          f"ordering={result.best_config.get('ordering')}")
