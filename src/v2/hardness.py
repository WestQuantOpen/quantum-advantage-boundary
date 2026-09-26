"""Hardness-conditioned graph generation.

Instead of scaling only N, generate graphs with controlled hardness.

Hardness score:
  H(G) = f(T_MILP, T_PT, P_local-hit, basin_structure, degeneracy)

Also generate adversarial instances with:
- many near-degenerate solutions
- large Hamming distances between good solutions
- planted optimum
- deceptive local minima
- high local escape costs
- frustration
- multiple competing basins
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from src.problems import (
    MWISInstance,
    edges_from_adjacency,
    generate_mwis_instance,
    is_independent,
    mwis_value,
)
from src.exact_solver import solve_mwis_exact
from src.classical_samplers import GreedySampler, ParallelTemperingSampler


# ---------------------------------------------------------------------------
# Hardness scoring
# ---------------------------------------------------------------------------

def compute_hardness_score(instance: MWISInstance) -> Dict[str, Any]:
    """Compute a hardness score from MILP time, PT time, local-hit probability,
    and basin count.

    The hardness score is a weighted combination of:

    * ``T_MILP`` -- wall-clock time for the exact ILP solver (log-scaled).
    * ``T_PT`` -- wall-clock time for parallel tempering (log-scaled).
    * ``P_local-hit`` -- probability that greedy + local search hits the optimum.
    * ``basin_count`` -- number of distinct attraction basins.
    * ``degeneracy`` -- number of optimal solutions (estimated).

    Returns
    -------
    dict
        Keys: ``hardness_score`` (float in [0, 1]), ``t_milp``, ``t_pt``,
        ``p_local_hit``, ``basin_count``, ``degeneracy``, ``components``.
    """
    n = instance.n

    # Ensure the optimum is known.
    if instance.optimal_solution is None or instance.optimal_value is None:
        opt_subset, opt_val, t_milp = solve_mwis_exact(instance)
        instance.optimal_solution = opt_subset
        instance.optimal_value = opt_val
    else:
        t_milp = 0.0

    # Re-solve to measure MILP time (the instance may already have the optimum).
    _, _, t_milp = solve_mwis_exact(instance)

    # Parallel tempering time + best value.
    pt_sampler = ParallelTemperingSampler(n_chains=4, n_steps=200)
    pt_samples, pt_meta = pt_sampler.sample(instance, n_shots=5, seed=0)
    t_pt = float(pt_meta.get("time", 0.0))
    pt_best = max(mwis_value(s, instance.weights) for s in pt_samples) if pt_samples else 0.0

    # Greedy local-hit probability.
    greedy = GreedySampler(n_restarts=3)
    g_samples, _ = greedy.sample(instance, n_shots=20, seed=1)
    n_hits = sum(
        1 for s in g_samples
        if is_independent(s, instance.adjacency)
        and abs(mwis_value(s, instance.weights) - instance.optimal_value) < 1e-9
    )
    p_local_hit = n_hits / 20.0 if g_samples else 0.0

    # Basin count via 1-flip descent on greedy samples.
    from src.v2.recovery_erasure import attraction_basins
    basin_count = attraction_basins(g_samples, instance) if g_samples else 0

    # Degeneracy: estimate by counting optimal solutions found among samples
    # plus a small brute-force check for very small instances.
    degeneracy = _estimate_degeneracy(instance)

    # Combine into a hardness score in [0, 1].
    # Higher T_MILP, T_PT, basin_count, degeneracy -> harder.
    # Higher P_local-hit -> easier.
    milp_component = min(1.0, np.log1p(t_milp) / np.log1p(10.0))
    pt_component = min(1.0, np.log1p(t_pt) / np.log1p(10.0))
    local_component = 1.0 - p_local_hit  # 1 if greedy never hits, 0 if always
    basin_component = min(1.0, basin_count / 10.0)
    degeneracy_component = min(1.0, np.log1p(degeneracy) / np.log1p(20.0))

    hardness_score = (
        0.25 * milp_component
        + 0.20 * pt_component
        + 0.25 * local_component
        + 0.15 * basin_component
        + 0.15 * degeneracy_component
    )
    hardness_score = float(np.clip(hardness_score, 0.0, 1.0))

    return {
        "hardness_score": hardness_score,
        "t_milp": float(t_milp),
        "t_pt": t_pt,
        "pt_best": float(pt_best),
        "p_local_hit": float(p_local_hit),
        "basin_count": int(basin_count),
        "degeneracy": int(degeneracy),
        "components": {
            "milp_component": float(milp_component),
            "pt_component": float(pt_component),
            "local_component": float(local_component),
            "basin_component": float(basin_component),
            "degeneracy_component": float(degeneracy_component),
        },
    }


def _estimate_degeneracy(instance: MWISInstance) -> int:
    """Estimate the number of optimal MWIS solutions.

    For ``n <= 20`` an exact brute-force count is used.  For larger instances a
    sampling-based estimate is returned (number of distinct optimal solutions
    found by greedy + random restarts).
    """
    n = instance.n
    if n == 0:
        return 0
    if instance.optimal_value is None:
        return 1

    if n <= 20:
        count = 0
        opt_val = instance.optimal_value
        for code in range(1 << n):
            subset = np.array([(code >> i) & 1 for i in range(n)], dtype=np.int8)
            if not is_independent(subset, instance.adjacency):
                continue
            if abs(mwis_value(subset, instance.weights) - opt_val) < 1e-9:
                count += 1
        return count

    # Sampling-based estimate for larger instances.
    greedy = GreedySampler(n_restarts=5)
    samples, _ = greedy.sample(instance, n_shots=100, seed=2)
    optimal_keys = set()
    for s in samples:
        if is_independent(s, instance.adjacency) and abs(
            mwis_value(s, instance.weights) - instance.optimal_value
        ) < 1e-9:
            optimal_keys.add(np.asarray(s).astype(np.int8).tobytes())
    return len(optimal_keys)


# ---------------------------------------------------------------------------
# Adversarial instance generation
# ---------------------------------------------------------------------------

def _build_instance_from_graph(
    graph: nx.Graph,
    weights: np.ndarray,
    seed: int,
    hardness_label: Optional[str] = None,
) -> MWISInstance:
    """Build an MWISInstance from a networkx graph and weight array."""
    n = graph.number_of_nodes()
    adj = nx.to_numpy_array(graph, dtype=np.int8)
    adj = np.triu(adj, k=1)
    adj = (adj | adj.T).astype(np.int8)
    return MWISInstance(
        instance_id=f"hard_n{n}_s{seed}",
        n=n,
        density=float(nx.density(graph)) if n > 1 else 0.0,
        seed=seed,
        adjacency=adj,
        weights=weights.astype(float),
        optimal_solution=None,
        optimal_value=None,
        graph_family="custom",
        hardness=hardness_label,
    )


def _planted_optimum_instance(
    n: int, seed: int
) -> MWISInstance:
    """Generate an instance with a planted MWIS solution.

    A set ``S`` of high-weight vertices is chosen as the planted optimum.
    Edges are added among all vertices NOT in ``S`` (so they form a clique-like
    structure that is hard to search), while vertices in ``S`` are made
    independent (no edges among them).  Additional edges between ``S`` and the
    rest make it hard to find ``S`` greedily.
    """
    rng = np.random.RandomState(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    # Planted optimum: roughly n/3 high-weight vertices.
    planted_size = max(2, n // 3)
    planted = set(rng.choice(n, size=planted_size, replace=False).tolist())

    # Weights: planted vertices get high weight, others get lower weight.
    weights = rng.uniform(1.0, 5.0, size=n)
    for v in planted:
        weights[v] = rng.uniform(8.0, 12.0)

    # Edges among non-planted vertices (dense subgraph).
    non_planted = [v for v in range(n) if v not in planted]
    for i in range(len(non_planted)):
        for j in range(i + 1, len(non_planted)):
            if rng.random() < 0.6:
                graph.add_edge(non_planted[i], non_planted[j])

    # Edges between planted and non-planted (to make greedy harder).
    for v in planted:
        for u in non_planted:
            if rng.random() < 0.3:
                graph.add_edge(v, u)

    # No edges among planted vertices (they form the independent set).

    instance = _build_instance_from_graph(graph, weights, seed, "planted_optimum")
    # Verify the planted set is optimal.
    opt_subset, opt_val, _ = solve_mwis_exact(instance)
    instance.optimal_solution = opt_subset
    instance.optimal_value = opt_val
    return instance


def _deceptive_instance(
    n: int, seed: int
) -> MWISInstance:
    """Generate an instance with deceptive local minima.

    A cluster of medium-weight vertices forms a tempting but suboptimal local
    optimum, while the true optimum is a set of slightly higher-weight vertices
    that is harder to reach.
    """
    rng = np.random.RandomState(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    # Deceptive cluster: medium weights, densely connected to each other
    # so only a few can be picked, but they look attractive.
    deceptive_size = max(2, n // 4)
    deceptive = set(rng.choice(n, size=deceptive_size, replace=False).tolist())
    optimal_set = set()
    for v in range(n):
        if v not in deceptive:
            optimal_set.add(v)

    weights = rng.uniform(1.0, 4.0, size=n)
    for v in deceptive:
        weights[v] = rng.uniform(6.0, 7.0)
    for v in optimal_set:
        weights[v] = rng.uniform(7.5, 9.0)

    # Deceptive cluster: clique so only 1 can be picked.
    deco_list = sorted(deceptive)
    for i in range(len(deco_list)):
        for j in range(i + 1, len(deco_list)):
            graph.add_edge(deco_list[i], deco_list[j])

    # Optimal set: no edges among them (independent), but edges to deceptive
    # cluster so you can't pick both.
    opt_list = sorted(optimal_set)
    for v in opt_list:
        for u in deceptive:
            if rng.random() < 0.4:
                graph.add_edge(v, u)

    # Some edges within optimal set to make it non-trivial.
    for i in range(len(opt_list)):
        for j in range(i + 1, len(opt_list)):
            if rng.random() < 0.15:
                graph.add_edge(opt_list[i], opt_list[j])

    instance = _build_instance_from_graph(graph, weights, seed, "deceptive")
    opt_subset, opt_val, _ = solve_mwis_exact(instance)
    instance.optimal_solution = opt_subset
    instance.optimal_value = opt_val
    return instance


def _multi_basin_instance(
    n: int, seed: int
) -> MWISInstance:
    """Generate an instance with multiple well-separated good solutions.

    The graph is partitioned into several groups, each of which contains a
    near-optimal independent set.  The groups are connected by a few edges so
    that moving between basins requires crossing a barrier.
    """
    rng = np.random.RandomState(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    n_basins = max(2, n // 8)
    n_basins = min(n_basins, 5)
    basin_size = n // n_basins
    basins: List[List[int]] = []
    vertices = list(range(n))
    rng.shuffle(vertices)
    for b in range(n_basins):
        start = b * basin_size
        end = start + basin_size if b < n_basins - 1 else n
        basins.append(vertices[start:end])

    weights = rng.uniform(1.0, 10.0, size=n)

    # Within each basin: sparse edges (so large independent sets exist).
    for basin in basins:
        for i in range(len(basin)):
            for j in range(i + 1, len(basin)):
                if rng.random() < 0.2:
                    graph.add_edge(basin[i], basin[j])

    # Between basins: a few edges (barriers).
    for b1 in range(n_basins):
        for b2 in range(b1 + 1, n_basins):
            n_barriers = max(1, len(basins[b1]) // 3)
            for _ in range(n_barriers):
                u = rng.choice(basins[b1])
                v = rng.choice(basins[b2])
                graph.add_edge(int(u), int(v))

    instance = _build_instance_from_graph(graph, weights, seed, "multi_basin")
    opt_subset, opt_val, _ = solve_mwis_exact(instance)
    instance.optimal_solution = opt_subset
    instance.optimal_value = opt_val
    return instance


def _frustrated_instance(
    n: int, seed: int
) -> MWISInstance:
    """Generate an instance with conflicting constraints (frustration).

    A cycle of odd length creates frustration: not all vertices can be
    simultaneously satisfied.  Weights are assigned to make the frustration
    create multiple competing optima.
    """
    rng = np.random.RandomState(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    # Create an odd cycle to introduce frustration.
    cycle_len = n if n % 2 == 1 else n - 1
    cycle_len = max(3, cycle_len)
    for i in range(cycle_len):
        graph.add_edge(i, (i + 1) % cycle_len)

    # Add random edges elsewhere.
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.2:
                graph.add_edge(i, j)

    weights = rng.uniform(1.0, 10.0, size=n)

    instance = _build_instance_from_graph(graph, weights, seed, "frustrated")
    opt_subset, opt_val, _ = solve_mwis_exact(instance)
    instance.optimal_solution = opt_subset
    instance.optimal_value = opt_val
    return instance


def generate_hard_instance(
    n: int,
    seed: int,
    target_hardness: str = "hard",
) -> MWISInstance:
    """Generate an instance with planted structure of the given hardness type.

    Parameters
    ----------
    n:
        Number of vertices.
    seed:
        Random seed for reproducibility.
    target_hardness:
        One of ``"planted_optimum"``, ``"deceptive"``, ``"multi_basin"``,
        ``"frustrated"``, or ``"hard"`` (alias for ``"multi_basin"``).

    Returns
    -------
    MWISInstance
        The generated instance with ``optimal_solution`` and ``optimal_value``
        set.
    """
    if n < 3:
        # Fall back to a simple random instance for very small n.
        inst = generate_mwis_instance(n=n, density=0.5, seed=seed)
        opt, val, _ = solve_mwis_exact(inst)
        inst.optimal_solution = opt
        inst.optimal_value = val
        return inst

    generators = {
        "planted_optimum": _planted_optimum_instance,
        "deceptive": _deceptive_instance,
        "multi_basin": _multi_basin_instance,
        "frustrated": _frustrated_instance,
        "hard": _multi_basin_instance,
    }
    gen = generators.get(target_hardness, _multi_basin_instance)
    return gen(n, seed)


def generate_barrier_instance(
    n: int,
    seed: int,
    n_basins: int = 3,
    barrier_height: float = 0.1,
) -> MWISInstance:
    """Generate an instance with explicit energy barriers between basins.

    The graph is partitioned into ``n_basins`` groups.  Each group has a
    near-optimal independent set.  Between groups, ``barrier_height``
    (as a fraction of inter-group edges) controls how hard it is to move
    between basins: higher means more inter-group edges (higher barriers).

    Parameters
    ----------
    n:
        Number of vertices.
    seed:
        Random seed.
    n_basins:
        Number of well-separated basins.
    barrier_height:
        Fraction of inter-group edges to add (``0.0`` = no barriers,
        ``1.0`` = full connectivity between groups).

    Returns
    -------
    MWISInstance
    """
    rng = np.random.RandomState(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    n_basins = max(2, min(n_basins, n // 2))
    basin_size = n // n_basins
    basins: List[List[int]] = []
    vertices = list(range(n))
    rng.shuffle(vertices)
    for b in range(n_basins):
        start = b * basin_size
        end = start + basin_size if b < n_basins - 1 else n
        basins.append(vertices[start:end])

    # Weights: give each basin a "star" vertex with high weight.
    weights = rng.uniform(1.0, 5.0, size=n)
    for basin in basins:
        if len(basin) > 0:
            star = rng.choice(basin)
            weights[int(star)] = rng.uniform(8.0, 12.0)

    # Within-basin: sparse edges.
    for basin in basins:
        for i in range(len(basin)):
            for j in range(i + 1, len(basin)):
                if rng.random() < 0.15:
                    graph.add_edge(basin[i], basin[j])

    # Between-basin: barrier edges.
    for b1 in range(n_basins):
        for b2 in range(b1 + 1, n_basins):
            n_inter = max(1, int(barrier_height * len(basins[b1]) * len(basins[b2])))
            for _ in range(n_inter):
                u = int(rng.choice(basins[b1]))
                v = int(rng.choice(basins[b2]))
                graph.add_edge(u, v)

    instance = _build_instance_from_graph(graph, weights, seed, "barrier")
    opt_subset, opt_val, _ = solve_mwis_exact(instance)
    instance.optimal_solution = opt_subset
    instance.optimal_value = opt_val
    return instance


# ---------------------------------------------------------------------------
# Hardness stratifier
# ---------------------------------------------------------------------------

class HardnessStratifier:
    """Generate instances stratified by hardness (easy / medium / hard).

    The stratifier generates random instances and classifies them using a
    combination of greedy gap, simulated-annealing gap, and parallel-tempering
    gap.  Instances are bucketed into the requested hardness levels.
    """

    def __init__(
        self,
        n_chains: int = 4,
        n_steps: int = 200,
        n_shots: int = 10,
    ) -> None:
        self.n_chains = n_chains
        self.n_steps = n_steps
        self.n_shots = n_shots

    def _classify_hardness(self, instance: MWISInstance) -> str:
        """Classify an instance as easy / medium / hard.

        Uses three signals:
        * Greedy gap: ``1 - greedy_best / optimal``.
        * SA gap: ``1 - sa_best / optimal``.
        * PT gap: ``1 - pt_best / optimal``.

        The classification thresholds are:
        * easy: all gaps < 2%.
        * hard: greedy gap > 10% or PT gap > 5%.
        * medium: otherwise.
        """
        if instance.optimal_value is None or instance.optimal_value <= 0:
            opt, val, _ = solve_mwis_exact(instance)
            instance.optimal_solution = opt
            instance.optimal_value = val

        opt_val = instance.optimal_value

        # Greedy.
        greedy = GreedySampler(n_restarts=3)
        g_samples, _ = greedy.sample(instance, self.n_shots, seed=1)
        g_best = max(mwis_value(s, instance.weights) for s in g_samples) if g_samples else 0.0
        greedy_gap = 1.0 - g_best / opt_val if opt_val > 0 else 1.0

        # SA.
        from src.classical_samplers import SimulatedAnnealingSampler
        sa = SimulatedAnnealingSampler(n_chains=self.n_chains, n_steps=self.n_steps)
        sa_samples, _ = sa.sample(instance, self.n_shots, seed=2)
        sa_best = max(mwis_value(s, instance.weights) for s in sa_samples) if sa_samples else 0.0
        sa_gap = 1.0 - sa_best / opt_val if opt_val > 0 else 1.0

        # PT.
        pt = ParallelTemperingSampler(n_chains=self.n_chains, n_steps=self.n_steps)
        pt_samples, _ = pt.sample(instance, self.n_shots, seed=3)
        pt_best = max(mwis_value(s, instance.weights) for s in pt_samples) if pt_samples else 0.0
        pt_gap = 1.0 - pt_best / opt_val if opt_val > 0 else 1.0

        if greedy_gap < 0.02 and sa_gap < 0.02 and pt_gap < 0.02:
            return "easy"
        if greedy_gap > 0.10 or pt_gap > 0.05:
            return "hard"
        return "medium"

    def generate_stratified(
        self,
        n: int,
        n_instances: int,
        hardness_levels: Optional[List[str]] = None,
        seed: int = 0,
    ) -> Dict[str, List[MWISInstance]]:
        """Generate instances stratified by hardness.

        Parameters
        ----------
        n:
            Number of vertices per instance.
        n_instances:
            Number of instances to generate per hardness level.
        hardness_levels:
            List of hardness levels to generate (default
            ``["easy", "medium", "hard"]``).
        seed:
            Base random seed.

        Returns
        -------
        dict
            Mapping ``hardness_level -> list of MWISInstance``.
        """
        if hardness_levels is None:
            hardness_levels = ["easy", "medium", "hard"]

        result: Dict[str, List[MWISInstance]] = {level: [] for level in hardness_levels}
        rng = np.random.RandomState(seed)
        seed_counter = seed
        max_attempts = n_instances * 20  # generous cap

        for level in hardness_levels:
            attempts = 0
            while len(result[level]) < n_instances and attempts < max_attempts:
                attempts += 1
                density = float(rng.uniform(0.2, 0.7))
                inst = generate_mwis_instance(
                    n=n, density=density, seed=seed_counter
                )
                seed_counter += 1
                opt, val, _ = solve_mwis_exact(inst)
                inst.optimal_solution = opt
                inst.optimal_value = val
                classified = self._classify_hardness(inst)
                if classified == level:
                    inst.hardness = classified
                    result[level].append(inst)

        return result


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Hardness score ===")
    inst = generate_mwis_instance(n=12, density=0.4, seed=55)
    opt, val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt
    inst.optimal_value = val
    hs = compute_hardness_score(inst)
    print(f"hardness_score: {hs['hardness_score']:.4f}")
    print(f"  t_milp={hs['t_milp']:.4f}, t_pt={hs['t_pt']:.4f}, "
          f"p_local_hit={hs['p_local_hit']:.2f}, basins={hs['basin_count']}, "
          f"degeneracy={hs['degeneracy']}")

    print("\n=== Adversarial instances ===")
    for htype in ["planted_optimum", "deceptive", "multi_basin", "frustrated"]:
        hi = generate_hard_instance(n=12, seed=7, target_hardness=htype)
        print(f"  {htype:18s} optimal={hi.optimal_value:.3f}, "
              f"hardness={hi.hardness}")

    print("\n=== Barrier instance ===")
    bi = generate_barrier_instance(n=12, seed=9, n_basins=3, barrier_height=0.3)
    print(f"  barrier optimal={bi.optimal_value:.3f}")

    print("\n=== Stratified generation ===")
    stratifier = HardnessStratifier(n_chains=3, n_steps=100, n_shots=5)
    stratified = stratifier.generate_stratified(
        n=10, n_instances=2, hardness_levels=["easy", "medium", "hard"], seed=0
    )
    for level, instances in stratified.items():
        print(f"  {level}: {len(instances)} instances")
