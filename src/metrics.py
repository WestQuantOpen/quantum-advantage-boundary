"""Metrics computation for the quantum advantage boundary hunter experiment.

This module computes per-sampler metrics (pre/post recovery quality,
diversity, near-optimal coverage, hit probability) as well as cross-sampler
comparison metrics used to quantify a potential quantum advantage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.problems import MWISInstance, is_independent, mwis_value


def _bitstring_key(subset: np.ndarray) -> bytes:
    """Convert a binary array to a hashable bytes key."""
    return np.asarray(subset).astype(np.int8).tobytes()


def _normalized_entropy(samples: List[np.ndarray], n: int) -> float:
    """Normalized Shannon entropy over the empirical sample distribution.

    The entropy of the empirical distribution of distinct bitstrings is
    computed and then normalized by ``log2(min(K, 2^n))`` so that the result
    lies in ``[0, 1]``. A value of 1 means every sample is distinct (maximally
    diverse), 0 means all samples are identical.
    """
    if not samples or n == 0:
        return 0.0
    counts: Dict[bytes, int] = {}
    for s in samples:
        key = _bitstring_key(s)
        counts[key] = counts.get(key, 0) + 1
    total = len(samples)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    if len(probs) <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(min(total, 1 << n))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def compute_metrics(
    samples: List[np.ndarray],
    instance: MWISInstance,
    recovery_result: Optional[Dict[str, Any]] = None,
    epsilon: float = 0.05,
) -> Dict[str, Any]:
    """Compute the full metric suite for a set of samples.

    Parameters
    ----------
    samples:
        List of binary candidate bitstrings.
    instance:
        The MWIS instance the samples refer to.
    recovery_result:
        Optional output of :func:`recovery.recover`. When provided its
        ``post_recovery_best`` is used for the post-recovery metrics.
    epsilon:
        Relative tolerance for the near-optimal coverage metric
        ``r_epsilon`` (default 5%).

    Returns
    -------
    dict
        Keys: ``pre_recovery_best``, ``pre_recovery_mean``, ``n_unique``,
        ``n_shots``, ``diversity``, ``n_independent``, ``independence_rate``,
        ``r_epsilon``, ``p_hit``. When ``recovery_result`` is given, also
        ``post_recovery_best`` and ``post_recovery_gap``.
    """
    n = instance.n
    n_shots = len(samples)

    if n_shots == 0:
        base = {
            "pre_recovery_best": 0.0,
            "pre_recovery_mean": 0.0,
            "n_unique": 0,
            "n_shots": 0,
            "diversity": 0.0,
            "n_independent": 0,
            "independence_rate": 0.0,
            "r_epsilon": 0.0,
            "p_hit": 0.0,
        }
        if recovery_result is not None:
            base["post_recovery_best"] = float(recovery_result.get("post_recovery_best", 0.0))
            base["post_recovery_gap"] = float(recovery_result.get("optimality_gap", float("nan")))
        return base

    weights = instance.weights
    adjacency = instance.adjacency

    values = np.array([mwis_value(s, weights) for s in samples], dtype=float)
    pre_recovery_best = float(np.max(values))
    pre_recovery_mean = float(np.mean(values))

    # Unique samples.
    unique_keys = set(_bitstring_key(s) for s in samples)
    n_unique = len(unique_keys)

    # Diversity (normalized entropy).
    diversity = _normalized_entropy(samples, n)

    # Independence (feasibility) rate.
    independent_flags = [is_independent(s, adjacency) for s in samples]
    n_independent = int(sum(independent_flags))
    independence_rate = n_independent / n_shots

    # Near-optimal coverage within epsilon.
    optimal_value = instance.optimal_value
    if optimal_value is not None and optimal_value > 0:
        threshold = (1.0 - epsilon) * optimal_value
        near_optimal = int(np.sum(values >= threshold - 1e-9))
        r_epsilon = near_optimal / n_shots
        # Hit probability: fraction of samples that are feasible AND within
        # epsilon of the optimum.
        hit_mask = np.array(independent_flags, dtype=bool) & (values >= threshold - 1e-9)
        p_hit = float(np.sum(hit_mask) / n_shots)
    else:
        r_epsilon = float("nan")
        p_hit = float("nan")

    metrics: Dict[str, Any] = {
        "pre_recovery_best": pre_recovery_best,
        "pre_recovery_mean": pre_recovery_mean,
        "n_unique": n_unique,
        "n_shots": n_shots,
        "diversity": diversity,
        "n_independent": n_independent,
        "independence_rate": independence_rate,
        "r_epsilon": r_epsilon,
        "p_hit": p_hit,
    }

    if recovery_result is not None:
        metrics["post_recovery_best"] = float(recovery_result.get("post_recovery_best", 0.0))
        metrics["post_recovery_gap"] = float(recovery_result.get("optimality_gap", float("nan")))

    return metrics


def quantum_advantage_metric(
    quantum_metrics: Dict[str, Any],
    classical_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute quantum-vs-classical advantage metrics.

    The core quantity is ``A_Q``, the difference in post-recovery best value
    between the quantum and classical pipelines:

        A_Q = post_recovery_quantum - post_recovery_classical

    Positive ``A_Q`` favours the quantum pipeline. We also report the
    difference in near-optimal coverage (``delta_r_epsilon``), hit probability
    (``delta_p_hit``), diversity (``delta_diversity``), and the gap
    improvement (``delta_gap`` = classical_gap - quantum_gap, positive favours
    quantum).
    """
    post_q = float(quantum_metrics.get("post_recovery_best", quantum_metrics.get("pre_recovery_best", 0.0)))
    post_c = float(classical_metrics.get("post_recovery_best", classical_metrics.get("pre_recovery_best", 0.0)))
    a_q = post_q - post_c

    def _safe_diff(a: Any, b: Any) -> float:
        try:
            fa = float(a)
            fb = float(b)
            if math.isnan(fa) or math.isnan(fb):
                return float("nan")
            return fa - fb
        except (TypeError, ValueError):
            return float("nan")

    delta_r_epsilon = _safe_diff(
        quantum_metrics.get("r_epsilon"), classical_metrics.get("r_epsilon")
    )
    delta_p_hit = _safe_diff(
        quantum_metrics.get("p_hit"), classical_metrics.get("p_hit")
    )
    delta_diversity = _safe_diff(
        quantum_metrics.get("diversity"), classical_metrics.get("diversity")
    )
    delta_gap = _safe_diff(
        classical_metrics.get("post_recovery_gap"),
        quantum_metrics.get("post_recovery_gap"),
    )

    return {
        "A_Q": a_q,
        "post_recovery_quantum": post_q,
        "post_recovery_classical": post_c,
        "delta_r_epsilon": delta_r_epsilon,
        "delta_p_hit": delta_p_hit,
        "delta_diversity": delta_diversity,
        "delta_gap": delta_gap,
    }


def subspace_utility(metrics: Dict[str, Any], recovery_budget: int) -> float:
    """Compute the subspace utility: quality per unit of recovery budget.

    Defined as ``post_recovery_best / max(recovery_budget, 1)``. When no
    post-recovery value is available, falls back to ``pre_recovery_best``.
    """
    if recovery_budget <= 0:
        return 0.0
    quality = float(
        metrics.get("post_recovery_best", metrics.get("pre_recovery_best", 0.0))
    )
    return quality / float(recovery_budget)


if __name__ == "__main__":
    from problems import generate_mwis_instance
    from classical_samplers import UniformRandomSampler, GreedySampler
    from exact_solver import solve_mwis_exact
    from recovery import recover

    inst = generate_mwis_instance(n=12, density=0.4, seed=31)
    opt_subset, opt_val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt_subset
    inst.optimal_value = opt_val
    print(f"instance: {inst.instance_id}, optimal: {opt_val:.3f}")

    for sampler in [UniformRandomSampler(), GreedySampler()]:
        samples, _ = sampler.sample(inst, n_shots=100, seed=5)
        rr = recover(samples, inst, max_expansion=2, max_candidates=5000)
        m = compute_metrics(samples, inst, recovery_result=rr)
        util = subspace_utility(m, rr["n_candidates"])
        print(
            f"{sampler.name:18s} | pre_best={m['pre_recovery_best']:7.3f} | "
            f"post_best={m['post_recovery_best']:7.3f} | "
            f"diversity={m['diversity']:.3f} | indep_rate={m['independence_rate']:.2f} | "
            f"r_eps={m['r_epsilon']:.3f} | p_hit={m['p_hit']:.3f} | util={util:.5f}"
        )

    # Quantum vs classical advantage demo.
    q_samples, _ = GreedySampler().sample(inst, n_shots=100, seed=10)
    c_samples, _ = UniformRandomSampler().sample(inst, n_shots=100, seed=20)
    q_rr = recover(q_samples, inst, max_expansion=2, max_candidates=5000)
    c_rr = recover(c_samples, inst, max_expansion=2, max_candidates=5000)
    qm = compute_metrics(q_samples, inst, recovery_result=q_rr)
    cm = compute_metrics(c_samples, inst, recovery_result=c_rr)
    adv = quantum_advantage_metric(qm, cm)
    print(f"advantage metrics: {adv}")
