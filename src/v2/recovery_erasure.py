"""Recovery Erasure Experiment.

For the same saved quantum and classical samples, run downstream recovery
with budgets B_C = 0, 100, 500, 2000, 10000, 50000.

Also measure pre-recovery diagnostics:
- d_H(S, X*) = min Hamming distance from samples to optimum
- best raw objective
- low-energy tail mass
- P(d_H <= k) for k=1,2,3
- number of distinct attraction basins
- entropy/diversity

Three possible diagnoses:
A. Quantum is worse raw -> no amount of N scaling helps. Change quantum kernel.
B. Quantum is better raw but recovery eliminates difference -> subspace-info
   advantage hidden by strong recovery.
C. Difference appears only at bounded recovery budget -> QCSC compute-to-target
   result.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.problems import MWISInstance, edges_from_adjacency, is_independent, mwis_value
from src.recovery import recover


# ---------------------------------------------------------------------------
# Pre-recovery diagnostics
# ---------------------------------------------------------------------------

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two binary arrays.

    Parameters
    ----------
    a, b:
        Binary arrays of the same length.

    Returns
    -------
    int
        Number of positions at which ``a`` and ``b`` differ.
    """
    a_arr = np.asarray(a).astype(int)
    b_arr = np.asarray(b).astype(int)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"arrays must have the same shape, got {a_arr.shape} and {b_arr.shape}"
        )
    return int(np.sum(a_arr != b_arr))


def min_hamming_to_optimum(samples: List[np.ndarray], optimal: np.ndarray) -> int:
    """Minimum Hamming distance from any sample to the optimal solution.

    Parameters
    ----------
    samples:
        List of binary candidate bitstrings.
    optimal:
        Binary array describing the optimal solution.

    Returns
    -------
    int
        The smallest Hamming distance between any sample and ``optimal``.
        Returns ``-1`` if ``samples`` is empty.
    """
    if not samples:
        return -1
    opt = np.asarray(optimal).astype(int)
    distances = [hamming_distance(np.asarray(s).astype(int), opt) for s in samples]
    return int(min(distances))


def p_within_hamming(
    samples: List[np.ndarray], optimal: np.ndarray, k: int
) -> float:
    """Fraction of samples within Hamming distance ``k`` of the optimum.

    Parameters
    ----------
    samples:
        List of binary candidate bitstrings.
    optimal:
        Binary array describing the optimal solution.
    k:
        Maximum Hamming distance (inclusive).

    Returns
    -------
    float
        Fraction of samples with ``hamming_distance(sample, optimal) <= k``.
    """
    if not samples:
        return 0.0
    opt = np.asarray(optimal).astype(int)
    count = sum(
        1 for s in samples if hamming_distance(np.asarray(s).astype(int), opt) <= k
    )
    return count / len(samples)


def _one_flip_descent(
    subset: np.ndarray, instance: MWISInstance
) -> np.ndarray:
    """Run 1-flip descent (greedy hill climb) until a local optimum is reached.

    At each step the single flip that yields the largest objective improvement
    while maintaining feasibility is applied.  When no improving flip exists the
    current state is a local optimum and is returned.
    """
    n = instance.n
    adjacency = instance.adjacency
    weights = instance.weights
    current = np.asarray(subset).astype(int).copy()
    current_val = mwis_value(current, weights)

    improved = True
    while improved:
        improved = False
        best_delta = 1e-12
        best_v = -1
        for v in range(n):
            candidate = current.copy()
            candidate[v] = 1 - candidate[v]
            if not is_independent(candidate, adjacency):
                continue
            cand_val = mwis_value(candidate, weights)
            delta = cand_val - current_val
            if delta > best_delta:
                best_delta = delta
                best_v = v
        if best_v >= 0:
            current[best_v] = 1 - current[best_v]
            current_val += best_delta
            improved = True
    return current


def attraction_basins(samples: List[np.ndarray], instance: MWISInstance) -> int:
    """Cluster samples by which local optimum they converge to under 1-flip descent.

    Each sample is run through greedy 1-flip descent.  Samples that converge to
    the same local optimum belong to the same attraction basin.

    Parameters
    ----------
    samples:
        List of binary candidate bitstrings.
    instance:
        The MWIS instance.

    Returns
    -------
    int
        Number of distinct attraction basins (distinct local optima reached).
    """
    if not samples:
        return 0
    local_optima: Dict[bytes, int] = {}
    for s in samples:
        lo = _one_flip_descent(s, instance)
        key = lo.astype(np.int8).tobytes()
        local_optima[key] = local_optima.get(key, 0) + 1
    return len(local_optima)


def low_energy_tail_mass(
    samples: List[np.ndarray],
    instance: MWISInstance,
    threshold_fraction: float = 0.95,
) -> float:
    """Fraction of samples within ``threshold_fraction`` of the optimal value.

    Parameters
    ----------
    samples:
        List of binary candidate bitstrings.
    instance:
        The MWIS instance.  ``instance.optimal_value`` must be set.
    threshold_fraction:
        Fraction of the optimal value that defines the low-energy tail
        (default 0.95, i.e. within 5% of optimal).

    Returns
    -------
    float
        Fraction of samples whose objective value is at least
        ``threshold_fraction * optimal_value``.  Returns ``nan`` if the optimum
        is unknown.
    """
    if not samples:
        return 0.0
    if instance.optimal_value is None or instance.optimal_value <= 0:
        return float("nan")
    threshold = threshold_fraction * instance.optimal_value
    values = np.array([mwis_value(s, instance.weights) for s in samples], dtype=float)
    return float(np.mean(values >= threshold - 1e-9))


def _normalized_entropy(samples: List[np.ndarray], n: int) -> float:
    """Normalized Shannon entropy over the empirical sample distribution."""
    if not samples or n == 0:
        return 0.0
    counts: Dict[bytes, int] = {}
    for s in samples:
        key = np.asarray(s).astype(np.int8).tobytes()
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


def pre_recovery_diagnostics(
    samples: List[np.ndarray], instance: MWISInstance
) -> Dict[str, Any]:
    """Compute the full set of pre-recovery diagnostics for a sample set.

    Returns a dict with keys: ``best_raw_objective``, ``mean_raw_objective``,
    ``min_hamming_to_optimum``, ``p_within_hamming_1``, ``p_within_hamming_2``,
    ``p_within_hamming_3``, ``n_attraction_basins``, ``low_energy_tail_mass``,
    ``diversity``, ``n_unique``.
    """
    n = instance.n
    if not samples:
        return {
            "best_raw_objective": 0.0,
            "mean_raw_objective": 0.0,
            "min_hamming_to_optimum": -1,
            "p_within_hamming_1": 0.0,
            "p_within_hamming_2": 0.0,
            "p_within_hamming_3": 0.0,
            "n_attraction_basins": 0,
            "low_energy_tail_mass": float("nan"),
            "diversity": 0.0,
            "n_unique": 0,
        }

    values = np.array([mwis_value(s, instance.weights) for s in samples], dtype=float)
    best_raw = float(np.max(values))
    mean_raw = float(np.mean(values))

    unique_keys = set(np.asarray(s).astype(np.int8).tobytes() for s in samples)
    n_unique = len(unique_keys)
    diversity = _normalized_entropy(samples, n)

    has_opt = instance.optimal_solution is not None
    if has_opt:
        opt = np.asarray(instance.optimal_solution).astype(int)
        d_min = min_hamming_to_optimum(samples, opt)
        p1 = p_within_hamming(samples, opt, 1)
        p2 = p_within_hamming(samples, opt, 2)
        p3 = p_within_hamming(samples, opt, 3)
    else:
        d_min = -1
        p1 = p2 = p3 = float("nan")

    tail = low_energy_tail_mass(samples, instance, 0.95)
    basins = attraction_basins(samples, instance)

    return {
        "best_raw_objective": best_raw,
        "mean_raw_objective": mean_raw,
        "min_hamming_to_optimum": d_min,
        "p_within_hamming_1": p1,
        "p_within_hamming_2": p2,
        "p_within_hamming_3": p3,
        "n_attraction_basins": basins,
        "low_energy_tail_mass": tail,
        "diversity": diversity,
        "n_unique": n_unique,
    }


# ---------------------------------------------------------------------------
# Recovery erasure
# ---------------------------------------------------------------------------

def recovery_erasure_experiment(
    samples: List[np.ndarray],
    instance: MWISInstance,
    budgets: Sequence[int] = (0, 100, 500, 2000, 10000, 50000),
) -> Dict[int, Dict[str, Any]]:
    """Run recovery with each budget and return a dict mapping budget -> result.

    For ``budget=0`` no recovery is performed; the result contains the raw best
    sample (repaired only for feasibility evaluation).

    Parameters
    ----------
    samples:
        List of binary candidate bitstrings.
    instance:
        The MWIS instance.
    budgets:
        Sequence of recovery budgets (``max_candidates``) to sweep.

    Returns
    -------
    dict
        Mapping ``budget -> recovery_result`` where each recovery result is the
        dict returned by :func:`src.recovery.recover` (or a raw-best dict for
        budget 0).
    """
    results: Dict[int, Dict[str, Any]] = {}
    for budget in budgets:
        budget_int = int(budget)
        if budget_int == 0:
            # No recovery: just evaluate raw best.
            if not samples:
                results[budget_int] = {
                    "pre_recovery_best": 0.0,
                    "post_recovery_best": 0.0,
                    "optimality_gap": float("nan") if instance.optimal_value is None
                    else 1.0 - 0.0 / instance.optimal_value,
                    "n_candidates": 0,
                    "n_repaired": 0,
                    "recovery_time": 0.0,
                    "best_subset": np.zeros(instance.n, dtype=int),
                }
                continue
            values = [mwis_value(s, instance.weights) for s in samples]
            best_idx = int(np.argmax(values))
            best_subset = np.asarray(samples[best_idx]).astype(int)
            best_val = float(values[best_idx])
            if instance.optimal_value is not None and instance.optimal_value > 0:
                gap = 1.0 - best_val / instance.optimal_value
            else:
                gap = float("nan")
            results[budget_int] = {
                "pre_recovery_best": best_val,
                "post_recovery_best": best_val,
                "optimality_gap": float(gap),
                "n_candidates": 0,
                "n_repaired": 0,
                "recovery_time": 0.0,
                "best_subset": best_subset,
            }
        else:
            results[budget_int] = recover(
                samples, instance, max_expansion=2, max_candidates=budget_int
            )
    return results


def diagnose(
    quantum_samples: List[np.ndarray],
    classical_samples: List[np.ndarray],
    instance: MWISInstance,
    budgets: Sequence[int] = (0, 100, 500, 2000, 10000, 50000),
) -> Dict[str, Any]:
    """Run the erasure experiment on both sample sets and classify the diagnosis.

    Computes ``U_Q(B_C) - U_C(B_C)`` (post-recovery best quantum minus
    post-recovery best classical) for each budget, then classifies into one of
    three diagnoses:

    * **A** -- Quantum is worse *raw* (at B_C=0).  No amount of N scaling helps;
      the quantum kernel itself is inferior.
    * **B** -- Quantum is better raw (B_C=0) but recovery eliminates the
      difference (advantage -> 0 or negative at large B_C).  The subspace-info
      advantage is hidden by strong recovery.
    * **C** -- The difference appears (or is largest) only at a bounded recovery
      budget.  This is a QCSC compute-to-target result.

    Parameters
    ----------
    quantum_samples, classical_samples:
        Saved sample sets from the quantum and classical pipelines.
    instance:
        The MWIS instance.
    budgets:
        Recovery budgets to sweep.

    Returns
    -------
    dict
        Keys: ``quantum_diagnostics``, ``classical_diagnostics``,
        ``quantum_erasure``, ``classical_erasure``, ``advantage_curve``
        (dict budget -> U_Q - U_C), ``diagnosis`` (``"A"``, ``"B"``, or ``"C"``),
        ``diagnosis_detail``.
    """
    q_diag = pre_recovery_diagnostics(quantum_samples, instance)
    c_diag = pre_recovery_diagnostics(classical_samples, instance)

    q_erasure = recovery_erasure_experiment(quantum_samples, instance, budgets)
    c_erasure = recovery_erasure_experiment(classical_samples, instance, budgets)

    advantage_curve: Dict[int, float] = {}
    for budget in budgets:
        budget_int = int(budget)
        u_q = float(q_erasure[budget_int].get("post_recovery_best", 0.0))
        u_c = float(c_erasure[budget_int].get("post_recovery_best", 0.0))
        advantage_curve[budget_int] = u_q - u_c

    # Classify the diagnosis.
    raw_advantage = advantage_curve.get(0, 0.0)
    max_budget = max(budgets) if budgets else 0
    final_advantage = advantage_curve.get(int(max_budget), 0.0)

    # Find the budget at which the advantage is maximised.
    best_budget = 0
    best_adv = -float("inf")
    for b, adv in advantage_curve.items():
        if adv > best_adv:
            best_adv = adv
            best_budget = b

    if raw_advantage <= 0:
        diagnosis = "A"
        detail = (
            "Quantum is worse or equal raw (B_C=0 advantage <= 0). "
            "No amount of N scaling helps; change the quantum kernel."
        )
    elif raw_advantage > 0 and final_advantage <= 1e-9:
        diagnosis = "B"
        detail = (
            "Quantum is better raw but recovery eliminates the difference. "
            "Subspace-info advantage is hidden by strong recovery."
        )
    else:
        # Advantage persists; check whether it peaks at an intermediate budget.
        if best_budget != 0 and best_budget != int(max_budget):
            diagnosis = "C"
            detail = (
                f"Difference appears/largest at bounded budget B_C={best_budget}. "
                "QCSC compute-to-target result."
            )
        elif best_budget == 0:
            diagnosis = "B"
            detail = (
                "Quantum advantage is largest at B_C=0 and shrinks with recovery. "
                "Subspace-info advantage is hidden by strong recovery."
            )
        else:
            diagnosis = "C"
            detail = (
                "Quantum advantage grows or persists with recovery budget. "
                "QCSC compute-to-target result."
            )

    return {
        "quantum_diagnostics": q_diag,
        "classical_diagnostics": c_diag,
        "quantum_erasure": q_erasure,
        "classical_erasure": c_erasure,
        "advantage_curve": advantage_curve,
        "diagnosis": diagnosis,
        "diagnosis_detail": detail,
    }


def plot_erasure_curve(
    quantum_results: Dict[int, Dict[str, Any]],
    classical_results: Dict[int, Dict[str, Any]],
    budgets: Sequence[int],
    output_path: str,
) -> None:
    """Plot ``U_Q(B_C) - U_C(B_C)`` versus ``B_C``.

    Parameters
    ----------
    quantum_results:
        Dict mapping budget -> recovery result for the quantum pipeline.
    classical_results:
        Dict mapping budget -> recovery result for the classical pipeline.
    budgets:
        The budgets to plot (x-axis).
    output_path:
        File path to save the figure (e.g. ``"erasure_curve.png"``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets_list = [int(b) for b in budgets]
    diffs = []
    for b in budgets_list:
        u_q = float(quantum_results.get(b, {}).get("post_recovery_best", 0.0))
        u_c = float(classical_results.get(b, {}).get("post_recovery_best", 0.0))
        diffs.append(u_q - u_c)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets_list, diffs, "o-", color="tab:blue", linewidth=2, markersize=6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("symlog" if min(budgets_list) == 0 else "log")
    ax.set_xlabel(r"Recovery budget $B_C$", fontsize=12)
    ax.set_ylabel(r"$U_Q(B_C) - U_C(B_C)$", fontsize=12)
    ax.set_title("Recovery Erasure Curve", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Annotate each point.
    for b, d in zip(budgets_list, diffs):
        ax.annotate(f"{d:.2f}", (b, d), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.problems import generate_mwis_instance
    from src.exact_solver import solve_mwis_exact
    from src.classical_samplers import GreedySampler, UniformRandomSampler

    inst = generate_mwis_instance(n=14, density=0.4, seed=55)
    opt_subset, opt_val, _ = solve_mwis_exact(inst)
    inst.optimal_solution = opt_subset
    inst.optimal_value = opt_val
    print(f"instance: {inst.instance_id}, optimal: {opt_val:.3f}")

    q_samples, _ = GreedySampler(n_restarts=5).sample(inst, n_shots=50, seed=10)
    c_samples, _ = UniformRandomSampler().sample(inst, n_shots=50, seed=20)

    budgets = [0, 100, 500, 2000, 10000, 50000]
    result = diagnose(q_samples, c_samples, inst, budgets)
    print(f"diagnosis: {result['diagnosis']}")
    print(f"detail: {result['diagnosis_detail']}")
    print("advantage curve:")
    for b, adv in result["advantage_curve"].items():
        print(f"  B_C={b:>6d}  U_Q-U_C = {adv:.4f}")

    plot_erasure_curve(
        result["quantum_erasure"], result["classical_erasure"], budgets,
        "/tmp/quantum-advantage-boundary/state/erasure_curve.png",
    )
    print("plot saved.")
