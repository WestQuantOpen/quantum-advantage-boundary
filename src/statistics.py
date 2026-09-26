"""Statistical analysis utilities for the quantum-advantage boundary hunter.

All functions are pure, fully type-hinted, and depend only on numpy / scipy.
They are designed to be composable: the output of :func:`paired_comparison`
can be fed directly into :func:`classify_advantage`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "paired_comparison",
    "bootstrap_ci",
    "effect_size_cohens_d",
    "success_probability",
    "multiple_comparison_correction",
    "classify_advantage",
]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(
    data: Sequence[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence interval for the mean of ``data``.

    Parameters
    ----------
    data:
        1-D sequence of observations.
    n_bootstrap:
        Number of bootstrap resamples.
    confidence:
        Two-sided confidence level (e.g. 0.95 -> [2.5%, 97.5%]).

    Returns
    -------
    dict
        ``mean``, ``ci_lower``, ``ci_upper``, ``std_error``.
    """
    arr = np.asarray(data, dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "std_error": float("nan")}

    rng = np.random.default_rng()
    # Resample indices with replacement and compute means vectorised.
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2.0))
    upper = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))

    return {
        "mean": float(arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "std_error": float(arr.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------
def effect_size_cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Cohen's *d* effect size between two independent groups.

    Uses the pooled standard deviation:
        s_pooled = sqrt(((n1-1) s1^2 + (n2-1) s2^2) / (n1 + n2 - 2))

    Returns 0.0 when the pooled variance is zero or a group is empty.
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return 0.0

    mean_diff = float(a.mean() - b.mean())
    var1 = float(a.var(ddof=1)) if n1 > 1 else 0.0
    var2 = float(b.var(ddof=1)) if n2 > 1 else 0.0

    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / denom
    if pooled_var <= 0:
        return 0.0
    return mean_diff / math.sqrt(pooled_var)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------
def paired_comparison(
    quantum_values: Sequence[float],
    classical_values: Sequence[float],
    n_bootstrap: int = 10000,
) -> dict[str, float]:
    """Paired bootstrap comparison between quantum and classical measurements.

    Parameters
    ----------
    quantum_values, classical_values:
        Paired observations (same length) of a metric for the two methods.
    n_bootstrap:
        Number of bootstrap resamples for the confidence interval.

    Returns
    -------
    dict
        ``mean_diff``, ``median_diff``, ``ci_lower``, ``ci_upper``,
        ``p_value``, ``effect_size`` (paired Cohen's d).
    """
    q = np.asarray(quantum_values, dtype=float)
    c = np.asarray(classical_values, dtype=float)
    if q.size != c.size:
        raise ValueError("quantum_values and classical_values must have equal length")
    n = q.size
    if n == 0:
        return {
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "effect_size": float("nan"),
        }

    diffs = q - c
    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))

    # Bootstrap CI on the mean difference (paired resampling).
    rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diffs[idx].mean(axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # p-value via paired t-test (falls back to NaN if degenerate).
    if n >= 2 and diffs.std(ddof=1) > 0:
        t_stat, p_value = stats.ttest_rel(q, c)
        p_value = float(p_value)
    else:
        p_value = 1.0 if mean_diff == 0 else 0.0

    # Paired Cohen's d: mean_diff / std(diffs)
    std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
    effect_size = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "effect_size": effect_size,
    }


# ---------------------------------------------------------------------------
# Success probability
# ---------------------------------------------------------------------------
def success_probability(values: Sequence[float], target: float) -> float:
    """Fraction of ``values`` that meet or exceed ``target``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= target))


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------
def multiple_comparison_correction(
    p_values: Sequence[float],
    method: str = "bonferroni",
) -> list[float]:
    """Adjust a family of p-values for multiple comparisons.

    Parameters
    ----------
    p_values:
        Raw p-values.
    method:
        ``"bonferroni"`` or ``"bh"`` (Benjamini-Hochberg FDR).

    Returns
    -------
    list[float]
        Adjusted p-values, in the original order.
    """
    arr = np.asarray(p_values, dtype=float)
    m = arr.size
    if m == 0:
        return []

    method = method.lower()
    if method == "bonferroni":
        return [min(float(p) * m, 1.0) for p in arr]

    if method in ("bh", "benjamini-hochberg", "fdr_bh"):
        order = np.argsort(arr)
        ranked = arr[order]
        adjusted = np.empty(m, dtype=float)
        # Walk from largest p-value down, enforcing monotonicity.
        prev: float = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            val = min(ranked[i] * m / rank, prev)
            adjusted[i] = min(val, 1.0)
            prev = adjusted[i]
        # Un-sort back to original order.
        result = np.empty(m, dtype=float)
        result[order] = adjusted
        return [float(v) for v in result]

    raise ValueError(f"Unknown correction method: {method!r}")


# ---------------------------------------------------------------------------
# Advantage classification
# ---------------------------------------------------------------------------
def classify_advantage(
    qa_results: dict[str, float],
    threshold: float = 0.0,
    ci_exclude_zero: bool = True,
) -> dict[str, Any]:
    """Classify the strength of quantum-advantage evidence.

    Parameters
    ----------
    qa_results:
        Dictionary produced by :func:`paired_comparison` (or compatible) with
        keys ``mean_diff``, ``ci_lower``, ``ci_upper``, ``p_value``,
        ``effect_size``.  Optionally also ``success_probability`` and
        ``n_samples``.
    threshold:
        Minimum ``mean_diff`` considered an advantage (default 0.0).
    ci_exclude_zero:
        When True, a CI that straddles zero downgrades the level.

    Returns
    -------
    dict
        ``level`` (int | str), ``label`` (str), ``reason`` (str), plus the
        derived boolean flags used for the decision.
    """
    from typing import Any  # local import keeps module surface clean

    mean_diff = float(qa_results.get("mean_diff", float("nan")))
    ci_lower = float(qa_results.get("ci_lower", float("nan")))
    ci_upper = float(qa_results.get("ci_upper", float("nan")))
    p_value = float(qa_results.get("p_value", float("nan")))
    effect_size = float(qa_results.get("effect_size", 0.0))
    success_prob = float(qa_results.get("success_probability", 0.0))
    n_samples = int(qa_results.get("n_samples", 0))

    flags: dict[str, Any] = {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "effect_size": effect_size,
        "success_probability": success_prob,
        "n_samples": n_samples,
    }

    # --- Inconclusive: not enough data -------------------------------
    if n_samples < 5 or math.isnan(mean_diff):
        flags.update(level="X", label="inconclusive", reason="insufficient samples")
        return flags

    ci_excludes_zero = (ci_lower > 0) or (not ci_exclude_zero)
    significant = (not math.isnan(p_value)) and (p_value < 0.05)
    meets_threshold = mean_diff >= threshold
    abs_d = abs(effect_size)

    flags["ci_excludes_zero"] = ci_excludes_zero
    flags["significant"] = significant
    flags["meets_threshold"] = meets_threshold

    # --- Level 0: no evidence ---------------------------------------
    if not significant or not ci_excludes_zero or not meets_threshold:
        flags.update(
            level=0,
            label="no_advantage",
            reason="not significant or CI includes zero",
        )
        return flags

    # --- Strong evidence (Level 3) ----------------------------------
    if (
        p_value < 0.01
        and abs_d >= 0.8
        and ci_excludes_zero
        and success_prob >= 0.8
    ):
        flags.update(
            level=3,
            label="strong_advantage",
            reason="p<0.01, |d|>=0.8, CI excludes zero, P(success)>=0.8",
        )
        return flags

    # --- Moderate evidence (Level 2) --------------------------------
    if abs_d >= 0.5 and success_prob >= 0.5:
        flags.update(
            level=2,
            label="moderate_advantage",
            reason="|d|>=0.5, P(success)>=0.5, CI excludes zero",
        )
        return flags

    # --- Weak evidence (Level 1) ------------------------------------
    flags.update(
        level=1,
        label="weak_advantage",
        reason="significant but small effect or low success probability",
    )
    return flags
