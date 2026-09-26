"""Quantum Advantage Phase Diagram synthesis.

    A = A(N, H, B_C, B_Q, R, K_Q, chi)

Where:
  N = problem size
  H = empirical hardness
  B_C = classical recovery budget
  B_Q = quantum sampling budget
  R = representation
  K_Q = quantum kernel (QAOA/QeMCMC/etc.)
  chi = classical MPS simulability capacity

The negative results to N=28 become the first data points in the A<0 region.
The AI agent uses these to actively search for the phase boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

__all__ = ["PhaseDiagramPoint", "PhaseDiagram"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class PhaseDiagramPoint:
    """A single data point in the quantum advantage phase diagram.

    Attributes
    ----------
    n:
        Problem size (number of qubits / variables).
    hardness:
        Empirical hardness label or score (e.g. "easy", "hard", or a float).
    b_c:
        Classical recovery budget.
    b_q:
        Quantum sampling budget (shots).
    representation:
        Quantum representation identifier (e.g. "qaoa_p2").
    kernel:
        Quantum kernel name (e.g. "QAOA", "QeMCMC").
    chi:
        Classical MPS simulability capacity (bond dimension).
    a_q:
        Quantum advantage metric (positive favours quantum).
    diagnosis:
        Free-form diagnosis string.
    """

    n: int
    hardness: Any  # str or float
    b_c: float
    b_q: float
    representation: str
    kernel: str
    chi: int
    a_q: float
    diagnosis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase diagram
# ---------------------------------------------------------------------------
class PhaseDiagram:
    """Quantum advantage phase diagram.

    Collects :class:`PhaseDiagramPoint` objects and provides methods to
    identify the phase boundary, suggest the next experiment, classify
    regions, and generate plots.
    """

    def __init__(self) -> None:
        self.points: List[PhaseDiagramPoint] = []

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------
    def add_point(self, point: PhaseDiagramPoint) -> None:
        """Add a data point to the phase diagram."""
        self.points.append(point)
        logger.info(
            "phase_diagram.point_added",
            n=point.n,
            a_q=point.a_q,
            n_points=len(self.points),
        )

    def add_from_results(self, results: Dict[str, Any]) -> None:
        """Add a point from an experiment results dict.

        The results dict should contain keys that map to the
        :class:`PhaseDiagramPoint` fields.  Missing fields are filled with
        sensible defaults.
        """
        point = PhaseDiagramPoint(
            n=int(results.get("n", 0)),
            hardness=results.get("hardness", "unknown"),
            b_c=float(results.get("b_c", results.get("recovery_budget", 5000))),
            b_q=float(results.get("b_q", results.get("n_shots", 256))),
            representation=str(results.get("representation", results.get("sampler", "unknown"))),
            kernel=str(results.get("kernel", "QAOA")),
            chi=int(results.get("chi", 0)),
            a_q=float(results.get("a_q", results.get("quantum_advantage", {}).get("A_Q", 0.0))),
            diagnosis=str(results.get("diagnosis", results.get("diagnosis", ""))),
        )
        self.add_point(point)

    # ------------------------------------------------------------------
    # Phase boundary
    # ------------------------------------------------------------------
    def find_boundary(self) -> Dict[str, Any]:
        """Identify the phase boundary (where A_Q crosses 0).

        Groups points by problem size ``n`` and finds the size where the
        mean A_Q transitions from positive to negative.

        Returns a dict with:
        - ``boundary_n``: the n where the crossover occurs (or a range).
        - ``per_n``: mean A_Q for each n.
        - ``boundary_points``: points near the boundary.
        """
        if not self.points:
            return {
                "boundary_n": None,
                "per_n": {},
                "boundary_points": [],
            }

        # Group by n.
        by_n: Dict[int, List[float]] = {}
        for p in self.points:
            by_n.setdefault(p.n, []).append(p.a_q)

        per_n: Dict[int, float] = {n: float(np.mean(aqs)) for n, aqs in by_n.items()}

        sorted_ns = sorted(per_n.keys())

        # Find the crossover: the n where mean A_Q goes from positive to
        # negative (or vice versa).
        boundary_n: int | Tuple[int, int] | None = None
        for i in range(len(sorted_ns) - 1):
            n1, n2 = sorted_ns[i], sorted_ns[i + 1]
            a1, a2 = per_n[n1], per_n[n2]
            if (a1 > 0 and a2 <= 0) or (a1 >= 0 and a2 < 0):
                boundary_n = (n1, n2)
                break

        if boundary_n is None:
            # No crossover found: all positive or all negative.
            if all(a > 0 for a in per_n.values()):
                boundary_n = sorted_ns[-1]  # boundary is beyond largest n
            elif all(a <= 0 for a in per_n.values()):
                boundary_n = sorted_ns[0]  # boundary is below smallest n
            else:
                boundary_n = sorted_ns[len(sorted_ns) // 2]

        # Points near the boundary.
        boundary_points: List[PhaseDiagramPoint] = []
        if isinstance(boundary_n, tuple):
            n1, n2 = boundary_n
            boundary_points = [p for p in self.points if p.n in (n1, n2)]
        elif isinstance(boundary_n, int):
            boundary_points = [p for p in self.points if p.n == boundary_n]

        return {
            "boundary_n": boundary_n,
            "per_n": per_n,
            "boundary_points": [p.to_dict() for p in boundary_points],
        }

    # ------------------------------------------------------------------
    # Suggest next experiment
    # ------------------------------------------------------------------
    def suggest_next_experiment(
        self,
        current_points: Optional[List[PhaseDiagramPoint]] = None,
        budget: int = 1,
    ) -> List[Dict[str, Any]]:
        """Suggest the highest information-gain experiment(s).

        Strategy: identify the current boundary estimate and suggest
        experiments that would most reduce uncertainty about the boundary
        location.  Uses a simple binary-search-like heuristic: test the
        midpoint between the largest n with A_Q > 0 and the smallest n
        with A_Q <= 0.

        Returns a list of ``budget`` experiment suggestions.
        """
        points = current_points if current_points is not None else self.points

        if not points:
            # No data: suggest a starting point.
            return [
                {
                    "n": 20,
                    "density": 0.5,
                    "reason": "No data yet; start at n=20",
                    "expected_information_gain": 1.0,
                }
            ]

        # Group by n and compute mean A_Q.
        by_n: Dict[int, List[float]] = {}
        for p in points:
            by_n.setdefault(p.n, []).append(p.a_q)
        per_n: Dict[int, float] = {n: float(np.mean(aqs)) for n, aqs in by_n.items()}
        sorted_ns = sorted(per_n.keys())

        # Find the boundary region.
        positive_ns = [n for n in sorted_ns if per_n[n] > 0]
        negative_ns = [n for n in sorted_ns if per_n[n] <= 0]

        suggestions: List[Dict[str, Any]] = []

        if positive_ns and negative_ns:
            # Binary search: test between the largest positive and smallest negative.
            n_pos = max(positive_ns)
            n_neg = min(negative_ns)
            if n_pos < n_neg:
                mid_n = (n_pos + n_neg) // 2
                suggestions.append({
                    "n": mid_n,
                    "density": 0.5,
                    "b_c": 5000,
                    "b_q": 256,
                    "reason": f"Binary search: test n={mid_n} between "
                              f"n={n_pos} (A_Q>0) and n={n_neg} (A_Q<=0)",
                    "expected_information_gain": 0.9,
                })
            elif n_pos > n_neg:
                # Boundary is between n_neg and n_pos (inverted).
                mid_n = (n_pos + n_neg) // 2
                suggestions.append({
                    "n": mid_n,
                    "density": 0.5,
                    "b_c": 5000,
                    "b_q": 256,
                    "reason": f"Test n={mid_n} between conflicting results",
                    "expected_information_gain": 0.8,
                })
            else:
                # Same n has both positive and negative — explore density.
                suggestions.append({
                    "n": n_pos,
                    "density": 0.5,
                    "b_c": 10000,
                    "b_q": 512,
                    "reason": f"n={n_pos} has mixed results; increase budget",
                    "expected_information_gain": 0.7,
                })
        elif positive_ns and not negative_ns:
            # All positive: push to larger n.
            max_n = max(positive_ns)
            suggestions.append({
                "n": max_n + 2,
                "density": 0.5,
                "b_c": 5000,
                "b_q": 256,
                "reason": f"All results positive up to n={max_n}; "
                          f"push to n={max_n + 2}",
                "expected_information_gain": 0.85,
            })
        elif negative_ns and not positive_ns:
            # All negative: try smaller n or different parameters.
            min_n = min(negative_ns)
            if min_n > 10:
                suggestions.append({
                    "n": min_n - 2,
                    "density": 0.5,
                    "b_c": 5000,
                    "b_q": 256,
                    "reason": f"All results negative from n={min_n}; "
                              f"try smaller n={min_n - 2}",
                    "expected_information_gain": 0.85,
                })
            else:
                suggestions.append({
                    "n": min_n,
                    "density": 0.3,
                    "b_c": 10000,
                    "b_q": 512,
                    "reason": "Even small n is negative; try easier density",
                    "expected_information_gain": 0.7,
                })

        # Fill remaining budget with density variations.
        while len(suggestions) < budget:
            base = suggestions[-1] if suggestions else {"n": 20, "density": 0.5}
            for d in [0.3, 0.4, 0.6, 0.7]:
                if len(suggestions) >= budget:
                    break
                suggestions.append({
                    "n": base["n"],
                    "density": d,
                    "b_c": 5000,
                    "b_q": 256,
                    "reason": f"Density variation at n={base['n']}, d={d}",
                    "expected_information_gain": 0.5,
                })

        return suggestions[:budget]

    # ------------------------------------------------------------------
    # Region classification
    # ------------------------------------------------------------------
    def classify_region(
        self,
        n: int,
        hardness: Any,
        b_c: float,
    ) -> str:
        """Classify a parameter region as classical-dominated, transition,
        or quantum-favourable.

        Uses the existing data points to estimate the expected A_Q at the
        given parameters.  If no data is available, uses heuristics based
        on problem size and classical budget.
        """
        # Find nearby points (same n or within +/- 2).
        nearby = [
            p for p in self.points
            if abs(p.n - n) <= 2 and abs(p.b_c - b_c) <= b_c * 0.5
        ]

        if nearby:
            mean_aq = float(np.mean([p.a_q for p in nearby]))
            if mean_aq > 0.05:
                return "quantum-favorable"
            elif mean_aq < -0.05:
                return "classical-dominated"
            else:
                return "transition"

        # Heuristic classification based on n and budget.
        # Small n with large budget: classical-dominated.
        # Large n with small budget: quantum-favorable (classical struggles).
        # Mid: transition.
        if n <= 18:
            return "classical-dominated"
        elif n >= 30:
            return "quantum-favorable"
        else:
            # Transition band: depends on budget.
            if b_c >= 50000:
                return "classical-dominated"
            elif b_c <= 1000:
                return "quantum-favorable"
            else:
                return "transition"

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_phase_diagram(
        self,
        output_path: str | Path,
        x_axis: str = "n",
        y_axis: str = "b_c",
        color: str = "a_q",
    ) -> None:
        """Generate a 2D phase diagram scatter plot.

        Each point is plotted at (x_axis, y_axis) and coloured by the
        ``color`` field.  A colourbar shows the A_Q scale.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.points:
            logger.warning("phase_diagram.no_data_to_plot")
            return

        xs = [self._get_field(p, x_axis) for p in self.points]
        ys = [self._get_field(p, y_axis) for p in self.points]
        colors = [self._get_field(p, color) for p in self.points]

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            xs, ys, c=colors, cmap="RdYlGn", s=100, edgecolors="black",
            linewidths=0.5, vmin=-max(abs(min(colors)), abs(max(colors))),
            vmax=max(abs(min(colors)), abs(max(colors))),
        )
        plt.colorbar(scatter, label=color)
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)
        ax.set_title(f"Quantum Advantage Phase Diagram ({color} vs {x_axis}, {y_axis})", fontsize=13)
        ax.grid(True, alpha=0.3)

        # Add a horizontal line at A_Q=0 if color is a_q.
        if color == "a_q":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
        logger.info("phase_diagram.plot_saved", path=str(output_path))

    def plot_3d_phase(
        self,
        output_path: str | Path,
        x_axis: str = "n",
        y_axis: str = "b_c",
        z_axis: str = "a_q",
    ) -> None:
        """Generate a 3D phase diagram if data permits.

        Requires at least 4 data points.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if len(self.points) < 4:
            logger.warning("phase_diagram.insufficient_data_for_3d", n_points=len(self.points))
            return

        xs = [self._get_field(p, x_axis) for p in self.points]
        ys = [self._get_field(p, y_axis) for p in self.points]
        zs = [self._get_field(p, z_axis) for p in self.points]

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Colour by z value.
        z_arr = np.array(zs, dtype=float)
        norm_z = (z_arr - z_arr.min()) / max(z_arr.max() - z_arr.min(), 1e-9)
        scatter = ax.scatter(
            xs, ys, zs, c=zs, cmap="RdYlGn", s=80, edgecolors="black",
            linewidths=0.5,
        )
        fig.colorbar(scatter, ax=ax, label=z_axis)

        ax.set_xlabel(x_axis, fontsize=11)
        ax.set_ylabel(y_axis, fontsize=11)
        ax.set_zlabel(z_axis, fontsize=11)
        ax.set_title("3D Quantum Advantage Phase Diagram", fontsize=13)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
        logger.info("phase_diagram.3d_plot_saved", path=str(output_path))

    @staticmethod
    def _get_field(point: PhaseDiagramPoint, field_name: str) -> float:
        """Extract a numeric field from a point."""
        val = getattr(point, field_name, 0.0)
        if isinstance(val, str):
            # Map hardness strings to numbers.
            mapping = {
                "easy": 0.0,
                "medium": 0.5,
                "hard": 1.0,
                "classically_easy": 0.0,
                "borderline": 0.5,
                "classically_hard": 1.0,
                "unknown": 0.5,
            }
            return mapping.get(val, 0.5)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the phase diagram to a dict."""
        return {
            "points": [p.to_dict() for p in self.points],
            "boundary": self.find_boundary(),
        }

    def save(self, path: str | Path) -> None:
        """Save the phase diagram to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def load(self, path: str | Path) -> None:
        """Load a phase diagram from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.points = [
            PhaseDiagramPoint(**{k: v for k, v in p.items() if k in PhaseDiagramPoint.__dataclass_fields__})
            for p in data.get("points", [])
        ]


if __name__ == "__main__":
    pd = PhaseDiagram()

    # Add some synthetic data points.
    for n in [18, 20, 22, 24, 26, 28, 30]:
        a_q = 5.0 - 0.3 * (n - 18) + np.random.RandomState(n).normal(0, 0.5)
        pd.add_point(PhaseDiagramPoint(
            n=n, hardness="medium", b_c=5000, b_q=256,
            representation="qaoa_p2", kernel="QAOA", chi=64,
            a_q=float(a_q),
            diagnosis="synthetic test point",
        ))

    boundary = pd.find_boundary()
    print(f"Boundary: {boundary['boundary_n']}")
    print(f"Per-n A_Q: {boundary['per_n']}")

    suggestions = pd.suggest_next_experiment(budget=3)
    for s in suggestions:
        print(f"Suggestion: n={s['n']}, reason={s['reason']}")

    for n in [16, 24, 32]:
        region = pd.classify_region(n, "medium", 5000)
        print(f"n={n}: {region}")

    pd.plot_phase_diagram("/tmp/phase_diagram_2d.png")
    pd.plot_3d_phase("/tmp/phase_diagram_3d.png")
    print("Plots saved.")
