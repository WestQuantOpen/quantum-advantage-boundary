"""V2 orchestrator: runs all research strands in priority order.

Priority:
1. Recovery erasure (diagnose negative results)
2. Oracle test (per-instance best quantum)
3. QeMCMC + parallel tempering
4. Hardness-conditioned instances + QCSC
5. Simulability-aware search
6. Peaked circuit secondary track

GPU budget allocation:
  Recovery-erasure + diagnostics: 10%
  Oracle quantum search: 15%
  QeMCMC + PT: 25%
  Hard-instance / QCSC: 25%
  MPS/PPS simulability: 15%
  Peaked-circuit: 10%
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.classical_samplers import (
    GreedySampler,
    ParallelTemperingSampler,
    SimulatedAnnealingSampler,
    UniformRandomSampler,
)
from src.exact_solver import solve_mwis_exact
from src.metrics import compute_metrics, quantum_advantage_metric
from src.problems import MWISInstance, generate_mwis_instance
from src.quantum_circuits import build_mwis_qaoa_circuit, build_random_qaoa, get_circuit_metrics
from src.recovery import recover

from src.v2.distribution_analysis import DistributionAnalysis
from src.v2.peaked_circuit import PeakedCircuitSearch
from src.v2.phase_diagram import PhaseDiagram, PhaseDiagramPoint
from src.v2.quantum_objective import RecoveryObjective
from src.v2.simulability import SimulabilityAttack

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

__all__ = ["OrchestratorV2"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _adapter_usable(adapter: Any) -> bool:
    """Check whether the adapter has a live BlueQubit client."""
    client = getattr(adapter, "_client", None)
    return client is not None


# ---------------------------------------------------------------------------
# Budget allocation fractions (must sum to 1.0)
# ---------------------------------------------------------------------------
BUDGET_FRACTIONS: Dict[str, float] = {
    "recovery_erasure": 0.10,
    "oracle_test": 0.15,
    "qemcmc": 0.25,
    "hardness_qcsc": 0.25,
    "simulability": 0.15,
    "peaked": 0.10,
}


class OrchestratorV2:
    """V2 multi-strand orchestrator.

    Runs six research strands in priority order, collects results, and
    synthesises a quantum advantage phase diagram.  Each strand is
    independently callable and returns a results dict.
    """

    def __init__(
        self,
        bluequbit_adapter: Any | None = None,
        output_dir: str = ".",
        max_cost_usd: float = 10.0,
    ) -> None:
        self.adapter = bluequbit_adapter
        self.output_dir = Path(output_dir)
        self.max_cost_usd = max_cost_usd

        self.state_dir = self.output_dir / "state"
        self.results_dir = self.output_dir / "results"
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.results_dir / "figures"

        for d in [self.state_dir, self.results_dir, self.reports_dir,
                  self.figures_dir, self.results_dir / "raw",
                  self.results_dir / "processed"]:
            d.mkdir(parents=True, exist_ok=True)

        self.phase_diagram = PhaseDiagram()
        self.all_results: List[Dict[str, Any]] = []
        self.strand_results: Dict[str, List[Dict[str, Any]]] = {
            "recovery_erasure": [],
            "oracle_test": [],
            "qemcmc": [],
            "hardness_qcsc": [],
            "simulability": [],
            "peaked": [],
        }
        self.decision_log: List[Dict[str, Any]] = []
        self.compute_used = 0.0

    # ------------------------------------------------------------------
    # Strand 1: Recovery Erasure
    # ------------------------------------------------------------------
    def run_strand_1_recovery_erasure(self) -> Dict[str, Any]:
        """Run recovery erasure on existing v1 results.

        Diagnosis strand: take the negative results from v1 (where quantum
        showed no advantage up to N=28) and determine whether the negative
        result is due to:
        - Recovery erasure (classical recovery is so strong it erases any
          quantum signal), or
        - Genuine quantum weakness (the quantum distribution is not better).

        For each instance, compare the pre-recovery and post-recovery
        distributions to see how much of the quantum signal is erased.
        """
        self._log("Strand 1: Recovery erasure diagnosis starting")

        results: List[Dict[str, Any]] = []
        sizes = [20, 24, 28]
        n_seeds = 3

        for n in sizes:
            for seed in range(n_seeds):
                instance = generate_mwis_instance(n=n, density=0.5, seed=seed)
                if n <= 28:
                    opt, opt_val, _ = solve_mwis_exact(instance)
                    instance.optimal_solution = opt
                    instance.optimal_value = opt_val

                # Quantum: QAOA
                circuit, _, _, _ = build_random_qaoa(instance, p=2, seed=seed)
                q_counts = self._run_circuit(circuit, n_shots=256)

                # Classical: SA
                sa = SimulatedAnnealingSampler()
                c_samples, _ = sa.sample(instance, n_shots=256, seed=seed)

                # Pre-recovery comparison.
                q_samples = self._counts_to_samples(q_counts, n)
                q_pre_best = max(
                    (float(np.dot(instance.weights, s.astype(float))) for s in q_samples),
                    default=0.0,
                )
                c_pre_best = max(
                    (float(np.dot(instance.weights, s.astype(float))) for s in c_samples),
                    default=0.0,
                )

                # Post-recovery.
                q_rec = recover(q_samples, instance, max_expansion=2, max_candidates=50000)
                c_rec = recover(c_samples, instance, max_expansion=2, max_candidates=50000)

                # Recovery erasure: how much of the pre-recovery gap is erased?
                pre_gap = q_pre_best - c_pre_best
                post_gap = q_rec["post_recovery_best"] - c_rec["post_recovery_best"]
                erasure = pre_gap - post_gap  # positive = recovery erased quantum advantage

                result = {
                    "strand": "recovery_erasure",
                    "n": n, "seed": seed,
                    "q_pre_best": q_pre_best,
                    "c_pre_best": c_pre_best,
                    "q_post_best": q_rec["post_recovery_best"],
                    "c_post_best": c_rec["post_recovery_best"],
                    "pre_gap": pre_gap,
                    "post_gap": post_gap,
                    "erasure": erasure,
                    "a_q": post_gap,
                    "timestamp": _utcnow(),
                }
                results.append(result)
                self.compute_used += 1

        self.strand_results["recovery_erasure"] = results
        self.all_results.extend(results)
        self._update_phase_diagram(results)

        mean_erasure = float(np.mean([r["erasure"] for r in results])) if results else 0.0
        self._log(f"Strand 1 complete: {len(results)} experiments, mean erasure={mean_erasure:.4f}")
        return {"strand": "recovery_erasure", "results": results, "mean_erasure": mean_erasure}

    # ------------------------------------------------------------------
    # Strand 2: Oracle Test
    # ------------------------------------------------------------------
    def run_strand_2_oracle_test(self) -> Dict[str, Any]:
        """Run oracle quantum search (per-instance best quantum).

        For each instance, search over QAOA parameters to find the best
        quantum circuit using the recovery-optimised objective.
        """
        self._log("Strand 2: Oracle quantum search starting")

        results: List[Dict[str, Any]] = []
        sizes = [22, 24, 26]
        n_seeds = 2

        for n in sizes:
            for seed in range(n_seeds):
                instance = generate_mwis_instance(n=n, density=0.5, seed=seed)
                if n <= 26:
                    opt, opt_val, _ = solve_mwis_exact(instance)
                    instance.optimal_solution = opt
                    instance.optimal_value = opt_val

                # Recovery-optimised objective search.
                objective = RecoveryObjective(instance, recovery_budget=5000, n_shots=128)
                best = objective.optimize_parameters(
                    instance, p=2, n_iterations=20, adapter=self.adapter,
                )

                # Classical baseline.
                sa = SimulatedAnnealingSampler()
                c_samples, _ = sa.sample(instance, n_shots=256, seed=seed)
                c_rec = recover(c_samples, instance, max_expansion=2, max_candidates=50000)

                a_q = best["metrics"].get("post_recovery_best", 0) - c_rec["post_recovery_best"]

                result = {
                    "strand": "oracle_test",
                    "n": n, "seed": seed,
                    "best_score": best["score"],
                    "q_post_best": best["metrics"].get("post_recovery_best", 0),
                    "c_post_best": c_rec["post_recovery_best"],
                    "a_q": a_q,
                    "gammas": best["gammas"],
                    "betas": best["betas"],
                    "timestamp": _utcnow(),
                }
                results.append(result)
                self.compute_used += 1

        self.strand_results["oracle_test"] = results
        self.all_results.extend(results)
        self._update_phase_diagram(results)

        mean_aq = float(np.mean([r["a_q"] for r in results])) if results else 0.0
        self._log(f"Strand 2 complete: {len(results)} experiments, mean A_Q={mean_aq:.4f}")
        return {"strand": "oracle_test", "results": results, "mean_a_q": mean_aq}

    # ------------------------------------------------------------------
    # Strand 3: QeMCMC + Parallel Tempering
    # ------------------------------------------------------------------
    def run_strand_3_qemcmc(self) -> Dict[str, Any]:
        """Run QeMCMC experiments.

        Quantum-enhanced MCMC: use quantum samples as proposals in a
        classical MCMC chain.  Compare against pure parallel tempering.
        """
        self._log("Strand 3: QeMCMC + parallel tempering starting")

        results: List[Dict[str, Any]] = []
        sizes = [20, 24, 28]
        n_seeds = 2

        for n in sizes:
            for seed in range(n_seeds):
                instance = generate_mwis_instance(n=n, density=0.5, seed=seed)
                if n <= 28:
                    opt, opt_val, _ = solve_mwis_exact(instance)
                    instance.optimal_solution = opt
                    instance.optimal_value = opt_val

                # Pure parallel tempering baseline.
                pt = ParallelTemperingSampler(n_chains=6, n_steps=500)
                pt_samples, _ = pt.sample(instance, n_shots=256, seed=seed)
                pt_rec = recover(pt_samples, instance, max_expansion=2, max_candidates=50000)

                # QeMCMC: use QAOA samples as MCMC proposals.
                circuit, _, _, _ = build_random_qaoa(instance, p=2, seed=seed)
                q_counts = self._run_circuit(circuit, n_shots=256)
                q_samples = self._counts_to_samples(q_counts, n)

                # Mix quantum proposals with local search (simple QeMCMC).
                rng = np.random.RandomState(seed)
                mixed_samples: List[np.ndarray] = []
                for q_sample in q_samples:
                    # Accept quantum proposal if it improves on a random local state.
                    local = pt_samples[rng.randint(len(pt_samples))]
                    q_val = float(np.dot(instance.weights, q_sample.astype(float)))
                    local_val = float(np.dot(instance.weights, local.astype(float)))
                    if q_val >= local_val or rng.random() < 0.3:
                        mixed_samples.append(q_sample)
                    else:
                        mixed_samples.append(local)

                qemcmc_rec = recover(mixed_samples, instance, max_expansion=2, max_candidates=50000)

                a_q = qemcmc_rec["post_recovery_best"] - pt_rec["post_recovery_best"]

                result = {
                    "strand": "qemcmc",
                    "n": n, "seed": seed,
                    "qemcmc_post_best": qemcmc_rec["post_recovery_best"],
                    "pt_post_best": pt_rec["post_recovery_best"],
                    "a_q": a_q,
                    "timestamp": _utcnow(),
                }
                results.append(result)
                self.compute_used += 1

        self.strand_results["qemcmc"] = results
        self.all_results.extend(results)
        self._update_phase_diagram(results)

        mean_aq = float(np.mean([r["a_q"] for r in results])) if results else 0.0
        self._log(f"Strand 3 complete: {len(results)} experiments, mean A_Q={mean_aq:.4f}")
        return {"strand": "qemcmc", "results": results, "mean_a_q": mean_aq}

    # ------------------------------------------------------------------
    # Strand 4: Hardness-conditioned instances + QCSC
    # ------------------------------------------------------------------
    def run_strand_4_hardness_qcsc(self) -> Dict[str, Any]:
        """Run hardness-conditioned + QCSC experiments.

        Generate instances conditioned on hardness (easy/medium/hard) and
        run Quantum-Conditioned Simulated Annealing (QCSC): use quantum
        samples to seed or guide classical SA.
        """
        self._log("Strand 4: Hardness-conditioned instances + QCSC starting")

        results: List[Dict[str, Any]] = []

        # Hardness-conditioned instances: vary density to control hardness.
        configs = [
            (22, 0.3, "easy"),
            (24, 0.5, "medium"),
            (26, 0.7, "hard"),
        ]

        for n, density, hardness_label in configs:
            for seed in range(3):
                instance = generate_mwis_instance(n=n, density=density, seed=seed)
                if n <= 26:
                    opt, opt_val, _ = solve_mwis_exact(instance)
                    instance.optimal_solution = opt
                    instance.optimal_value = opt_val
                instance.hardness = hardness_label

                # Classical SA baseline.
                sa = SimulatedAnnealingSampler(n_chains=4, n_steps=500)
                sa_samples, _ = sa.sample(instance, n_shots=256, seed=seed)
                sa_rec = recover(sa_samples, instance, max_expansion=2, max_candidates=50000)

                # QCSC: quantum-seeded SA.
                circuit, _, _, _ = build_random_qaoa(instance, p=2, seed=seed)
                q_counts = self._run_circuit(circuit, n_shots=128)
                q_samples = self._counts_to_samples(q_counts, n)

                # Use quantum samples as seeds for SA chains.
                rng = np.random.RandomState(seed + 1000)
                qcsc_samples: List[np.ndarray] = []
                for i in range(256):
                    if i < len(q_samples) and rng.random() < 0.5:
                        seed_state = q_samples[i % len(q_samples)]
                    else:
                        seed_state = sa_samples[i % len(sa_samples)]
                    # Short SA refinement from the seed.
                    refined = self._local_refine(seed_state, instance, rng, n_steps=50)
                    qcsc_samples.append(refined)

                qcsc_rec = recover(qcsc_samples, instance, max_expansion=2, max_candidates=50000)

                a_q = qcsc_rec["post_recovery_best"] - sa_rec["post_recovery_best"]

                result = {
                    "strand": "hardness_qcsc",
                    "n": n, "density": density, "seed": seed,
                    "hardness": hardness_label,
                    "qcsc_post_best": qcsc_rec["post_recovery_best"],
                    "sa_post_best": sa_rec["post_recovery_best"],
                    "a_q": a_q,
                    "timestamp": _utcnow(),
                }
                results.append(result)
                self.compute_used += 1

        self.strand_results["hardness_qcsc"] = results
        self.all_results.extend(results)
        self._update_phase_diagram(results)

        mean_aq = float(np.mean([r["a_q"] for r in results])) if results else 0.0
        self._log(f"Strand 4 complete: {len(results)} experiments, mean A_Q={mean_aq:.4f}")
        return {"strand": "hardness_qcsc", "results": results, "mean_a_q": mean_aq}

    # ------------------------------------------------------------------
    # Strand 5: Simulability-aware search
    # ------------------------------------------------------------------
    def run_strand_5_simulability(self) -> Dict[str, Any]:
        """Run MPS/PPS simulability attack.

        For promising quantum circuits, run the MPS bond-dimension sweep
        and Pauli-path verification to determine classical simulability.
        """
        self._log("Strand 5: MPS/PPS simulability attack starting")

        attack = SimulabilityAttack(bluequbit_adapter=self.adapter)
        results: List[Dict[str, Any]] = []

        sizes = [20, 24]
        n_seeds = 2

        for n in sizes:
            for seed in range(n_seeds):
                instance = generate_mwis_instance(n=n, density=0.5, seed=seed)
                if n <= 26:
                    opt, opt_val, _ = solve_mwis_exact(instance)
                    instance.optimal_solution = opt
                    instance.optimal_value = opt_val

                circuit, _, _, _ = build_random_qaoa(instance, p=2, seed=seed)

                # MPS sweep.
                sim_limit = attack.find_simulability_limit(
                    circuit, instance,
                    bond_dims=[8, 16, 32, 64],
                    n_shots=256,
                )

                # Pauli-path verification.
                pp_result = attack.pauli_path_verify(circuit, n_shots=256)

                result = {
                    "strand": "simulability",
                    "n": n, "seed": seed,
                    "simulability_limit": sim_limit["simulability_limit"],
                    "hardness": sim_limit["hardness"],
                    "u_exact": sim_limit["u_exact"],
                    "per_chi": sim_limit["per_chi"],
                    "pauli_path_verdict": pp_result["verdict"],
                    "pauli_path_js": pp_result["js_divergence"],
                    "timestamp": _utcnow(),
                }
                results.append(result)
                self.compute_used += 1

        self.strand_results["simulability"] = results
        self.all_results.extend(results)

        # Add to phase diagram with chi info.
        for r in results:
            self.phase_diagram.add_point(PhaseDiagramPoint(
                n=r["n"],
                hardness=r["hardness"],
                b_c=5000,
                b_q=256,
                representation="qaoa_p2",
                kernel="QAOA",
                chi=r["simulability_limit"] or 128,
                a_q=0.0,  # simulability strand doesn't directly measure A_Q
                diagnosis=f"sim_limit={r['simulability_limit']}, pp={r['pauli_path_verdict']}",
            ))

        n_hard = sum(1 for r in results if r["hardness"] == "classically hard")
        self._log(f"Strand 5 complete: {len(results)} experiments, {n_hard} classically hard")
        return {
            "strand": "simulability",
            "results": results,
            "n_classically_hard": n_hard,
        }

    # ------------------------------------------------------------------
    # Strand 6: Peaked circuit secondary track
    # ------------------------------------------------------------------
    def run_strand_6_peaked(self) -> Dict[str, Any]:
        """Run peaked circuit search.

        Secondary 10% sideline: search for circuits with strong verifiable
        peaks, high entanglement, and poor MPS approximability.
        """
        self._log("Strand 6: Peaked circuit search starting")

        search = PeakedCircuitSearch(n_qubits=16, depth_range=(2, 8))
        results = search.search(
            n_candidates=20,
            n_qubits=16,
            adapter=self.adapter,
            n_shots=512,
        )

        strand_results: List[Dict[str, Any]] = []
        for r in results[:10]:  # Top 10
            result = {
                "strand": "peaked",
                "candidate_id": r["candidate_id"],
                "n_qubits": r["n_qubits"],
                "depth": r["depth"],
                "peak_strength": r["peak_strength"],
                "normalized_peak": r["normalized_peak"],
                "mean_entropy": r["entanglement"]["mean_bipartite_entropy"],
                "mps_score": r["mps_approximability"]["approximability_score"],
                "peaked_score": r["peaked_score"],
                "timestamp": _utcnow(),
            }
            strand_results.append(result)
            self.compute_used += 1

        self.strand_results["peaked"] = strand_results
        self.all_results.extend(strand_results)

        best_score = max((r["peaked_score"] for r in strand_results), default=0.0)
        self._log(f"Strand 6 complete: {len(strand_results)} circuits, best score={best_score:.4f}")
        return {
            "strand": "peaked",
            "results": strand_results,
            "best_peaked_score": best_score,
        }

    # ------------------------------------------------------------------
    # Campaign
    # ------------------------------------------------------------------
    def run_campaign(self) -> Dict[str, Any]:
        """Run all strands in priority order, collect results, generate
        phase diagram.
        """
        self._log("V2 campaign starting")
        start_time = time.perf_counter()

        strand_outputs: Dict[str, Any] = {}

        # Run strands in priority order.
        strand_outputs["recovery_erasure"] = self.run_strand_1_recovery_erasure()
        strand_outputs["oracle_test"] = self.run_strand_2_oracle_test()
        strand_outputs["qemcmc"] = self.run_strand_3_qemcmc()
        strand_outputs["hardness_qcsc"] = self.run_strand_4_hardness_qcsc()
        strand_outputs["simulability"] = self.run_strand_5_simulability()
        strand_outputs["peaked"] = self.run_strand_6_peaked()

        # Generate reports.
        self._generate_v2_reports()

        # Save phase diagram.
        self.phase_diagram.save(self.state_dir / "phase_diagram.json")

        elapsed = time.perf_counter() - start_time
        self._log(f"V2 campaign complete in {elapsed:.1f}s, {self.compute_used} compute units")

        return {
            "strand_outputs": strand_outputs,
            "phase_diagram": self.phase_diagram.find_boundary(),
            "total_experiments": len(self.all_results),
            "compute_used": self.compute_used,
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Phase diagram update
    # ------------------------------------------------------------------
    def _update_phase_diagram(self, strand_results: List[Dict[str, Any]]) -> None:
        """Add results to the phase diagram."""
        for r in strand_results:
            self.phase_diagram.add_point(PhaseDiagramPoint(
                n=int(r.get("n", 0)),
                hardness=r.get("hardness", "unknown"),
                b_c=float(r.get("b_c", r.get("recovery_budget", 5000))),
                b_q=float(r.get("b_q", r.get("n_shots", 256))),
                representation=str(r.get("representation", r.get("strand", "unknown"))),
                kernel=str(r.get("kernel", "QAOA")),
                chi=int(r.get("chi", 0)),
                a_q=float(r.get("a_q", 0.0)),
                diagnosis=str(r.get("diagnosis", r.get("strand", ""))),
            ))

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def _generate_v2_reports(self) -> None:
        """Generate v2 reports and figures."""
        # Summary report.
        self._write_report("v2_campaign_report.md", self._format_campaign_report())

        # Per-strand summaries.
        for strand_name, results in self.strand_results.items():
            if results:
                self._write_report(
                    f"v2_strand_{strand_name}.md",
                    self._format_strand_report(strand_name, results),
                )

        # Phase diagram plots.
        if self.phase_diagram.points:
            self.phase_diagram.plot_phase_diagram(
                self.figures_dir / "v2_phase_diagram_2d.png",
                x_axis="n", y_axis="b_c", color="a_q",
            )
            self.phase_diagram.plot_3d_phase(
                self.figures_dir / "v2_phase_diagram_3d.png",
            )

        # Summary CSV.
        self._write_summary_csv()

        # Decision log.
        self._write_decision_log()

        # Save all raw results.
        with open(self.results_dir / "raw" / "v2_all_results.json", "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)

        logger.info("v2_reports_generated", dir=str(self.reports_dir))

    def _format_campaign_report(self) -> str:
        boundary = self.phase_diagram.find_boundary()
        per_n = boundary.get("per_n", {})

        lines = [
            "# V2 Campaign Report\n",
            f"Generated: {_utcnow()}\n",
            f"## Summary\n",
            f"- Total experiments: {len(self.all_results)}\n",
            f"- Compute used: {self.compute_used}\n",
            f"- Phase diagram points: {len(self.phase_diagram.points)}\n",
            f"\n## Strand Results\n",
        ]

        for strand_name, results in self.strand_results.items():
            n = len(results)
            mean_aq = float(np.mean([r.get("a_q", 0) for r in results])) if results else 0.0
            lines.append(f"- **{strand_name}**: {n} experiments, mean A_Q={mean_aq:.4f}\n")

        lines.append(f"\n## Phase Boundary\n")
        lines.append(f"- Boundary N: {boundary.get('boundary_n')}\n")
        if per_n:
            lines.append(f"- Per-N A_Q:\n")
            for n in sorted(per_n.keys()):
                lines.append(f"  - n={n}: A_Q={per_n[n]:.4f}\n")

        # Next experiment suggestions.
        suggestions = self.phase_diagram.suggest_next_experiment(budget=3)
        lines.append(f"\n## Suggested Next Experiments\n")
        for i, s in enumerate(suggestions):
            lines.append(f"{i+1}. n={s['n']}, density={s.get('density', 0.5)}: {s['reason']}\n")

        return "".join(lines)

    def _format_strand_report(
        self, strand_name: str, results: List[Dict[str, Any]]
    ) -> str:
        lines = [
            f"# V2 Strand: {strand_name}\n\n",
            f"Experiments: {len(results)}\n\n",
            f"## Results\n\n",
        ]

        for r in results:
            n = r.get("n", "?")
            seed = r.get("seed", "?")
            a_q = r.get("a_q", 0.0)
            lines.append(f"- n={n}, seed={seed}: A_Q={a_q:.4f}\n")

        aqs = [r.get("a_q", 0.0) for r in results]
        if aqs:
            lines.append(f"\n## Statistics\n\n")
            lines.append(f"- Mean A_Q: {np.mean(aqs):.4f}\n")
            lines.append(f"- Max A_Q: {np.max(aqs):.4f}\n")
            lines.append(f"- Min A_Q: {np.min(aqs):.4f}\n")
            lines.append(f"- Fraction A_Q > 0: {sum(1 for a in aqs if a > 0) / len(aqs):.1%}\n")

        return "".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_circuit(self, circuit: Any, n_shots: int) -> Dict[str, int]:
        """Run a circuit via the adapter or local sim."""
        if self.adapter is not None and _adapter_usable(self.adapter):
            try:
                device = self.adapter.select_device(circuit.num_qubits, prefer_gpu=True)
                result = self.adapter.run(circuit, device=device, shots=n_shots)
                counts = result.get("counts", {})
                if counts:
                    return counts
            except Exception as exc:
                logger.warning("v2.run_failed", error=repr(exc))

        from qiskit import transpile
        from qiskit_aer import AerSimulator

        sim = AerSimulator()
        t_circ = transpile(circuit, sim)
        sim_result = sim.run(t_circ, shots=n_shots).result()
        return sim_result.get_counts()

    @staticmethod
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

    @staticmethod
    def _local_refine(
        state: np.ndarray,
        instance: MWISInstance,
        rng: np.random.RandomState,
        n_steps: int = 50,
    ) -> np.ndarray:
        """Simple local search refinement from a seed state."""
        n = instance.n
        weights = instance.weights
        adjacency = instance.adjacency
        current = state.astype(int).copy()
        current_val = float(np.dot(weights, current.astype(float)))

        for _ in range(n_steps):
            if n == 0:
                break
            v = rng.randint(n)
            candidate = current.copy()
            candidate[v] = 1 - candidate[v]
            # Check feasibility.
            selected = candidate.astype(bool)
            sub = adjacency[np.ix_(selected, selected)]
            if np.any(sub.astype(bool)):
                continue
            cand_val = float(np.dot(weights, candidate.astype(float)))
            if cand_val > current_val:
                current = candidate
                current_val = cand_val

        return current

    def _log(self, decision: str) -> None:
        entry = {"timestamp": _utcnow(), "decision": decision}
        self.decision_log.append(entry)
        logger.info("v2.orchestrator", decision=decision)

    def _write_report(self, filename: str, content: str) -> None:
        path = self.reports_dir / filename
        with open(path, "w") as f:
            f.write(content)
        logger.info("v2.report_written", path=str(path))

    def _write_summary_csv(self) -> None:
        import csv

        path = self.results_dir / "processed" / "v2_summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "strand", "n", "seed", "a_q", "timestamp",
            ])
            writer.writeheader()
            for r in self.all_results:
                writer.writerow({
                    "strand": r.get("strand", ""),
                    "n": r.get("n", ""),
                    "seed": r.get("seed", ""),
                    "a_q": r.get("a_q", 0),
                    "timestamp": r.get("timestamp", ""),
                })

    def _write_decision_log(self) -> None:
        path = self.state_dir / "v2_decision_log.jsonl"
        with open(path, "w") as f:
            for entry in self.decision_log:
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Quick smoke test with the peaked-circuit strand (fastest).
    orchestrator = OrchestratorV2(output_dir="/tmp/v2_output", max_cost_usd=10.0)
    result = orchestrator.run_strand_6_peaked()
    print(f"Peaked strand: {len(result['results'])} circuits, "
          f"best score={result['best_peaked_score']:.4f}")
    print(f"Phase diagram points: {len(orchestrator.phase_diagram.points)}")
    print("OrchestratorV2 smoke test passed. Run run_campaign() for full campaign.")
