"""Experiment execution for the quantum-advantage boundary hunter.

The :class:`ExperimentRunner` orchestrates the full pipeline for a single
experiment:

    sample  ->  recover  ->  metrics  ->  save

It is designed to be self-contained: when no BlueQubit adapter is supplied it
falls back to a local simulator, and when no custom sampler / recovery
strategy is supplied it uses a built-in MaxCut QAOA sampler and a trivial
identity recovery.
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

__all__ = ["ExperimentConfig", "ExperimentRunner"]

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    """Configuration for a single experiment."""

    experiment_id: str
    round: int
    n: int
    density: float
    seed: int
    n_shots: int = 1000
    p: int = 1
    sampler_name: str = "qaoa"
    recovery_budget: float = 0.0
    representation_id: str | None = None
    device: str = "cpu"
    tags: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Graph / MaxCut helpers (self-contained)
# ---------------------------------------------------------------------------
def _extract_graph(instance: Any) -> Any:
    """Return a networkx Graph from a flexible ``instance`` argument."""
    if nx is not None and isinstance(instance, nx.Graph):
        return instance
    if isinstance(instance, dict) and "graph" in instance:
        return instance["graph"]
    if isinstance(instance, dict) and "edges" in instance:
        g = nx.Graph() if nx is not None else None
        if g is not None:
            g.add_edges_from(instance["edges"])
            return g
    return instance


def _graph_edges(graph: Any) -> list[tuple[int, int]]:
    if graph is None:
        return []
    if nx is not None and isinstance(graph, nx.Graph):
        return [(int(u), int(v)) for u, v in graph.edges()]
    if hasattr(graph, "edges"):
        return [(int(u), int(v)) for u, v in graph.edges()]
    return []


def _graph_num_nodes(graph: Any) -> int:
    if graph is None:
        return 0
    if nx is not None and isinstance(graph, nx.Graph):
        return graph.number_of_nodes()
    if hasattr(graph, "number_of_nodes"):
        return graph.number_of_nodes()
    edges = _graph_edges(graph)
    return max((max(u, v) for u, v in edges), default=-1) + 1


def _maxcut_value(bitstring: str, graph: Any) -> int:
    edges = _graph_edges(graph)
    val = 0
    for u, v in edges:
        if u < len(bitstring) and v < len(bitstring) and bitstring[u] != bitstring[v]:
            val += 1
    return val


def _brute_force_maxcut(graph: Any) -> int:
    n = _graph_num_nodes(graph)
    edges = _graph_edges(graph)
    if n == 0 or not edges:
        return 0
    if n > 22:
        return len(edges)
    best = 0
    for mask in range(1 << n):
        bs = format(mask, f"0{n}b")
        val = _maxcut_value(bs, graph)
        if val > best:
            best = val
    return best


def _build_qaoa_circuit(
    graph: Any,
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> Any:
    """Build a standard MaxCut QAOA circuit (x-mixer)."""
    from qiskit import QuantumCircuit

    n = _graph_num_nodes(graph)
    edges = _graph_edges(graph)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for layer in range(p):
        gamma = float(gammas[layer]) if layer < len(gammas) else 0.0
        beta = float(betas[layer]) if layer < len(betas) else 0.0
        for u, v in edges:
            if u == v:
                continue
            qc.rzz(2.0 * gamma, u, v)
        for i in range(n):
            qc.rx(2.0 * beta, i)

    qc.measure(range(n), range(n))
    return qc


def _run_local(circuit: Any, n_shots: int) -> dict[str, float]:
    """Run a circuit locally and return probability-normalised counts."""
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        backend = AerSimulator()
        tqc = transpile(circuit, backend)
        job = backend.run(tqc, shots=n_shots)
        counts = job.result().get_counts()
    except Exception:
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler()
        job = sampler.run([circuit], shots=n_shots)
        counts = job.result()[0].data.c.get_counts()

    normalised: dict[str, float] = {}
    total = sum(counts.values())
    if total == 0:
        return normalised
    for k, v in counts.items():
        normalised[k.replace(" ", "")] = float(v) / total
    return normalised


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Run single or batched experiments and persist results."""

    def __init__(
        self,
        bluequbit_adapter: Any | None = None,
        output_dir: str = "results",
    ) -> None:
        self.adapter = bluequbit_adapter
        # Resolve output dir relative to the project root when relative.
        op = Path(output_dir)
        if not op.is_absolute():
            op = Path(__file__).resolve().parent.parent / op
        self.output_dir = op

        # Pluggable strategies (override for advanced use).
        self.sampler: Callable[[ExperimentConfig, Any], tuple[Any, dict[str, Any]]] | None = None
        self.recovery: Callable[[dict[str, float], ExperimentConfig, dict[str, Any]], dict[str, float]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_experiment(
        self,
        config: ExperimentConfig,
        instance: Any,
    ) -> dict[str, Any]:
        """Run a single experiment: sample -> recover -> metrics -> save.

        Returns the result dictionary that was persisted.
        """
        graph = _extract_graph(instance)
        logger.info(
            "experiment.start",
            experiment_id=config.experiment_id,
            n=config.n,
            p=config.p,
            sampler=config.sampler_name,
        )

        # 1. Sample ----------------------------------------------------
        circuit, sample_meta = self._sample(config, graph)

        # 2. Execute ---------------------------------------------------
        counts, run_meta = self._execute(circuit, config)

        # 3. Recover ---------------------------------------------------
        recovered_counts, recovery_meta = self._recover(counts, config, sample_meta)

        # 4. Metrics ---------------------------------------------------
        metrics = self._compute_metrics(recovered_counts, graph)

        # 5. Assemble result ------------------------------------------
        result: dict[str, Any] = {
            "experiment_id": config.experiment_id,
            "config": config.model_dump(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sample": sample_meta,
            "execution": run_meta,
            "recovery": recovery_meta,
            "metrics": metrics,
        }

        # 6. Save ------------------------------------------------------
        self._save_result(config, result)
        logger.info(
            "experiment.complete",
            experiment_id=config.experiment_id,
            approx_ratio=round(metrics.get("approximation_ratio", 0.0), 4),
        )
        return result

    def run_batch(
        self,
        configs: Sequence[ExperimentConfig],
        instances: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Run multiple experiments and persist a manifest.

        ``instances`` may be a single instance (applied to all configs) or a
        sequence aligned with ``configs``.
        """
        configs = list(configs)
        if len(instances) == 1 and len(configs) > 1:
            instances = list(instances) * len(configs)
        results: list[dict[str, Any]] = []
        for cfg, inst in zip(configs, instances):
            try:
                res = self.run_experiment(cfg, inst)
                results.append(res)
            except Exception as exc:
                logger.error("experiment.failed", experiment_id=cfg.experiment_id, error=repr(exc))
                results.append(
                    {
                        "experiment_id": cfg.experiment_id,
                        "error": repr(exc),
                        "status": "failed",
                    }
                )
        self._save_manifest(configs, results)
        return results

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def _sample(
        self,
        config: ExperimentConfig,
        graph: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Produce a circuit + metadata for the experiment."""
        if self.sampler is not None:
            return self.sampler(config, graph)

        # Default: MaxCut QAOA with deterministic-ish parameters.
        rng = np.random.default_rng(config.seed)
        gammas = [float(rng.uniform(0, math.pi)) for _ in range(config.p)]
        betas = [float(rng.uniform(0, math.pi / 2)) for _ in range(config.p)]
        circuit = _build_qaoa_circuit(graph, config.p, gammas, betas)
        meta = {
            "sampler_name": config.sampler_name,
            "p": config.p,
            "gammas": gammas,
            "betas": betas,
            "n_qubits": _graph_num_nodes(graph),
            "depth": circuit.depth(),
            "gate_count": circuit.size(),
        }
        return circuit, meta

    def _execute(
        self,
        circuit: Any,
        config: ExperimentConfig,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Execute the circuit and return probability-normalised counts."""
        if self.adapter is not None:
            try:
                result = self.adapter.run(
                    circuit,
                    device=config.device,
                    shots=config.n_shots,
                    job_name=config.experiment_id,
                    tags={"experiment_id": config.experiment_id},
                )
                counts = result.get("counts") or {}
                # Normalise to probabilities if raw counts returned.
                total = sum(counts.values()) if counts else 0
                if total > 0 and any(v > 1.0 for v in counts.values()):
                    counts = {k.replace(" ", ""): float(v) / total for k, v in counts.items()}
                else:
                    counts = {k.replace(" ", ""): float(v) for k, v in counts.items()}
                run_meta = {
                    "backend": result.get("device", config.device),
                    "job_id": result.get("job_id"),
                    "cost": result.get("cost"),
                    "runtime": result.get("runtime"),
                    "shots": config.n_shots,
                }
                return counts, run_meta
            except Exception as exc:
                logger.warning("experiment.adapter_failed", error=repr(exc))

        # Local fallback.
        counts = _run_local(circuit, config.n_shots)
        run_meta = {
            "backend": "local",
            "job_id": None,
            "cost": 0.0,
            "runtime": None,
            "shots": config.n_shots,
        }
        return counts, run_meta

    def _recover(
        self,
        counts: dict[str, float],
        config: ExperimentConfig,
        sample_meta: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Apply error-mitigation / recovery within the budget."""
        if self.recovery is not None:
            recovered = self.recovery(counts, config, sample_meta)
            return recovered, {"method": "custom", "budget": config.recovery_budget}

        # Default identity recovery (no-op).  A non-trivial budget would allow
        # e.g. majority-vote / symmetry restoration in a richer impl.
        return counts, {"method": "identity", "budget": config.recovery_budget}

    def _compute_metrics(self, counts: dict[str, float], graph: Any) -> dict[str, Any]:
        """Compute solution-quality metrics from measurement counts."""
        max_cut = _brute_force_maxcut(graph)
        expected_cut = 0.0
        best_cut = 0
        optimal_prob = 0.0
        n_samples = len(counts)

        for bitstring, prob in counts.items():
            cv = _maxcut_value(bitstring, graph)
            expected_cut += prob * cv
            if cv > best_cut:
                best_cut = cv
            if max_cut > 0 and cv >= max_cut:
                optimal_prob += prob

        approx_ratio = (expected_cut / max_cut) if max_cut > 0 else 0.0
        return {
            "approximation_ratio": approx_ratio,
            "expected_cut": expected_cut,
            "best_cut": best_cut,
            "max_cut": max_cut,
            "success_probability": optimal_prob,
            "n_outcomes": n_samples,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_result(self, config: ExperimentConfig, result: dict[str, Any]) -> None:
        """Save a single result to ``results/raw/{experiment_id}.json``."""
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{config.experiment_id}.json"
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("experiment.saved", path=str(path))

    def _save_manifest(
        self,
        configs: Sequence[ExperimentConfig],
        results: Sequence[dict[str, Any]],
    ) -> None:
        """Save a batch manifest to ``results/processed/manifest.json``."""
        proc_dir = self.output_dir / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        path = proc_dir / "manifest.json"

        manifest = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_experiments": len(configs),
            "experiments": [],
        }
        for cfg, res in zip(configs, results):
            metrics = res.get("metrics", {})
            manifest["experiments"].append(
                {
                    "experiment_id": cfg.experiment_id,
                    "round": cfg.round,
                    "n": cfg.n,
                    "density": cfg.density,
                    "p": cfg.p,
                    "sampler_name": cfg.sampler_name,
                    "representation_id": cfg.representation_id,
                    "device": cfg.device,
                    "status": res.get("status", "completed") if "error" in res else "completed",
                    "approximation_ratio": metrics.get("approximation_ratio"),
                    "success_probability": metrics.get("success_probability"),
                    "error": res.get("error"),
                }
            )

        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        logger.info("experiment.manifest_saved", path=str(path), n=len(configs))
