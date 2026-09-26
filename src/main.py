"""Main entry point for the Quantum Advantage Boundary Hunter."""

from __future__ import annotations

import argparse
import os
import sys

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)

from src.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(
        prog="quantum-advantage-boundary",
        description="AI-orchestrated multi-agent quantum advantage boundary hunter",
    )
    parser.add_argument("--budget", type=int, default=100,
                        help="Maximum number of experiment conditions")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--no-bluequbit", action="store_true",
                        help="Use local simulation only")
    parser.add_argument("--max-cost-usd", type=float, default=10.0,
                        help="Maximum BlueQubit cost in USD")
    args = parser.parse_args()

    use_bq = not args.no_bluequbit and bool(os.getenv("BLUEQUBIT_API_TOKEN"))

    adapter = None
    if use_bq:
        from src.bluequbit_adapter import BlueQubitAdapter
        adapter = BlueQubitAdapter(max_cost_usd=args.max_cost_usd)
        print(f"BlueQubit: enabled (budget=${args.max_cost_usd})")
    else:
        print("BlueQubit: disabled (local Qiskit Aer simulation)")

    print(f"Compute budget: {args.budget} experiment conditions")
    print(f"Output: {args.output}/")
    print()

    orchestrator = Orchestrator(
        bluequbit_adapter=adapter,
        output_dir=args.output,
        max_compute_budget=args.budget,
        use_bluequbit=use_bq,
    )

    report = orchestrator.run_campaign()

    # Print summary
    print("\n" + "=" * 70)
    print("QUANTUM ADVANTAGE BOUNDARY HUNTER — CAMPAIGN COMPLETE")
    print("=" * 70)
    print(f"Experiments completed: {len(orchestrator.all_results)}")
    print(f"BlueQubit jobs: {report.get('bluequbit_jobs', 0)}")
    print(f"Compute used: {orchestrator.compute_used}/{orchestrator.max_compute_budget}")
    print(f"Transition region: {report.get('transition_band', 'unknown')}")
    print(f"Advantage level: {report.get('advantage_level', 'LEVEL_X')}")
    print()
    print(f"Reports: reports/")
    print(f"Figures: results/figures/")
    print(f"Decision log: state/decision_log.jsonl")
    print(f"Agent trace: state/agent_trace.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    main()
