# Quantum Advantage Boundary Hunter

**AI-orchestrated multi-agent quantum research campaign using BlueQubit compute.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![BlueQubit](https://img.shields.io/badge/BlueQubit-powered-blueviolet)](https://bluequbit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Research Question

**V1**: When does the quality of a quantum-selected subspace begin to matter?

**V2 (current)**: What combination of problem structure, representation, quantum kernel, and classical compute budget creates a crossover?

```
A = A(N, H, B_C, B_Q, R, K_Q, chi)
```

Where N = problem size, H = hardness, B_C = classical recovery budget, B_Q = quantum sampling budget, R = representation, K_Q = quantum kernel, chi = classical simulability capacity.

The negative v1 results (N≤28) are the first data points in the A<0 region. The v2 campaign diagnosed WHY (recovery erases the signal), found the hidden signal (oracle QAOA beats classical by +8-10), and identified the path to a positive result.

---

## The Quantum Flywheel

```
Hypothesis → Agents debate → BlueQubit compute → Evidence → Agents disagree → new experiment → falsification → boundary
```

This is a working prototype of a **Quantum Flywheel** — an AI agent that autonomously:
1. Designs experiments
2. Allocates compute adaptively
3. Searches quantum representations
4. Attempts to falsify its own results
5. Replicates findings

---

## Campaign Results

### V1: Coarse Boundary Scan

| Metric | Value |
|--------|-------|
| Experiments orchestrated | 26 |
| BlueQubit quantum jobs | 14 |
| Backends used | CPU, GPU |
| Research rounds | 5 |
| Agent councils | 3 |
| Transition band | n=18-24, density=0.5-0.7 |
| Advantage level | LEVEL 0 (no quantum advantage) |

### V2: Six-Strand Extended Campaign

| Strand | Experiments | Key Finding |
|--------|-------------|-------------|
| 1: Recovery Erasure | 16 | Diagnosis B: quantum diverse but recovery erases signal |
| 2: Oracle Test | 2 | Oracle QAOA beats classical by +8-10 points |
| 3: QeMCMC | 3 | Matches PT at n=20 (too easy) |
| 4: Hardness + QCSC | 20 | Deceptive instances have 17% SA gap; QCSC core=0 |
| 5: Simulability | 3 | Shallow QAOA (p=1) is classically easy |
| 6: Peaked Circuits | 5 | Peak strengths 2-3x mean probability |

**Phase Diagram**: 123 data points. Boundary at N=(28, 100).
- A > 0: 18 points (oracle + B_C=0)
- A = 0: 88 points (recovery erasure)
- A < 0: 17 points (hardness + QCSC)

### Scientific Conclusion

**V1**: No quantum advantage detected. Classical recovery dominates through N=26.

**V2**: The negative result was diagnosed — quantum IS more diverse (128 basins vs 26-54) but recovery erases the advantage. Per-instance optimized QAOA (oracle) CAN beat classical by +8-10 points. The challenge is predicting the oracle config from graph structure — a perfect ML problem for WestQuant.

---

## Multi-Agent Architecture

Eight agents with distinct and adversarial roles:

| Agent | Role | Mission |
|-------|------|---------|
| A | Principal Investigator | Maximize information gain per compute |
| B | Quantum Architect | Find useful quantum circuits |
| C | Representation Search | Search the formulation itself |
| D | Classical Adversary | Destroy quantum advantage claims |
| E | Compute Scheduler | Efficient BlueQubit allocation |
| F | Statistician | Prevent noise from becoming a result |
| G | Skeptical Reviewer | Hostile conference reviewer |
| H | Reproducibility Auditor | Verify full reproducibility |

---

## Campaign Rounds

| Round | Phase | Description |
|-------|-------|-------------|
| 0 | Validation | Verify pipeline on n=6-10 |
| 1 | Coarse scan | Broad sweep n=18-26, identify transition band |
| 2 | Adaptive | Focus compute on transition band |
| 3 | Representation | Search penalty coefficients, QAOA depths |
| 4 | Red team | Adversarial classical validation |
| 5 | Held-out | Confirmation with new instances |

---

## Problem: MWIS

**Maximum Weighted Independent Set** on Erdos-Renyi graphs.

- Graph sizes: n = {18, 20, 22, 24, 26}
- Densities: {0.3, 0.5, 0.7}
- Exact solutions via scipy.optimize.milp (MILP)

---

## Samplers

| Sampler | Type | Description |
|---------|------|-------------|
| Uniform random | Classical | Null hypothesis baseline |
| Greedy | Classical | Randomized greedy with restarts |
| Simulated annealing | Classical | Temperature schedule, 1-flip moves |
| Parallel tempering | Classical | Multi-temperature replica exchange |
| QAOA p=1,2,3 | Quantum | Shallow QAOA with random parameters |

All samplers feed the **same** classical recovery: `S → repair → 1-flip → 2-flip → restricted solve`

---

## BlueQubit Integration

- **CPU simulation**: n=18-20 (free)
- **GPU simulation**: n=22-26 ($0.20/job)
- **MPS**: n>26 (available but slow for prototype)
- **QPU**: Deferred (see `reports/qpu_validation_plan.md`)

```python
import bluequbit
bq = bluequbit.init()  # reads BLUEQUBIT_API_TOKEN
result = bq.run(circuit, device='gpu', shots=256)
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set BlueQubit token (never commit this)
export BLUEQUBIT_API_TOKEN="your_token"

# Run campaign
python -m src.main --budget 25 --max-cost-usd 5.0

# Run without BlueQubit (local simulation only)
python -m src.main --budget 25 --no-bluequbit
```

---

## Outputs

| File | Description |
|------|-------------|
| `reports/PROTOTYPE_REPORT.md` | Full scientific report |
| `reports/bluequbit_application_evidence.md` | Application evidence |
| `reports/agent_council_round_*.md` | Agent debate transcripts |
| `reports/qpu_validation_plan.md` | Future QPU experiment plan |
| `results/figures/*.png` | Publication-quality figures |
| `results/processed/summary.csv` | Aggregated metrics |
| `state/decision_log.jsonl` | Agent decision history |
| `state/agent_trace.jsonl` | Agent reasoning trace |
| `state/compute_ledger.csv` | BlueQubit cost tracking |

---

## Figures

| Figure | Description |
|--------|-------------|
| `boundary_scan.png` | Post-recovery performance vs problem size |
| `subspace_advantage.png` | Quantum minus classical performance |
| `recovery_phase_diagram.png` | Optimality gap vs problem size |
| `compute_allocation.png` | Compute reallocation across rounds |
| `falsification_waterfall.png` | Apparent advantage under classical attacks |

---

## Pre-Registered Hypotheses

| Level | Description |
|-------|-------------|
| LEVEL 0 | No measurable quantum contribution |
| LEVEL 1 | Quantum subspace advantage (matched budgets) |
| LEVEL 2 | End-to-end hybrid compute advantage |
| LEVEL 3 | Survives adversarial challenges + held-out replication |
| LEVEL X | Insufficient evidence |

**Result: LEVEL 0** — No quantum advantage detected.

---

## Reproducibility

- All seeds recorded via `numpy.random.RandomState`
- All configurations serialized in JSON
- All agent decisions logged in `state/agent_trace.jsonl`
- All BlueQubit jobs tagged with experiment provenance
- Exact package versions in `requirements.txt`

---

## Security

- BlueQubit API token read from `BLUEQUBIT_API_TOKEN` environment variable
- Token never printed, logged, or committed
- `.gitignore` excludes `.env`, credentials, and raw data
- No QPU jobs submitted unless `RUN_QPU=YES`

---

## License

MIT

---

## Author

**David Vesterlund** — ORCID: 0009-0000-6455-1141
Vesterlund Ventures / WestQuant Open, Stockholm
