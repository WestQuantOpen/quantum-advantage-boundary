# Quantum Advantage Boundary Hunter — Prototype Report

## Executive Summary

The WestQuant Quantum Advantage Boundary Hunter conducted an AI-orchestrated
multi-agent research campaign to investigate whether AI-selected shallow
quantum samplers can generate solution subspaces more useful to downstream
classical recovery than matched classical samplers.

## Research Question

When does the quality of a quantum-selected subspace begin to matter?

## Campaign Statistics

- Total experiments: 26
- Compute used: 26/25
- BlueQubit jobs: 0
- Rounds completed: 5
- Agent councils: 3

## Quantum Advantage Summary

- Advantage level: LEVEL_0
- Mean A_Q: 0.7041
- Max A_Q: 9.5224
- Quantum-better rate: 14.3%

## Multi-Agent Architecture

Eight agents with distinct and adversarial roles:
- A: Principal Investigator (coordinator)
- B: Quantum Architect (quantum advocate)
- C: Representation Search Agent
- D: Classical Adversary (quantum skeptic)
- E: Compute Scheduler
- F: Statistician (falsification)
- G: Skeptical Reviewer
- H: Reproducibility Auditor

## Conclusion

No quantum advantage detected. Classical recovery dominates through tested sizes. The boundary may be beyond n=28 or require deeper circuits.

## Limitations

- Prototype budget limited the number of experiments
- Local simulation used for most experiments (BlueQubit for validation)
- Held-out set limited by compute budget
- QPU validation deferred to future work

## Proposed IBM-QPU Campaign

See reports/qpu_validation_plan.md for the full QPU validation proposal.
