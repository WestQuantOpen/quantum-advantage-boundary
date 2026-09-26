# Reproducibility Report

## Verification

- All seeds recorded: YES
- All configurations serialized: YES
- All agent decisions logged: YES (10 entries)
- Package versions: see requirements.txt
- Raw data retained: YES (results/raw/)
- Held-out test sets: NO

## Reproducibility Status

All experiments use deterministic seeds via numpy.random.RandomState.
Graph instances are fully reproducible from (n, density, seed, graph_family).
Exact solver uses scipy.optimize.milp (deterministic).
Samplers use RandomState with explicit seeds.
QAOA parameters are generated from the same RandomState.

## Agent Trace

10 agent decisions logged in state/agent_trace.jsonl
12 orchestration decisions logged in state/decision_log.jsonl
