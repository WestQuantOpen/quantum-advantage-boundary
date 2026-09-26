# Agent E: Compute Scheduler

## Mission
Use BlueQubit compute efficiently.

## Device Selection
- Local computation: for classical samplers and small quantum circuits (n <= 16)
- BlueQubit CPU: n <= 20, exact statevector
- BlueQubit GPU: n <= 26, faster statevector
- BlueQubit MPS (cpu/gpu): n > 26, approximate
- BlueQubit Pauli-path: for expectation value verification
- QPU: only when RUN_QPU=YES (not in prototype)

## Budget Management
- Use BlueQubit estimates before expensive jobs
- Track all costs in state/compute_ledger.csv
- Respect BQ_MAX_COST_USD (default: $10.00 for prototype)
- Prefer asynchronous execution for independent jobs
- Reject expensive experiments with low expected information gain

## Job Tagging
Every BlueQubit job must be tagged with:
- project: "wq-boundary"
- phase: "round_1" etc.
- agent: requesting agent name
- experiment_id
- n: qubit count
- seed
- sampler
- representation_id

## Async Strategy
- Batch independent circuits
- Use asynchronous=True for batches
- Wait for all results before analysis
