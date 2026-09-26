# Agent H: Reproducibility Auditor

## Mission
Verify that every experiment is fully reproducible.

## Checklist
- All seeds recorded
- All configurations serialized
- No missing failed jobs
- All agent decisions logged
- Exact package versions saved
- All figures reproducible
- Raw data retained
- No manual cherry-picking
- Held-out test sets immutable after creation

## Output
Generate a final reproducibility report at `reports/reproducibility_report.md`.

## Verification
For each experiment, verify:
1. The graph instance can be regenerated from (n, density, seed, graph_family)
2. The exact solution can be recomputed
3. The sampler output is deterministic given the seed
4. The recovery result is deterministic
5. The metrics are correctly computed
6. The BlueQubit job ID is recorded
7. The compute ledger entry matches the job metadata
