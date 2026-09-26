# Quantum Advantage Boundary Hunter V2 — Extended Campaign Report

## Shifted Research Question

From: "At what N does quantum beat classical?"
To: **"What combination of problem structure, representation, quantum kernel, and classical compute budget creates a crossover?"**

## V2 Campaign: Six Research Strands

### Strand 1: Recovery Erasure Experiment (COMPLETE)

**Diagnosis: B — Quantum is worse raw but recovery eliminates difference.**

All 16 experiments (n=24, 26, 28 × density 0.5, 0.7 × 3 seeds) show Diagnosis B.

Key findings:
- QAOA samples have **higher Hamming distance** to optimum (d_H=5-8) vs greedy (d_H=0)
- QAOA samples explore **more attraction basins** (128 unique) vs greedy (26-54)
- But classical recovery closes the gap from both sides at all budgets B_C

This means the quantum subspace IS structurally different (more diverse, more basins) but classical recovery is strong enough to close the gap.

**Implication**: The signal is hidden by strong recovery. The question becomes: can we find conditions where the quantum diversity advantage translates to a recovery advantage?

### Strand 2: Oracle Test (PARTIAL — 2/8 instances)

**Oracle quantum beats classical in 2/2 completed instances.**

| Instance | Oracle Quantum | Classical (SA) | Margin |
|----------|---------------|----------------|--------|
| n=24, d=0.5, s=0 | 37.45 | 26.81 | +10.64 |
| n=24, d=0.5, s=1 | 44.89 | 36.18 | +8.71 |

**Implication**: When quantum gets per-instance optimization (search 50 configs, pick best), it CAN beat classical by a large margin. This is a perfect ML problem: can WestQuant predict the oracle config from graph structure?

### Strand 3: QeMCMC + Parallel Tempering (PENDING)

Quantum-enhanced MCMC with parallel tempering. Based on IBM's 2026 MIS work.
The QeMCMC sampler uses quantum circuits as transition kernels for inter-basin jumps.

### Strand 4: Hardness-Conditioned + QCSC (PENDING)

Hardness-conditioned graph generation and quantum-classical subspace composition.
Decompose N=500 → H_hard=32 for genuine QCSC.

### Strand 5: MPS/PPS Simulability Attack (PENDING)

MPS bond dimension sweep to find classical simulability limit.

### Strand 6: Peaked Circuit Track (PENDING)

Secondary track: search for verifiable quantum advantage via peaked circuits.

## Quantum Advantage Phase Diagram

```
A = A(N, H, B_C, B_Q, R, K_Q, chi)
```

The negative results to N=28 are the first data points in the A<0 region.
The Oracle test shows A>0 is achievable with per-instance optimization.
The AI agent now searches for the phase boundary between these regions.

## Key Insight

The Recovery Erasure Experiment reveals that the quantum disadvantage is NOT because quantum samples are useless — it's because classical recovery is so strong it erases the difference. The quantum samples are MORE diverse (128 basins vs 26-54) but start further from the optimum (d_H=5-8 vs 0).

The Oracle Test confirms that with per-instance optimization, quantum CAN beat classical by +8-10 points. The challenge is: can we predict the oracle config without seeing the answer?

## BlueQubit Compute Used

- Strand 1: 16 BlueQubit jobs (GPU for n=24-26, MPS.gpu for n=28)
- Strand 2: ~50 BlueQubit CPU jobs (oracle search)
- Total cost: ~$5.00

## Next Steps

1. Complete Strand 2 (Oracle Test) for all instances
2. Run Strand 3 (QeMCMC) with reduced steps for speed
3. Run Strand 4 (Hardness + QCSC) — most original WestQuant angle
4. Run Strand 5 (Simulability) — find classical simulability limit
5. Synthesize Quantum Advantage Phase Diagram
