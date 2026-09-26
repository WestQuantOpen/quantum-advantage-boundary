# V2 Research Strategy: Quantum Advantage Phase Diagram

## Shifted Research Question

From: "At what N does quantum beat classical?"
To: **"What combination of problem structure, representation, quantum kernel, and classical compute budget creates a crossover?"**

## Phase Diagram

```
A = A(N, H, B_C, B_Q, R, K_Q, chi)
```

Where:
- N = problem size
- H = empirical hardness
- B_C = classical recovery budget
- B_Q = quantum sampling budget
- R = representation
- K_Q = quantum kernel (QAOA / QeMCMC / etc.)
- chi = classical MPS simulability capacity

The negative results to N=28 are the first data points in the A<0 region.
The AI agent uses these to actively search for the phase boundary.

## Six Research Strands (Priority Order)

### Strand 1: Recovery Erasure Experiment (10% GPU)
Diagnose WHY results are negative. Run same samples through recovery with
budgets B_C = 0, 100, 500, 2000, 10000, 50000.

Three diagnoses:
- A: Quantum is worse raw → change quantum kernel
- B: Quantum is better raw but recovery eliminates difference → hidden advantage
- C: Difference appears only at bounded recovery budget → QCSC result

### Strand 2: Oracle Test (15% GPU)
Give quantum an unfair advantage: search 10^3-10^4 configs per instance,
select best in hindsight. If oracle-QAOA still loses, stop training WestQuant
on this family. If oracle wins but general loses → ML problem.

### Strand 3: QeMCMC + Parallel Tempering (25% GPU)
Replace standard QAOA with quantum-enhanced MCMC. Quantum kernel makes
inter-basin jumps. Based on IBM's 2026 MIS work.

### Strand 4: Hardness-Conditioned + QCSC (25% GPU)
Generate hardness-conditioned graphs. Decompose N=500 → H_hard=32.
Classical preprocessing fixes easy variables, quantum solves hard core.

### Strand 5: MPS/PPS Simulability Attack (15% GPU)
MPS bond dimension sweep. Pauli-path red-team.
Find classical simulability limit.

### Strand 6: Peaked Circuit Track (10% GPU)
Secondary track: search for verifiable quantum advantage via peaked circuits.
