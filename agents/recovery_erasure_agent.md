# Agent V2-1: Recovery Erasure Diagnostician

## Mission
Diagnose WHY the quantum advantage is negative. Is the signal hidden by strong classical recovery?

## Method
For the same saved quantum and classical samples, run downstream recovery with budgets:
B_C = 0, 100, 500, 2000, 10000, 50000

## Pre-Recovery Diagnostics
- d_H(S, X*) = min Hamming distance from samples to optimum
- Best raw objective
- Low-energy tail mass
- P(d_H <= k) for k=1,2,3
- Number of distinct attraction basins
- Entropy/diversity

## Three Diagnoses
- **A**: Quantum is worse raw → no amount of N scaling helps. Change quantum kernel.
- **B**: Quantum is better raw but recovery eliminates difference → subspace-info advantage hidden.
- **C**: Difference appears only at bounded recovery budget → QCSC compute-to-target result.
