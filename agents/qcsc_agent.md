# Agent V2-6: QCSC Decomposer

## Mission
Solve 500-variable problems using only 32-qubit quantum circuits.

## Pipeline
```
G_500 → classical decomposition → H_32 → quantum sampler → classical reconstruction → x*
```

## Method
1. Classical preprocessing (LP relaxation, greedy) fixes easy variables
2. AI identifies uncertain hard core
3. Quantum works only on H_hard (24-40 qubits)
4. Result recombined with fixed variables

## Why This Works
- The problem the system solves is 500 variables
- But the QPU only uses 32 qubits
- This is genuine Quantum-Classical Subspace Composition (QCSC)
- Fits BlueQubit GPU perfectly

## Key Question
Can QCSC find solutions that pure classical decomposition misses?
