# Agent V2-3: QeMCMC Designer

## Mission
Replace standard QAOA with quantum-enhanced MCMC for MWIS.

## Rationale
Standard QAOA tries to create a good final distribution.
Our actual need is: move between hard-to-reach solution basins.
This is much closer to sampling/MCMC.

Based on IBM's 2026 work on QeMCMC + parallel tempering for MIS.

## Method
Parallel Tempering chain at temperatures T_1 < T_2 < ... < T_K.
At each step:
1. Classical 1-flip proposal (local)
2. Quantum proposal: QAOA-like circuit proposes a jump
3. Metropolis accept/reject
4. Replica exchange between temperatures

## Metrics
- T_hit = time to reach optimal basin
- tau_mix = mixing time (autocorrelation)
- Round-trip time between temperatures
- P(reach low-energy basin within budget B)
