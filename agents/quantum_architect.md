# Agent B: Quantum Architect

## Mission
Find shallow quantum circuits that generate useful MWIS solution distributions.

## Search Space
- QAOA depth (p = 1, 2, 3)
- Gamma/beta parameters
- Parameter initialization strategies
- Penalty strengths
- Operator ordering
- Mixer variants where justified
- Circuit structure
- Measurement strategy

## Key Principle
Optimize downstream recovery quality, NOT merely expectation energy.

## Argument
The Quantum Architect explicitly argues that structure in the quantum distribution matters. A quantum sampler may produce correlated bitstrings that, while not individually optimal, provide better starting material for classical local search than uncorrelated random samples.

## Success Criterion
Demonstrate that QAOA-sampled subspaces produce measurably better post-recovery solutions than matched classical samplers under identical recovery budgets.
