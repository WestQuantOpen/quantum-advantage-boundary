# Agent C: Representation Search Agent

## Mission
Search the representation itself — not just parameters, but the formulation.

## Search Space
- MWIS/QUBO Hamiltonian formulation
- Penalty scaling
- Equivalent Hamiltonian representations
- Variable ordering
- Graph relabeling
- Circuit scheduling
- Compilation representation
- Layout/routing choices
- Hardware-aware representations

## Key Question
Can AI identify representations particularly favorable to quantum sampling rather than merely favorable to all solvers?

## Method
- Maintain Pareto fronts rather than a single scalar winner
- Each candidate has: representation_id, parent_id, transformation, mathematical-equivalence status
- Reuse WestQuant representation-search concepts where practical

## Critical Test
The AI x QUANTUM interaction. Compare:
- Standard representation + quantum sampler
- AI-selected representation + quantum sampler
- Same representation + classical sampler

If the representation helps both equally, it's not quantum-specific.
