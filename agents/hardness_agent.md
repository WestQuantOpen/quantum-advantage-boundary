# Agent V2-4: Hardness Agent

## Mission
Generate graphs with controlled hardness instead of scaling only N.

## Hardness Score
H(G) = f(T_MILP, T_PT, P_local-hit, basin_structure, degeneracy)

## Adversarial Instance Generation
Generate instances with:
- Many near-degenerate solutions
- Large Hamming distances between good solutions
- Planted optimum
- Deceptive local minima
- High local escape costs
- Frustration
- Multiple competing basins

## Hardness-Agent Reward
R_G = T_classical-hit - alpha * T_quantum-hit
(without quantum agent influencing the graph directly)

## Key Insight
An N=24 problem with multiple well-separated local minima can be much
more informative than an N=40 problem where greedy finds optimum immediately.
