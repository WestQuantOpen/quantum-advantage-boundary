# Agent V2-2: Oracle Quantum Searcher

## Mission
Give quantum an unfair advantage to test if ANY quantum configuration can beat classical.

## Method
For each graph, search 10^3-10^4 quantum configurations:
(gamma, beta, p, penalty, mixer, representation, ordering)

Select the BEST per-instance quantum config in hindsight.

## Key Question
max_theta U_Q(G, theta) > U_best_classical(G)?

## Interpretation
- **NO**: Stop training WestQuant on this family. No generalizable policy exists.
- **YES but general loses**: Perfect ML problem — can WestQuant predict the oracle config from graph structure?
- **YES and general wins**: Real quantum advantage.
