# QPU Validation Plan

## Overview

This document specifies the IBM QPU validation campaign for the strongest
circuits identified by the Quantum Advantage Boundary Hunter.

## Selected Conditions

The following 5-20 frozen circuit conditions deserve scarce QPU time:

1. **n=24, density=0.5, p=2, penalty=2x** — Transition band center
2. **n=26, density=0.5, p=2, penalty=2x** — Upper transition band
3. **n=24, density=0.6, p=3, penalty=4x** — Harder instance, deeper circuit
4. **n=22, density=0.4, p=1, penalty=1x** — Easier instance, shallow circuit
5. **n=26, density=0.4, p=2, penalty=2x** — Sparse, medium depth

## Per-Condition Specification

For each condition:
- Problem: MWIS on Erdos-Renyi graph
- Shots: 4096 (QPU noise requires more shots)
- Repetitions: 5 independent runs
- Classical comparison: SA with matched budget
- Success criterion: A_Q > 0 with 95% CI excluding zero

## Total QPU Shots

Estimated: 5 conditions × 4096 shots × 5 repetitions = 102,400 shots

## Calibration Drift

Design includes:
- Multiple calibration windows
- Independent repeats across different days
- Mapping variation studies
