# Quasi-Monte Carlo UQ for Stochastic Dynamics

The program under observation is a deterministic uncertainty quantification demonstrator packaged as a single executable.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Exact internal algorithms are less important than externally stable scientific summaries and return codes.

## Workflow outline

- main.py -- Unified Entry Point
- Quasi-Monte Carlo Uncertainty Quantification for Coupled Seismic-Acoustic-Chemical
- Stochastic Dynamical Systems
- This script executes the complete UQ pipeline:

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

Version output:

```text
synthesis-python-203 1.0
```

## Stable landmarks

```text
PROJECT 203: Quasi-Monte Carlo UQ for Stochastic Dynamics
Coupled Seismic Wave - Ozone Chemistry - Signal Analysis
Scientific problem:
Quantify uncertainty in seismic wave propagation through
heterogeneous random media with coupled atmospheric chemistry,
using polynomial chaos expansion (Gegenbauer basis) and
quasi-Monte Carlo sampling with CVT-adaptive refinement.
Integrating 15 seed projects:
1. 275_dg1d_poisson        -> DG spatial discretization
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.
