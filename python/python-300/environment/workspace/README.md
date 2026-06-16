# Tritium Breeding Blanket Neutron Transport Solver

The task is to rebuild a small Python implementation that behaves like a reference fusion and radiation transport executable.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Observed behavior

- main.py - Tritium Breeding Blanket Neutron Transport Solver
- Unified entry point for the high-order finite-difference discrete-ordinates
- (S_N) solver of the multigroup neutron transport equation in a fusion
- breeding blanket.  Zero-argument execution.

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Identity check:

```text
synthesis-python-300 1.0
```

## Transcript anchors

```text
#  Tritium Breeding Blanket Neutron Transport Solver
#  High-order compact finite-difference SN method
#  with von Neumann / spectral stability analysis
#  and Feynman-Kac stochastic verification
Tritium Breeding Blanket Neutron Transport Solver
High-order compact finite-difference SN method
with von Neumann / spectral stability analysis
and Feynman-Kac stochastic verification
1. Blanket geometry (1-D Voronoi pebble bed)
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
