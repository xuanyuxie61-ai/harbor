# Nuclear Fission Simulation

The task is to rebuild a small Python implementation that behaves like a reference fusion and radiation transport executable.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Workflow outline

- Unified entry point for the nuclear fission simulation project.
- PROJECT_245: Nuclear Fission Simulation - Fragment Mass Distribution
- and Energy Release Modelling with High-Order Finite Differences
- and Stability Analysis (Small-Scale Reproducible Experiments)

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Version output:

```text
synthesis-python-245 1.0
```

## Stable landmarks

```text
PROJECT_245: Nuclear Fission Simulation
Fragment Mass Distribution & Energy Release Modelling
High-Order Finite Differences & Stability Analysis
Binding energy per nucleon B/A (Liquid Drop Model):
1. Nuclear Constants and Binding Energy [nuclear_constants.py]
Nucleus        A     Z   B(A,Z) [MeV]    B/A [MeV]
He-4           4     2        35.7595       8.9399
Fe-56         56    26       495.5984       8.8500
Sn-132       132    50      1081.2751       8.1915
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
