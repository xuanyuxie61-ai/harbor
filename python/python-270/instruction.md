# Reverse-Engineer Project 270: Edwards-Anderson Spin Glass Simulation

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `270`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for Edwards-Anderson Spin Glass Simulation is:

```text
synthesis-python-270 1.0
```

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in uncertainty quantification. The binary is meant for observation only; rebuild the behavior in your own files after probing it.

## Observable behavior

Reconstruct the command-line behavior for this uncertainty quantification target:

```text
Edwards-Anderson Spin Glass Simulation
```

Start by matching these lines:

```text
Edwards-Anderson Spin Glass Simulation
High-Order Finite Difference + Stability Analysis
(Small-Scale Reproducible Experiment)
Configuration:
Lattice: 4x4x4 (N=64 spins, z=6)
Boundary: periodic
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 270 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
