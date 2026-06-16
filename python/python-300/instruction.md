# Reverse-Engineer Project 300: Tritium Breeding Blanket Neutron Transport Solver

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `300`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for Tritium Breeding Blanket Neutron Transport Solver is:

```text
synthesis-python-300 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference fusion and radiation transport executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## Observable behavior

Reconstruct the command-line behavior for this fusion and radiation transport target:

```text
Tritium Breeding Blanket Neutron Transport Solver
```

Start by matching these lines:

```text
#  Tritium Breeding Blanket Neutron Transport Solver
#  High-order compact finite-difference SN method
#  with von Neumann / spectral stability analysis
#  and Feynman-Kac stochastic verification
Tritium Breeding Blanket Neutron Transport Solver
High-order compact finite-difference SN method
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 300 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
