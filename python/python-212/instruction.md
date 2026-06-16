# Reverse-Engineer Project 212: KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in

## Task target

You are in `/app/workspace` with a reference executable for project `212`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in is:

```text
synthesis-python-212 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference uncertainty quantification executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## External contract

Use the following label as the center of the reconstruction:

```text
KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in
```

Reference lines worth preserving for project 212:

```text
KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in
PDE-Constrained Optimal Control with Mixed Constraints
Scientific domain : Mathematical Optimisation / KKT Conditions
PDE constraint    : Bloch-Torrey-type reaction-diffusion
Inequality constr : Box bounds + integral energy budget
Solver            : Semismooth Newton / primal-dual active set
```

For project 212, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 212, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
