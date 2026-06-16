# KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in

The task is to rebuild a small Python implementation that behaves like a reference uncertainty quantification executable.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Scientific role

- Unified entry point for the KKT-VPDE solver.
- This script solves the following PDE-constrained optimal control problem:
- min  J(y, u) = 0.5 ||y - y_d||_M^2 + 0.5 alpha ||u||_M^2 + 0.5 (u - u_b)^T B^{-1} (u - u_b)
- s.t. -nu Laplacian(y) + R(y; T1, T2) = u + f    in Omega = [0,1]^2

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Stable version text:

```text
synthesis-python-212 1.0
```

## Report signatures

```text
KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in
PDE-Constrained Optimal Control with Mixed Constraints
Scientific domain : Mathematical Optimisation / KKT Conditions
PDE constraint    : Bloch-Torrey-type reaction-diffusion
Inequality constr : Box bounds + integral energy budget
Solver            : Semismooth Newton / primal-dual active set
[SETUP] Grid: N=10, n2=100, h=0.0909
[SETUP] Physical: nu=0.1, T1=1.0, T2=0.3
[SETUP] Control bounds: [0.0, 3.5]
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
