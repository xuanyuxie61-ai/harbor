# Reconstruction Notes: Project 212

Expect hidden checks to care most about the externally visible parts of sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: matrix_kernels, nelder_mead_subsolver, pde_operator
- Model layer: physics_models
- Diagnostics/reporting: cost_functional, variational_assimilation
- Supporting modules: active_set, bayesian_precond, chebyshev_accelerator, integral_constraints, kkt_system, line_search

## Practical reconstruction

- For KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the uncertainty quantification reconstruction clearer and faster.
- If KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `KKT-VPDE: Primal-Dual Active-Set Solver for KKT Systems in`; hidden tests are not limited to them.

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
[SETUP] Energy budget: E_max=1.5
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
