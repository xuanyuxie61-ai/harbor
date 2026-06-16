# Reconstruction Notes: Project 263

This task can be solved by rebuilding a concise pipeline for high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: coronal_grid, coronal_solver, fd_operators
- Diagnostics/reporting: stability_analysis
- Supporting modules: coronal_basis, coronal_energy, heom_hierarchy, nanoflare_oscillator, parker_wind, solar_constants

## Implementation strategy

- For this numerical-methods benchmark task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 263's printed diagnostics.
- Keep the final report for project 263 deterministic, compact, and ordered like the reference.
- For Coronal Heating & Solar Wind Acceleration, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Public report cues

Use these public lines as reconstruction checkpoints for `Coronal Heating & Solar Wind Acceleration`; hidden tests are not limited to them.

```text
PROJECT 263 - Coronal Heating & Solar Wind Acceleration
High-Order Finite Differences with Stability Analysis
Python 3.11.11, NumPy 1.26.4
Random seed: 263
Corona loop grid (nodes=64)
n_nodes                 : 64
h_min                   :   1.391037e+06
h_max                   :   4.677348e+06
h_mean                  :   3.145125e+06
ratio                   :   3.362491e+00
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
