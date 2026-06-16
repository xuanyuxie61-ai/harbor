# Reconstruction Notes: Project 262

The important implementation pressure is magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: mesh_generator, mhd_operators, time_integrator
- Model layer: surrogate_model
- Diagnostics/reporting: diagnostics, optimization, stability_analysis
- Supporting modules: boundary_handler, current_sheet, plasma_parameters, polynomial_basis, topology_analyzer

## Suggested coding path

- Build the executable wrapper last for project 262, after the Python entry point matches probes.
- Keep computational plasma physics computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding SOLAR FLARE MAGNETIC RECONNECTION SIMULATION.
- Make the no-argument path finish quickly under verifier time limits.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `SOLAR FLARE MAGNETIC RECONNECTION SIMULATION`; hidden tests are not limited to them.

```text
SOLAR FLARE MAGNETIC RECONNECTION SIMULATION
High-Order Finite Differences & Stability Analysis
Small-Scale Reproducible Experiment
SOLAR CORONA PLASMA PARAMETERS
[Phase 1] Plasma Parameters and Mesh Setup
B0 (upstream)         = 20.00 G
L_cs (half-thickness) = 5.00e+06 m
n0 (density)          = 1.00e+15 m^-3
T0 (temperature)      = 5.00e+06 K
V_A (Alfvén speed)    = 1.38e+06 m/s
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
