# Reconstruction Notes: Project 270

The reference run suggests a small driver that combines sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: disorder_quadrature, langevin_fd_integration
- Diagnostics/reporting: langevin_fd_integration, simulation_engine
- Supporting modules: hessian_magnon_dos, monte_carlo_core, output_writer, replica_exchange, spin_glass_boundary, spin_glass_complex

## Rebuild approach

- Keep the command surface for project 270 narrow: version, help, and the deterministic default workflow.
- Let statistical estimators, quadrature rules, and compact diagnostic reports drive the helper functions, but keep the default run under verifier time limits.
- For project 270, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `Edwards-Anderson Spin Glass Simulation`; hidden tests are not limited to them.

```text
Edwards-Anderson Spin Glass Simulation
High-Order Finite Difference + Stability Analysis
(Small-Scale Reproducible Experiment)
Configuration:
Lattice: 4x4x4 (N=64 spins, z=6)
Boundary: periodic
Coupling: gaussian, J_var=1.0
Disorder realizations: 3
Temperature list: [3.0, 2.0, 1.5, 1.0, 0.7]
MC sweeps: 300 (equil: 100)
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
