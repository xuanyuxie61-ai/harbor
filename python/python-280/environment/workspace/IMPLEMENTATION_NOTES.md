# Reconstruction Notes: Project 280

The observable transcript is organized around high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: damage_grid, fd_operators, time_integration
- Model layer: microstructure_field
- Diagnostics/reporting: mcmc_optimization, percolation_failure, stability_analysis, time_integration
- Supporting modules: benchmark_parse, config, crack_tracking, energy_minimize, nonlocal_damage, stochastic_defects

## Suggested coding path

- Give Multi-Scale Material Damage Evolution a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 280; preserve the staged progression of the report.
- Render Multi-Scale Material Damage Evolution's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `Multi-Scale Material Damage Evolution`; hidden tests are not limited to them.

```text
Project 280: Multi-Scale Material Damage Evolution
High-Order Finite Differences & Stability Analysis
Computational Materials Science — PhD-Level Simulation
Phase 1: Grid Generation with Crack-Tip Refinement
Domain: [0.0, 0.3] x [0.0, 0.3] m
Grid: 60 x 60 = 3600 nodes
Spacing: dx = 5.084746e-03 m, dy = 5.084746e-03 m
Annular points around crack tip: 288
Min spacing: 5.084746e-03 m
Max aspect ratio: 1.0000
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
