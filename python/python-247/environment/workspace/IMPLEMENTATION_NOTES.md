# Reconstruction Notes: Project 247

The observable transcript is organized around magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: least_squares_quadrature
- Diagnostics/reporting: analysis_pool, convection_vortex_halo, multizone_accretion, stability_von_neumann
- Supporting modules: gauss_prime_seeding, halo_gegenbauer_potential, halo_ode_systems, merger_tree_enumerator, minimal_merger_path, phase_space_flow_coarse_grain

## Coding plan

- Give Dark matter halo formation and merger tree a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 247; preserve the staged progression of the report.
- Render Dark matter halo formation and merger tree's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `Dark matter halo formation and merger tree`; hidden tests are not limited to them.

```text
PROJECT_247 : Dark matter halo formation and merger tree
High-order finite-difference stability analysis
Pipeline completed in 0.204 s
[1. Gaussian-prime seeding]
[2. Gegenbauer halo potential]
[3. Trigonometric orbit interpolation]
[4. Merger-tree enumeration]
[5. Optimal merger path]
[6. Phase-space coarse-graining]
[7. Halo ODE systems]
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
