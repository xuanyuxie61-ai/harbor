# Reconstruction Notes: Project 253

The binary behaves like a scripted experiment focused on matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: high_order_fd, regge_wheeler_solver, spectral_quadrature
- Model layer: physics_constants
- Diagnostics/reporting: domain_decomposition, qnm_companion, stability_analyzer
- Supporting modules: adaptive_topology, boundary_engine, monte_carlo_explorer, spacetime_geometry, waveform_template

## Coding plan

- Use a tiny dispatch layer for — Gravitational-wave template computation; hidden tests should not depend on accidental framework formatting.
- If a real spectral physics solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around — Gravitational-wave template computation; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `— Gravitational-wave template computation`; hidden tests are not limited to them.

```text
PROJECT 253 — Gravitational-wave template computation
via high-order finite differences and stability analysis
(small-scale reproducible experiment)
1. Binary black-hole parameters
chirp mass   M_c   = 8.7055 Msun
total mass   M     = 20.0000 Msun
sym. ratio   nu    = 0.2500
distance     D_L   = 100.0 Mpc
parameter set valid: True
2. Schwarzschild background and Regge-Wheeler potential
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
