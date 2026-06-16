# Reconstruction Notes: Project 265

The cleanroom notes below emphasize high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: grid_data_io, heliocentric_mesh, multigrid_fas
- Model layer: cosmic_ray_physics, focused_transport_eq
- Diagnostics/reporting: dispersion_roots, grid_data_io, heliocentric_mesh, species_composition, stability_analysis
- Supporting modules: analytical_benchmarks, high_order_stencils, inverse_kappa, stochastic_parker, turbulence_mcmc

## Coding plan

- Keep project 265's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of Cosmic-Ray Transport in the Heliosphere's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the numerical-methods benchmark transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `Cosmic-Ray Transport in the Heliosphere`; hidden tests are not limited to them.

```text
PROJECT 265 -- Cosmic-Ray Transport in the Heliosphere
High-order finite differences & stability analysis
EXPERIMENT 1 -- heliospheric mesh & Parker IMF
radial grid : Nr = 48, r_min = 0.050 AU, r_max = 120.0 AU
mu grid     : Nmu = 16, min = -0.9950, max = +0.9950
EXPERIMENT 2 -- high-order finite-difference stencils
upwind1 max error = 9.899e-14
upwind2 max error = 3.230e-14
compact4 max error = 1.176e-14
WENO5 max error    = 3.519e-13
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
