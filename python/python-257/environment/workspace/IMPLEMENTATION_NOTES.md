# Reconstruction Notes: Project 257

The source evidence points to reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: fd_coefficients, quadrature_validator, sparse_operator, spherical_mesh
- Model layer: foreground_model
- Diagnostics/reporting: stability_analysis
- Supporting modules: beam_simulator, cosmology_params, gradient_contour, mode_classifier, monte_carlo_spectrum, parameter_optimizer

## Implementation strategy

- Make the flag paths cheap for project 257; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by CMB POWER SPECTRUM ESTIMATION PIPELINE; random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 257.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Public report cues

Use these public lines as reconstruction checkpoints for `CMB POWER SPECTRUM ESTIMATION PIPELINE`; hidden tests are not limited to them.

```text
#                                                                      #
#   CMB POWER SPECTRUM ESTIMATION PIPELINE                            #
#   High-Order Finite Differences & Stability Analysis                #
#   (Small-Scale Reproducible Experiment)                             #
CMB POWER SPECTRUM ESTIMATION PIPELINE
High-Order Finite Differences & Stability Analysis
(Small-Scale Reproducible Experiment)
Preliminary:  Planck 2018 derived parameters
theta_*    = 3.683033e-23
C_l reference:
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
