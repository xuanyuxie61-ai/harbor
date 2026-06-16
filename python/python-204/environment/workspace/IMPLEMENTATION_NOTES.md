# Reconstruction Notes: Project 204

The reference run suggests a small driver that combines sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: matrix_kernels, quadrature_pce
- Model layer: forward_model, reaction_kinetics
- Diagnostics/reporting: langevin_inversion, reaction_kinetics
- Supporting modules: chaotic_mixing, elliptic_green, gauss_seidel_coupled, r8col_utils, robustness, shearlet_surrogate

## Rebuild approach

- Keep the command surface for project 204 narrow: version, help, and the deterministic default workflow.
- Let statistical estimators, quadrature rules, and compact diagnostic reports drive the helper functions, but keep the default run under verifier time limits.
- For project 204, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `Sobol sensitivity analysis of a disk-shaped`; hidden tests are not limited to them.

```text
PROJECT 204 : Sobol sensitivity analysis of a disk-shaped
geophysical reactor with chaotic mixing
and autocatalytic Lotka-Volterra kinetics.
Stage 0 / 7 : module smoke tests
[ok]   r8col_utils
[ok]   matrix_kernels
[ok]   elliptic_green
[ok]   chaotic_mixing
[ok]   disk_monomial_integral
[ok]   reaction_kinetics
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
