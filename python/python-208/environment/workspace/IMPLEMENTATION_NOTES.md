# Reconstruction Notes: Project 208

This task can be solved by rebuilding a concise pipeline for sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: circle_quadrature
- Model layer: molecular_dynamics
- Supporting modules: adaptive_fidelity_sampler, banded_covariance, confidence_bounds, coordinate_transform, denoising_filter, experiment_tracker

## Suggested coding path

- For this uncertainty quantification task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 208's printed diagnostics.
- Keep the final report for project 208 deterministic, compact, and ordered like the reference.
- For MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE`; hidden tests are not limited to them.

```text
MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE
Protoplanetary-disk chemistry with autoregressive GP fusion
Python 3.11.11
1. Physical-system initialization
Reference disk state assembled at t = 0.5 Myr
QoI (column CO abundance) = 9.543e+27
2. Fidelity hierarchy initialization
Level 0: Analytic, tags=['power-law', 'steady-state', 'no-planet', 'isothermal']
Level 1: Surrogate, tags=['lagrange', 'sparse-grid', 'no-planet', 'thermal']
Level 2: ReducedODE, tags=['ode-chemistry', 'coarse-grid', 'migrating-planet']
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
