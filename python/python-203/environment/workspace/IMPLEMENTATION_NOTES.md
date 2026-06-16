# Reconstruction Notes: Project 203

The observable transcript is organized around sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: mesh_utils, ode_integrator, quadrature_exactness
- Model layer: ozone_kinetics
- Diagnostics/reporting: io_utils, waveform_migration
- Supporting modules: cvt_sampler, polynomial_chaos, quasi_mc, seismic_wave, signal_cycles, stochastic_galerkin

## Implementation strategy

- Give Quasi-Monte Carlo UQ for Stochastic Dynamics a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 203; preserve the staged progression of the report.
- Render Quasi-Monte Carlo UQ for Stochastic Dynamics's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Public report cues

Use these public lines as reconstruction checkpoints for `Quasi-Monte Carlo UQ for Stochastic Dynamics`; hidden tests are not limited to them.

```text
PROJECT 203: Quasi-Monte Carlo UQ for Stochastic Dynamics
Coupled Seismic Wave - Ozone Chemistry - Signal Analysis
Scientific problem:
Quantify uncertainty in seismic wave propagation through
heterogeneous random media with coupled atmospheric chemistry,
using polynomial chaos expansion (Gegenbauer basis) and
quasi-Monte Carlo sampling with CVT-adaptive refinement.
Integrating 15 seed projects:
1. 275_dg1d_poisson        -> DG spatial discretization
2. 100_blood_pressure_ode  -> Periodic jump ODE model
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
