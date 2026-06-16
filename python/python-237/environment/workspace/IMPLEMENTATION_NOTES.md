# Reconstruction Notes: Project 237

The reference run suggests a small driver that combines small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Model layer: gauge_field
- Diagnostics/reporting: canalization_topology, interpolation_utils, spectral_analysis
- Supporting modules: codeword_gauge_fix, constants, coupled_eigenvalue, gauge_orbit, gauge_sampler, lattice_geometry

## Solver-shaped plan

- Keep the command surface for project 237 narrow: version, help, and the deterministic default workflow.
- Let gauge-field state, action terms, and stability checks drive the helper functions, but keep the default run under verifier time limits.
- For project 237, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Transcript details

Use these public lines as reconstruction checkpoints for `Lattice QCD: Quark Propagator & Gauge Field Sampling`; hidden tests are not limited to them.

```text
Lattice QCD: Quark Propagator & Gauge Field Sampling
High-Order Finite Differences and Stability Analysis
(Small-Scale Reproducible Experiment)
Step 1: Lattice Initialization
Lattice: 4^3 x 4 = 256 sites
Lattice spacing: a = 0.1 fm
Gauge coupling: beta = 2.5, g^2 = 1.6000
Bare quark mass: m0 = 0.1
Hopping parameter: kappa = 0.121951
Critical kappa: kappa_c = 0.125000
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
