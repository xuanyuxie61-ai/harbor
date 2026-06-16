# Reconstruction Notes: Project 221

The cleanroom notes below emphasize sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: fd_poisson_solver, ode_integrators, quadrature_engines
- Model layer: physics_constants
- Diagnostics/reporting: markov_transitions, stability_analysis
- Supporting modules: cascade_compartments, event_generator, hankel_moments, kernel_methods, phase_space_geometry

## Implementation strategy

- Keep project 221's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of Computational High-Energy Physics's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the uncertainty quantification transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Public report cues

Use these public lines as reconstruction checkpoints for `Computational High-Energy Physics`; hidden tests are not limited to them.

```text
PROJECT 221: Computational High-Energy Physics
Monte Carlo Event Generation & Phase-Space Integration:
High-Order Finite Differences & Stability Analysis
Scientific problem: 2->3 scattering in simplified QCD
Center-of-mass energy: 13 TeV (LHC Run 2)
Process: pp -> 3 jets at LO + NLO K-factor
Modules:
A. Phase-space geometry and fractal hadronization boundary
B. High-order quadrature and kinematic root-finding
C. Finite difference DGLAP/GLR-MQ evolution
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
