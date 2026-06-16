# Reconstruction Notes: Project 274

This task can be solved by rebuilding a concise pipeline for small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: tridiagonal_fd
- Model layer: ion_transport_kinetics
- Diagnostics/reporting: diophantine_phonon, graph_optimization, ion_transport_kinetics, kmeans_gap_quantization, monte_carlo_fluctuations, special_functions
- Supporting modules: brillouin_zone, ema_optimizer, lattice_geometry, pipeline, polynomial_basis

## Suggested coding path

- For this lattice field theory task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 274's printed diagnostics.
- Keep the final report for project 274 deterministic, compact, and ordered like the reference.
- For Electron-Phonon Coupling and Tc Prediction, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `Electron-Phonon Coupling and Tc Prediction`; hidden tests are not limited to them.

```text
PROJECT 274 -- Electron-Phonon Coupling and Tc Prediction
High-order finite differences, stability analysis,
small-scale reproducible experiment.
Pipeline execution
[pipeline] Stage 1 : adaptive k-point mesh        ... OK  (0.198s)
[pipeline] Stage 2 : Brillouin zone               ... OK  (0.003s)
[pipeline] Stage 3 : phonon-shell spectrum        ... OK  (0.003s)
[pipeline] Stage 4 : Eliashberg kernel            ... OK  (0.025s)
[pipeline] Stage 5 : spectral gap expansion       ... OK  (0.003s)
[pipeline] Stage 6 : Tc search                    ... OK  (0.333s)
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
