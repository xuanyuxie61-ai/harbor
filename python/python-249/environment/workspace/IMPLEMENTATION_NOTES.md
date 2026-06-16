# Reconstruction Notes: Project 249

For reconstruction, model the program as a staged report over reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: mesh_adaptation, nuclear_quadrature, time_integrator
- Diagnostics/reporting: mesh_adaptation, perturbation_engine, stability_analysis
- Supporting modules: burning_classifier, data_integrity, epoch_tracker, finite_difference, nuclear_network, physical_constants

## Solver-shaped plan

- Write the CLI by hand if needed; the visible contract for Stellar evolution with coupled nuclear networks is more important than argparse styling.
- Use calibrated constants for astrophysics simulation quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in Stellar evolution with coupled nuclear networks.
- Keep the generated executable at the workspace root and make it executable.

## Transcript details

Use these public lines as reconstruction checkpoints for `Stellar evolution with coupled nuclear networks`; hidden tests are not limited to them.

```text
PROJECT_249 - Stellar evolution with coupled nuclear networks
High-order finite difference + adaptive mesh + stiff integrators
Stage 1 : Adaptive mass grid construction (1D CVT)
max/min spacing ratio = 2.650
smoothness measure    = 0.635
first 6 mass coords   = ['0.000e+00', '4.208e+32', '1.143e+33', '1.887e+33', '2.660e+33', '3.530e+33']
Stage 2 : Initial stellar model construction
constructed 48 stellar zones
centre  r = 5.191e+09 cm   T = 3.300e+07 K   P = 4.816e+18
surface r = 2.270e+10 cm   T = 3.300e+07 K   P = 1.833e+18
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
