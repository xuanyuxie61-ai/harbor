# Reconstruction Notes: Project 239

A practical clone can treat the scientific core as magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: qgp_grid, qgp_operators, qgp_spectral_integration, qgp_time_integration
- Diagnostics/reporting: qgp_conservation, qgp_exact_solutions, qgp_initial_conditions, qgp_parameter_estimation, qgp_spectral_integration, qgp_stability_analysis
- Supporting modules: qgp_config, qgp_eos, qgp_flow_harmonics, qgp_freeze_out, qgp_spiking_source

## Implementation strategy

- Let `main.py` own the public behavior for QGP hydrodynamics module smoke, while helper modules hold constants and small calculations.
- For QGP hydrodynamics module smoke, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for computational plasma physics summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Public report cues

Use these public lines as reconstruction checkpoints for `QGP hydrodynamics module smoke`; hidden tests are not limited to them.

```text
PROJECT 239: QGP hydrodynamics module smoke
import qgp_config                   ok
import qgp_eos                      ok
import qgp_grid                     ok
import qgp_initial_conditions       ok
import qgp_operators                ok
import qgp_time_integration         ok
import qgp_stability_analysis       ok
import qgp_flow_harmonics           ok
import qgp_spectral_integration     ok
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
