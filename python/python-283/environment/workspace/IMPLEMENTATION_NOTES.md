# Reconstruction Notes: Project 283

A practical clone can treat the scientific core as high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: high_order_fd, mesh_generator, poisson_defect_solver
- Model layer: carrier_transport_dg, defect_field_interpolation, defect_rate_equations, fem_scalar_field
- Diagnostics/reporting: defect_field_interpolation, defect_rate_equations, stability_analysis
- Supporting modules: benchmark_analyzer, langevin_defect_sampler, ml_defect_predictor, perovskite_constants

## Coding plan

- Let `main.py` own the public behavior for Perovskite Solar-Cell Defect-State Calculator, while helper modules hold constants and small calculations.
- For Perovskite Solar-Cell Defect-State Calculator, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for numerical-methods benchmark summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `Perovskite Solar-Cell Defect-State Calculator`; hidden tests are not limited to them.

```text
PROJECT 283: Perovskite Solar-Cell Defect-State Calculator
High-Order Finite Differences + Stability Analysis
Device length: 500.0 nm
Built-in voltage: 1.00 V
FD order (2p): 12
Defect density (default): 1.00e+21 m^-3
1. Physical constants sanity check
V_t [mV]               = 25.852
N_c [1/m^3]            = 1.043e+24
N_v [1/m^3]            = 1.458e+24
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
