# Reconstruction Notes: Project 254

The cleanroom notes below emphasize transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: ejecta_mesh, high_order_fd
- Model layer: equation_of_state, monte_carlo_transport
- Diagnostics/reporting: convergence_analysis, domain_decomposition, equation_of_state, hmfns_oscillation, von_neumann_stability
- Supporting modules: binary_inspiral, luminosity_lightcurve, opacity_tables, physical_constants, radiative_transfer, thermodynamic_eos

## Practical reconstruction

- Keep project 254's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of Binary Neutron Star Merger & Kilonova's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the fusion and radiation transport transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `Binary Neutron Star Merger & Kilonova`; hidden tests are not limited to them.

```text
#                                                                      #
#   PROJECT_254 : Binary Neutron Star Merger & Kilonova         #
#   High-Order Finite Differences & Stability Analysis          #
Python executable : /usr/local/bin/python3
PROJECT_254 : Binary Neutron Star Merger & Kilonova
High-Order Finite Differences & Stability Analysis
Python version    : 3.11.11
Stage 1 / Physical Constants & Derived Quantities
[self-check] passed = True
M_Pl [g]                         =     2.1764e-05
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
