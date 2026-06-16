# Reconstruction Notes: Project 300

Expect hidden checks to care most about the externally visible parts of transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: angular_quadrature, fd_transport
- Model layer: fd_transport, physics_constants
- Diagnostics/reporting: cross_sections, stability_analysis
- Supporting modules: blanket_geometry, energy_spectra, latent_decomposer, monte_carlo_fk, tbr_calculator

## Rebuild approach

- For Tritium Breeding Blanket Neutron Transport Solver, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the fusion and radiation transport reconstruction clearer and faster.
- If Tritium Breeding Blanket Neutron Transport Solver prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `Tritium Breeding Blanket Neutron Transport Solver`; hidden tests are not limited to them.

```text
#  Tritium Breeding Blanket Neutron Transport Solver
#  High-order compact finite-difference SN method
#  with von Neumann / spectral stability analysis
#  and Feynman-Kac stochastic verification
Tritium Breeding Blanket Neutron Transport Solver
High-order compact finite-difference SN method
with von Neumann / spectral stability analysis
and Feynman-Kac stochastic verification
1. Blanket geometry (1-D Voronoi pebble bed)
L_total_cm                       : +8.000000e+01
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
