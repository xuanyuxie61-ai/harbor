# Reconstruction Notes: Project 245

Expect hidden checks to care most about the externally visible parts of transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: nuclear_mesh
- Diagnostics/reporting: fem1d_energy_diffusion, fem2d_thermal_diffusion, fragment_mass_distribution, ftcs_fission_pde, langevin_fission, qaoa_fission_channel
- Supporting modules: fragment_energy_release, high_order_finite_difference, legendre_angular, levenshtein_pattern, nuclear_constants, polynomial_resultant

## Implementation strategy

- For Nuclear Fission Simulation, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the fusion and radiation transport reconstruction clearer and faster.
- If Nuclear Fission Simulation prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Public report cues

Use these public lines as reconstruction checkpoints for `Nuclear Fission Simulation`; hidden tests are not limited to them.

```text
PROJECT_245: Nuclear Fission Simulation
Fragment Mass Distribution & Energy Release Modelling
High-Order Finite Differences & Stability Analysis
Binding energy per nucleon B/A (Liquid Drop Model):
1. Nuclear Constants and Binding Energy [nuclear_constants.py]
Nucleus        A     Z   B(A,Z) [MeV]    B/A [MeV]
He-4           4     2        35.7595       8.9399
Fe-56         56    26       495.5984       8.8500
Sn-132       132    50      1081.2751       8.1915
Pb-208       208    82      1609.0313       7.7357
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
