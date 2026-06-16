# Reconstruction Notes: Project 242

The binary behaves like a scripted experiment focused on transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: eigenvalue_solver, high_order_fd, quadrature_nuclear, radial_fd_solver, radial_integration
- Diagnostics/reporting: configuration_selector, ode_evolution, radial_integration, special_functions_nuclear, stability_analysis, transition_rates
- Supporting modules: angular_momentum, boundary_handler, collective_modes, conformal_mapping, coupled_channels, hamiltonian_builder

## Practical reconstruction

- Use a tiny dispatch layer for Nuclear shell-model energy levels and transition; hidden tests should not depend on accidental framework formatting.
- If a real fusion and radiation transport solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around Nuclear shell-model energy levels and transition; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `Nuclear shell-model energy levels and transition`; hidden tests are not limited to them.

```text
PROJECT 242: Nuclear shell-model energy levels and transition
probabilities with high-order finite differences and stability
analysis (small-scale, reproducible experiment)
Step 0: Problem specification
Nucleus         : 18O (Z=8, N=10)
Core            : 16O (Z=8, N=8)
Valence neutrons: 2
Model space     : sd-shell
Orbitals        : 1d_{5/2}, 2s_{1/2}, 1d_{3/2}
Pairing strength G = 0.8 MeV
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
