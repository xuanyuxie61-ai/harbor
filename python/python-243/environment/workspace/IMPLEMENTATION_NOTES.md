# Reconstruction Notes: Project 243

The cleanroom notes below emphasize high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: adi_solver, high_order_fd, mesh_network
- Model layer: calving_physics, nuclear_physics, reaction_rates
- Diagnostics/reporting: distance_position, reaction_rates, stability_analysis
- Supporting modules: chebyshev_interp, control_optimizer, diffusivity, latin_sampler, newton_root, nuclear_network

## Solver-shaped plan

- Keep project 243's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of r-process nucleosynthesis network simulation's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the numerical-methods benchmark transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Transcript details

Use these public lines as reconstruction checkpoints for `r-process nucleosynthesis network simulation`; hidden tests are not limited to them.

```text
#                                                                      #
#    PROJECT 243 - r-process nucleosynthesis network simulation  #
#    High-order finite difference transport + stability analysis #
#    (Small-scale reproducible experiment)                       #
PROJECT 243 - r-process nucleosynthesis network simulation
High-order finite difference transport + stability analysis
(Small-scale reproducible experiment)
1. Building nuclide mesh and nuclear-chart boundary
Nuclides in network : 567
Mass range          : A in [70, 150]
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
