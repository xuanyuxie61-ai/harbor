# Reconstruction Notes: Project 205

For reconstruction, model the program as a staged report over sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Model layer: truth_model
- Diagnostics/reporting: io_utils, stability_analysis, uncertainty_propagation
- Supporting modules: adaptive_refinement, design_of_experiments, geometry_domain, numerical_utils, optimal_control_uq, polynomial_chaos

## Coding plan

- Write the CLI by hand if needed; the visible contract for ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING is more important than argparse styling.
- Use calibrated constants for uncertainty quantification quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING.
- Keep the generated executable at the workspace root and make it executable.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING`; hidden tests are not limited to them.

```text
ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING
for Uncertainty Quantification in
Coupled Laser-Reaction-Diffusion Systems
STEP 1: Parameter Definition and Distributions
Dimension d = 4
Parameters: ['D_u', 'D_v', 'alpha', 'beta']
D_u ~ U(0.005, 0.02)
D_v ~ U(0.3, 0.8)
alpha ~ U(1.5, 2.5)
beta ~ U(0.5, 1.5)
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
