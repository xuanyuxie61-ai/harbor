# Reconstruction Notes: Project 271

For reconstruction, model the program as a staged report over small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: boundary_mesh, high_order_fd, mc_quadrature
- Diagnostics/reporting: naca_dispersion, spectral_calibration, stability_analysis
- Supporting modules: benchmark_kernels, constants, cvt_sampler, elliptic_determinants, finite_size_scaling, lattice_geometry

## Coding plan

- Write the CLI by hand if needed; the visible contract for 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis is more important than argparse styling.
- Use calibrated constants for lattice field theory quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis.
- Keep the generated executable at the workspace root and make it executable.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis`; hidden tests are not limited to them.

```text
(small-scale reproducible experiment, Python)
Stage 1 : NAS-style benchmark kernels
btrix                  residual = 1.619e-16
cfft2d                 residual = 1.361e-15
PROJECT 271 : 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis
cholsky                residual = 1.349e-16
emit                   residual = 5.554e-16
dct                    residual = 8.882e-16
su2_chain              residual = 1.333e-15
random_mean            residual = 5.002e-01
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
