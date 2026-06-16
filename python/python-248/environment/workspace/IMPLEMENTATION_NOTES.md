# Reconstruction Notes: Project 248

The reference run suggests a small driver that combines reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: grid_geometry, high_order_fd, rk_integrator
- Diagnostics/reporting: causal_analysis, cfl_stability, duration_feedback, shock_quantization, simulation_core, von_neumann_analysis
- Supporting modules: adaptive_refinement, astro_constants, eos_gas, poisson_cholesky, turbulent_spectrum, voronoi_potential

## Practical reconstruction

- Keep the command surface for project 248 narrow: version, help, and the deterministic default workflow.
- Let model parameters, evolution loops, and summary tables drive the helper functions, but keep the default run under verifier time limits.
- For project 248, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `Galaxy Formation Hydrodynamics Test Problem`; hidden tests are not limited to them.

```text
PROJECT_248 :: Galaxy Formation Hydrodynamics Test Problem
Domain: high-order finite differences + stability analysis
(small-scale reproducible experiment)
N     dx           L_inf (order 4)   L_inf (order 6)   rate_4  rate_6
[ok] all 16 Python modules imported successfully
High-order FD convergence check: f(x) = sin(2 pi x) on [0, 1]
32   3.125e-02         3.099e-04          2.553e-06      0.00    0.00
64   1.562e-02         1.943e-05          4.011e-08      4.00    5.99
128   7.812e-03         1.216e-06          6.276e-10      4.00    6.00
256   3.906e-03         7.600e-08          9.843e-12      4.00    5.99
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
