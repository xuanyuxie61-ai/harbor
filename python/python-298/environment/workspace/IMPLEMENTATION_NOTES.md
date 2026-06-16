# Reconstruction Notes: Project 298

The cleanroom notes below emphasize magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: grid_generator, high_order_fd, poisson_solver, velocity_quadrature
- Diagnostics/reporting: diagnostics, dispersion_analysis, stability_analyzer
- Supporting modules: boris_pusher, parameter_sweep, particle_loader, plasma_constants, root_finder

## Suggested coding path

- Keep project 298's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of 高阶有限差分 PIC 稳定性分析's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the computational plasma physics transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `高阶有限差分 PIC 稳定性分析`; hidden tests are not limited to them.

```text
PROJECT 298: 高阶有限差分 PIC 稳定性分析
电子数密度 n_e        = 1.000e+20 m^-3
电子温度 T_e          = 100.00 eV (1.160e+06 K)
等离子体频率 omega_pe = 5.641e+11 rad/s
[PART 1] 等离子体物理参数
德拜长度 lambda_D     = 7.434e-06 m
热速度 v_th           = 5.931e+06 m/s
等离子体参数 Lambda   = 1.721e+05
库仑对数 ln_Lambda    = 13.790
碰撞频率 nu_ei        = 1.202e+07 Hz
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
