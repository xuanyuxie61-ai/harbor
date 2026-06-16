# Reconstruction Notes: Project 260

For reconstruction, model the program as a staged report over reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: adaptive_mesh, cvt_mesh, fd_operators, growth_solver
- Diagnostics/reporting: sensitivity_analysis, stability_analysis
- Supporting modules: cosmo_constants, fisher_forecast, monte_carlo_geom

## Practical reconstruction

- Write the CLI by hand if needed; the visible contract for 暗能量状态方程约束: 高阶有限差分与稳定性分析 is more important than argparse styling.
- Use calibrated constants for astrophysics simulation quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in 暗能量状态方程约束: 高阶有限差分与稳定性分析.
- Keep the generated executable at the workspace root and make it executable.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `暗能量状态方程约束: 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
暗能量状态方程约束: 高阶有限差分与稳定性分析
High-Order FD + von Neumann Stability for DE EoS
1. CPL 暗能量状态方程与 FLRW 背景
基准模型 (LCDM):
CPL 模型 (w0=-0.9, wa=-0.3):
w(z=0) = -0.9000
w(z=1) = -1.0500
Phantom crossing: True
Phantom crossing a: 0.6666666666666667
t(0) = 13.79 Gyr
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
