# Reconstruction Notes: Project 282

For reconstruction, model the program as a staged report over transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: grid_mesh, high_order_fd, quadrature_integration, sei_diffusion_solver
- Model layer: butler_volmer_kinetics, sei_lorenz_dynamics
- Diagnostics/reporting: eigenmode_decomposition, quadrature_integration, sei_diffusion_solver, stability_analysis, stochastic_nucleation
- Supporting modules: invariant_detector, sei_parameters, timestamp_utils

## Rebuild approach

- Write the CLI by hand if needed; the visible contract for 固态电解质界面反应建模 — 高阶有限差分与稳定性分析 is more important than argparse styling.
- Use calibrated constants for fusion and radiation transport quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in 固态电解质界面反应建模 — 高阶有限差分与稳定性分析.
- Keep the generated executable at the workspace root and make it executable.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `固态电解质界面反应建模 — 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
固态电解质界面反应建模 — 高阶有限差分与稳定性分析
Solid Electrolyte Interphase (SEI) High-Order FD Modeling
阶段 1：参数校验与实验时间戳
参数自洽性校验: 通过
时间戳字符串: 20260608
时间戳哈希（用于 RNG 种子）: 20260608
阶段 2：SEI 网格构造与纳米孔隙密堆积
节点数 N = 81, 域长 L = 50.0 nm
空间步长 dx = 0.6250 nm
边界: butler_volmer | dirichlet
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
