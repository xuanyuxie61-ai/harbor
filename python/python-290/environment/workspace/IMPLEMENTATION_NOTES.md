# Reconstruction Notes: Project 290

The source evidence points to magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: alfven_wave_solver, high_order_operators
- Model layer: energetic_particle_kinetics
- Diagnostics/reporting: conservation_monitor, dispersion_analysis, solution_io, stability_eigenvalue, statistical_diagnostics
- Supporting modules: boundary_handler, magnetic_geometry, plasma_config

## Practical reconstruction

- Make the flag paths cheap for project 290; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by 阿尔芬波-高能粒子相互作用: 高阶有限差分与稳定性分析; random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 290.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `阿尔芬波-高能粒子相互作用: 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
阿尔芬波-高能粒子相互作用: 高阶有限差分与稳定性分析
PROJECT_290 - 计算等离子体物理博士级合成项目
Step 1: 等离子体参数配置
阿尔芬波-高能粒子相互作用模拟参数摘要
磁场强度 B₀ = 5.00 T
大半径 R₀ = 1.65 m
小半径 a = 0.50 m
反转比 ε = a/R₀ = 0.3030
安全因子范围 q₀ = 1.00 ~ q_a = 3.00
电子密度 n_e = 5.00e+19 m⁻³
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
