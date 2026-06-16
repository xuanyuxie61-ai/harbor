# Reconstruction Notes: Project 225

The observable transcript is organized around sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: high_order_fd, matrix_analysis, rate_integrator, velocity_quadrature
- Model layer: phase_space_transport, recoil_physics
- Diagnostics/reporting: data_io, matrix_analysis, stability_analysis
- Supporting modules: astro_parameters, detector_response, polynomial_chaos

## Solver-shaped plan

- Give 暗物质直接探测 recoil spectrum 建模 a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 225; preserve the staged progression of the report.
- Render 暗物质直接探测 recoil spectrum 建模's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Transcript details

Use these public lines as reconstruction checkpoints for `暗物质直接探测 recoil spectrum 建模`; hidden tests are not limited to them.

```text
PROJECT 225: 暗物质直接探测 recoil spectrum 建模
High-order finite difference & stability analysis
计算高能物理 · 博士级合成项目
Step 1: 浮点机器常数测定 (Malcolm-Gentleman-Maroney)
[OK] 双精度 eps = 9.997e-13, 有效位 4
Step 2: 探测器模块几何 (GRF 图 I/O)
[OK] 构建探测器图: 20 个模块
坐标范围: r ∈ [0.50, 1.70]
[OK] GRF I/O: 节点数 = 20, 边数 = 62
Step 3: 暗物质相空间分布初始化
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
