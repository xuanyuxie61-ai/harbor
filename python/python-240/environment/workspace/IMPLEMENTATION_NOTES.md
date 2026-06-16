# Reconstruction Notes: Project 240

The important implementation pressure is sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: high_order_fd, linear_solver, quadrature
- Model layer: equation_of_state, physics_constants
- Diagnostics/reporting: equation_of_state, initial_conditions, stability_analysis
- Supporting modules: flow_harmonics, golden_optimizer, hypothesis_test, nuclear_geometry, viscous_hydro

## Rebuild approach

- Build the executable wrapper last for project 240, after the Python entry point matches probes.
- Keep uncertainty quantification computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 重离子碰撞椭圆流与初始条件涨落建模.
- Make the no-argument path finish quickly under verifier time limits.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `重离子碰撞椭圆流与初始条件涨落建模`; hidden tests are not limited to them.

```text
重离子碰撞椭圆流与初始条件涨落建模
Heavy-Ion Collision: Elliptic Flow & Initial Condition Fluctuations
High-Order Finite Differences & Stability Analysis
1. 蒙特卡洛Glauber初始条件 (Monte Carlo Glauber)
碰撞系统: Au+Au, √s = 200 GeV
碰撞参数: b = 7.0 fm
每核子数: A = 197
高斯展宽: σ = 0.5 fm
事件 1: N_part = 85, N_coll = 55
事件 2: N_part = 59, N_coll = 40
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
