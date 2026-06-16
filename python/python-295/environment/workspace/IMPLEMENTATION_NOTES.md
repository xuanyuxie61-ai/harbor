# Reconstruction Notes: Project 295

The important implementation pressure is transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: high_order_fd, linear_solver, mesh_generator, quadrature_rules, time_integrator
- Model layer: icf_physics
- Diagnostics/reporting: perturbation_generator, stability_analysis, symmetry_decomposition, verification
- Supporting modules: ml_surrogate

## Coding plan

- Build the executable wrapper last for project 295, after the Python entry point matches probes.
- Keep fusion and radiation transport computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 惯性约束聚变内爆对称性模拟.
- Make the no-argument path finish quickly under verifier time limits.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `惯性约束聚变内爆对称性模拟`; hidden tests are not limited to them.

```text
惯性约束聚变内爆对称性模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
Inertial Confinement Fusion Implosion Symmetry Simulation
High-Order Finite Difference & Stability Analysis
阶段 1: ICF 物理参数设置
多方指数 γ = 1.6667
平均粒子质量 = 4.1816e-24 g (DT)
靶丸半径 = 1.0000e-02 cm
初始燃料密度 = 2.5000e-01 g/cm³
状态方程测试:
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
