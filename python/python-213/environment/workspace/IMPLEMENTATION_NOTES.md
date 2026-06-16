# Reconstruction Notes: Project 213

The source evidence points to transport discretization, source terms, and verification statistics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: interior_point_solver, sparse_grid_quadrature
- Model layer: laser_physics, reaction_diffusion
- Diagnostics/reporting: fem_discretization, interior_point_solver, polynomial_approximation, reaction_diffusion
- Supporting modules: convex_program, neural_surrogate, newton_kkt, special_barrier

## Solver-shaped plan

- Make the flag paths cheap for project 213; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by 反应-扩散系统的 PDE 约束最优控制; random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 213.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Transcript details

Use these public lines as reconstruction checkpoints for `反应-扩散系统的 PDE 约束最优控制`; hidden tests are not limited to them.

```text
PROJECT 213: 凸优化与内点法
反应-扩散系统的 PDE 约束最优控制
博士级科学计算合成项目
Python 版本: 3.11.11
NumPy 版本: 1.26.4
演示 1: 凸多边形优化域 (882_polygon)
域顶点数: 6
凸性检测: 凸
面积: 2.457347
质心: (-0.0418, 0.0241)
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
