# Reconstruction Notes: Project 261

A practical clone can treat the scientific core as reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: reion_grid, reion_imex_integrator, reion_solver
- Diagnostics/reporting: reion_brent_optimizer, reion_cholesky, reion_cosmology, reion_finite_difference, reion_grid, reion_imex_integrator

## Solver-shaped plan

- Let `main.py` own the public behavior for 计算宇宙学: 再电离历史数值模拟, while helper modules hold constants and small calculations.
- For 计算宇宙学: 再电离历史数值模拟, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for astrophysics simulation summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Transcript details

Use these public lines as reconstruction checkpoints for `计算宇宙学: 再电离历史数值模拟`; hidden tests are not limited to them.

```text
#  计算宇宙学: 再电离历史数值模拟
#  高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
reion_cosmology.py       : 物理与宇宙学常数
reion_grid.py            : 计算网格生成
计算宇宙学: 再电离历史数值模拟
高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
项目结构:
reion_transfer.py        : 辐射传输 BVP
reion_recombination.py   : 复合动力学 (含延迟反馈)
reion_imex_integrator.py : IMEX Runge-Kutta 积分
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
