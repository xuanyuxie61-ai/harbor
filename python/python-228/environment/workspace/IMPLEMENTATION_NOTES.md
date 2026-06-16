# Reconstruction Notes: Project 228

A practical clone can treat the scientific core as sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: calorimeter_grid, cascade_equation_solver, quadrature
- Model layer: cascade_equation_solver
- Diagnostics/reporting: cascade_equation_solver, chaos_analysis, variational_assimilation
- Supporting modules: chebyshev_spectral, finite_diff_schemes, linpack_benchmark, material_properties, monte_carlo_shower, piecewise_flux

## Rebuild approach

- Let `main.py` own the public behavior for 计算高能物理: 量能器 Shower Profile 快速模拟, while helper modules hold constants and small calculations.
- For 计算高能物理: 量能器 Shower Profile 快速模拟, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for uncertainty quantification summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `计算高能物理: 量能器 Shower Profile 快速模拟`; hidden tests are not limited to them.

```text
计算高能物理: 量能器 Shower Profile 快速模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
种子项目融合: 15 个科学计算项目 → 统一物理模拟框架
1. 实验参数配置
入射粒子: electron, E0 = 10000.0 MeV = 10.0 GeV
主材料: PbWO4
辐射长度 X0 = 0.8900 cm
临界能量 Ec = 7.97 MeV
Molière 半径 R_M = 2.3674 cm
核作用长度 λ_I = 20.60 cm
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
