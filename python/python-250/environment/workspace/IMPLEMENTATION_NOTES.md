# Reconstruction Notes: Project 250

A practical clone can treat the scientific core as sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: implicit_solver, mesh, radiation_quadrature
- Model layer: sasi_dynamics
- Diagnostics/reporting: diagnostics, radiation_quadrature, reconstruction, stability
- Supporting modules: bayesian_entropy, causality, constants, eos, hydro, initial

## Suggested coding path

- Let `main.py` own the public behavior for 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析, while helper modules hold constants and small calculations.
- For 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for uncertainty quantification summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `超新星爆发辐射流体模拟：高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
1. 球对称网格构造
最小 dr = 1.01e+05 cm,  最大 dr = 4.63e+07 cm
2. 物态、不透明度、离散纵标
物态: γ=1.667, μ=0.617, Y_e=0.42
单元数: 64,  r ∈ [1.00e+06, 5.00e+08] cm
不透明度: 10 × 8 切比雪夫节点
κ(T=5e9, ρ=1e9) = 9.766e+18 cm^2/g
S_N 求积: 18 方向, 权重和 = 1.5708e+00
S_4 level-symmetric: 8 方向
3. 前身星初始剖面
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
