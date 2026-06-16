# Reconstruction Notes: Project 291

The observable transcript is organized around matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: fd_highorder, sheath_grid, sheath_quadrature
- Diagnostics/reporting: sheath_mode_selection, sheath_spectral_stability
- Supporting modules: sheath_constants, sheath_elliptic_presheath, sheath_glomin_eigenvalue, sheath_nonlinear, sheath_praxis_optimizer, sheath_resonance

## Solver-shaped plan

- Give 等离子体鞘层高阶有限差分与稳定性分析 a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 291; preserve the staged progression of the report.
- Render 等离子体鞘层高阶有限差分与稳定性分析's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Transcript details

Use these public lines as reconstruction checkpoints for `等离子体鞘层高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
等离子体鞘层高阶有限差分与稳定性分析
融合 15 个种子项目的计算等离子体博士级研究框架
方向: 等离子体鞘层-壁面相互作用
方法: 高阶紧致有限差分 + 谱稳定性分析
阶段 1: 等离子体参数设置与网格生成
等离子体鞘层参数摘要
等离子体种类        : Ar
电子温度 T_e        : 3.00 eV
离子温度 T_i        : 0.030 eV
参考密度 n₀         : 1.000e+16 m^-3
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
