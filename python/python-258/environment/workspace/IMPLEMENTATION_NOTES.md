# Reconstruction Notes: Project 258

The observable transcript is organized around reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Model layer: lensing_mass_model, lensing_shear_field
- Diagnostics/reporting: lensing_stability
- Supporting modules: lensing_config, lensing_cosmology, lensing_finite_diff, lensing_kaiser_squires, lensing_mass_sheet, lensing_pde_reconstruct

## Rebuild approach

- Give 弱引力透镜质量重建: 高阶有限差分与稳定性分析 a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 258; preserve the staged progression of the report.
- Render 弱引力透镜质量重建: 高阶有限差分与稳定性分析's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `弱引力透镜质量重建: 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
弱引力透镜质量重建: 高阶有限差分与稳定性分析
Weak Lensing Mass Reconstruction with High-Order FD
PROJECT 258 — 计算宇宙学博士级可复现实验
阶段 1: 宇宙学距离与临界密度计算
透镜红移 z_L = 0.3
源红移   z_S = 1.0
Σ_crit = 2.750e+15 M_sun/Mpc²
共动距离 χ(z_S) = 3397.82 Mpc
E(z) at z=[0.  0.5 1.  2. ]: [1.         1.3224362  1.79083779 3.0327875 ]
源红移分布峰值位置: z = 1.05
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
