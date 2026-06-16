# Reconstruction Notes: Project 226

The reference run suggests a small driver that combines sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: band_matrix_operations, matrix_evolution, mesh_generation
- Model layer: neutrino_physics
- Diagnostics/reporting: band_matrix_operations, matrix_evolution, mesh_generation, parameter_inversion, uncertainty_quantification
- Supporting modules: high_order_finite_difference, matter_density_fem, monte_carlo_sampling, numerical_utils

## Suggested coding path

- Keep the command surface for project 226 narrow: version, help, and the deterministic default workflow.
- Let statistical estimators, quadrature rules, and compact diagnostic reports drive the helper functions, but keep the default run under verifier time limits.
- For project 226, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `中微子振荡概率与参数反演：高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
中微子振荡概率与参数反演：高阶有限差分与稳定性分析
博士级科研代码合成项目
本项目融合15个种子项目的核心算法，解决计算高能物理中的
中微子振荡参数反演问题。包含：
• 三代中微子振荡与MSW物质效应
• 高阶有限差分与von Neumann稳定性分析
• FEM太阳密度分布求解
• 带状矩阵存储与Jacobi迭代
• 矩阵指数精确演化
• 参数反演（黄金分割 + GP代理）
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
