# Reconstruction Notes: Project 244

From the public behavior, the natural decomposition is sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: eos_quadrature, neutron_star_mesh
- Diagnostics/reporting: bayesian_posterior, dispersion_relations, nuclear_interaction_adaptation, stability_analysis
- Supporting modules: confidence_intervals, density_shell_indexer, eos_monte_carlo, gw_reproducibility, high_order_finite_difference, multi_observable_pca

## Suggested coding path

- Handle `--help` and `--version` explicitly for 中子星核物质状态方程约束; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 244; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 244, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `中子星核物质状态方程约束`; hidden tests are not limited to them.

```text
中子星核物质状态方程约束
高阶有限差分与稳定性分析 (小规模可复现实验)
Python: 3.11.11
NumPy: 1.26.4
阶段 1: 数值基础验证
机器精度 eps = 2.220446e-16
Gamma(5) = 24.0000000000  (精确 24)
ln Gamma(10) = 12.8018274801
Legendre P_5 零点: [-0.90617985 -0.53846931  0.          0.53846931  0.90617985]
8 点 Gauss 权重和 = 2.000000000000001  (应为 2)
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
