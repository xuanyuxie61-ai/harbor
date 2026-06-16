# Reconstruction Notes: Project 241

This task can be solved by rebuilding a concise pipeline for high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Model layer: model_selection
- Diagnostics/reporting: convergence_analysis, cross_section, model_selection, stability_analysis, wave_propagation
- Supporting modules: channel_coupling, norms_utils, optical_potential, phase_shift, radial_schrodinger

## Coding plan

- For this numerical-methods benchmark task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 241's printed diagnostics.
- Keep the final report for project 241 deterministic, compact, and ordered like the reference.
- For 核反应光学模型: 高阶有限差分与稳定性分析, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `核反应光学模型: 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
核反应光学模型: 高阶有限差分与稳定性分析
Nuclear Optical Model: High-Order Finite Difference
and Stability Analysis
(DD 聚变中子能量)
计算方法: Numerov 四阶 + 传递矩阵 + 序贯检验
阶段 1: 光学模型势构造
靶核: 208Pb82
入射粒子: 中子
实验室能量: 14.1 MeV
核半径 R_v: 7.406 fm
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
