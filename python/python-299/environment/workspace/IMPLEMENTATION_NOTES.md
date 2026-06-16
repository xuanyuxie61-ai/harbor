# Reconstruction Notes: Project 299

From the public behavior, the natural decomposition is sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: adaptive_mesh, banded_solver, fp_solver, velocity_grid
- Diagnostics/reporting: fp_collision, special_functions, stability_analysis
- Supporting modules: calendar_utils, entropy_norm, phase_space_contrastive, physical_constants, pwl_velocity, uncertainty_tracking

## Implementation strategy

- Handle `--help` and `--version` explicitly for Fokker-Planck 碰撞输运计算; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 299; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 299, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Public report cues

Use these public lines as reconstruction checkpoints for `Fokker-Planck 碰撞输运计算`; hidden tests are not limited to them.

```text
Fokker-Planck 碰撞输运计算
高阶有限差分与稳定性分析 — 博士级科学计算项目
儒略日: 2461207
Unix 时间戳: 1781543210.293
§1 等离子体参数设置
等离子体参数摘要
ln(Lambda)    = 16.3040
碰撞时间 τ_c = 4.3011e+15 s
碰撞频率 ν_c = 2.3250e-16 s⁻¹
§2 速度空间网格构造
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
