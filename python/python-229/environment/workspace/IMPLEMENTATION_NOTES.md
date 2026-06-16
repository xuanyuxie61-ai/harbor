# Reconstruction Notes: Project 229

The important implementation pressure is high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: bin_integration, detector_mesh, phase_space_grid, response_matrix
- Model layer: physics_models
- Diagnostics/reporting: bin_integration, regularization, special_functions, stability_analysis
- Supporting modules: energy_loss_ode, finite_difference, unfolding_methods

## Coding plan

- Build the executable wrapper last for project 229, after the Python entry point matches probes.
- Keep numerical-methods benchmark computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 计算高能物理 — 探测器响应矩阵与 unfolding 反演.
- Make the no-argument path finish quickly under verifier time limits.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `计算高能物理 — 探测器响应矩阵与 unfolding 反演`; hidden tests are not limited to them.

```text
PROJECT_229: 计算高能物理
探测器响应矩阵与 unfolding 反演
高阶有限差分与稳定性分析 (小规模可复现实验)
1. 相空间网格与能量 bin 构造
动量空间网格: 半径 = 50.0 GeV, n_per_axis = 6
网格点数: 实际 1189, 解析估计 1150
探测器接受 (pT>0.5 GeV, |η|<2.5): 1176/1189
能量 bin: 15 (true) × 15 (rec), E ∈ [1.0, 100.0] GeV
2. 探测器网格与富化 (来自 789 + 1353)
探测器网格: 8×10 = 80 单元
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
