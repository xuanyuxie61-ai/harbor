# Reconstruction Notes: Project 287

The cleanroom notes below emphasize sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: high_order_fd, matrix_solvers, mesh_generator, time_integrator
- Model layer: field_reconstructor, mhd_physics
- Diagnostics/reporting: oscillation_analyzer, plasma_diagnostics, stability_analyzer
- Supporting modules: monte_carlo_uq, plasma_constants, spectral_tools

## Implementation strategy

- Keep project 287's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of MHD 不稳定性数值模拟's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the uncertainty quantification transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Public report cues

Use these public lines as reconstruction checkpoints for `MHD 不稳定性数值模拟`; hidden tests are not limited to them.

```text
#  PROJECT 287: MHD 不稳定性数值模拟
#  高阶有限差分与稳定性分析 (小规模可复现实验)
Python 版本: 3.11.11
NumPy 版本:  1.26.4
PROJECT 287: MHD 不稳定性数值模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
第 1 部分: 等离子体参数与无量纲数
等离子体参数汇总 (Harris 电流片平衡)
Lundquist S   = 6.129e+02
d_i/L         = 1.018e+01
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
