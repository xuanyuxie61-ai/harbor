# Reconstruction Notes: Project 227

For reconstruction, model the program as a staged report over magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Model layer: magnetic_field, noise_model
- Diagnostics/reporting: numerical_propagation, pattern_recognition, stability_analysis
- Supporting modules: chi_square, constants, detector_geometry, kalman_engine, material_effects

## Implementation strategy

- Write the CLI by hand if needed; the visible contract for 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合 is more important than argparse styling.
- Use calibrated constants for computational plasma physics quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合.
- Keep the generated executable at the workspace root and make it executable.

## Public report cues

Use these public lines as reconstruction checkpoints for `计算高能物理: 粒子径迹重建与 Kalman 滤波拟合`; hidden tests are not limited to them.

```text
计算高能物理: 粒子径迹重建与 Kalman 滤波拟合
高阶有限差分与稳定性分析 (小规模可复现实验)
PROJECT_227 — 博士级合成项目
1. 探测器配置
TrackingDetector: 8 layers
Radius range: [33.0, 580.0] mm
Total material budget: 0.2400 X₀
Layer 0: r=33.0mm, t/X₀=0.0200, σ_r=10.0μm, σ_z=115.0μm, mat=silicon
Layer 1: r=50.5mm, t/X₀=0.0200, σ_r=10.0μm, σ_z=115.0μm, mat=silicon
Layer 2: r=65.5mm, t/X₀=0.0250, σ_r=10.0μm, σ_z=115.0μm, mat=silicon
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
