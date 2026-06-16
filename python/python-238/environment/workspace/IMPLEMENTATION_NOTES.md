# Reconstruction Notes: Project 238

For reconstruction, model the program as a staged report over small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: fermion_solver, high_order_fd
- Model layer: gauge_field, molecular_dynamics
- Diagnostics/reporting: config_io, fermion_solver, gauge_actions, phase_transition, stability_analysis
- Supporting modules: lattice_geometry, observables, su3_algebra

## Suggested coding path

- Write the CLI by hand if needed; the visible contract for 格点 QCD 有限温相变模拟配置 is more important than argparse styling.
- Use calibrated constants for lattice field theory quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in 格点 QCD 有限温相变模拟配置.
- Keep the generated executable at the workspace root and make it executable.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `格点 QCD 有限温相变模拟配置`; hidden tests are not limited to them.

```text
格点 QCD 有限温相变模拟配置
格点尺寸: 3^3 × 4
作用量类型: tree_level_symanzik
积分器: leapfrog
β 值扫描: [5.0, 5.5, 5.7, 6.0]
轨迹数: 5
MD 步数: 8, 步长: 0.02
夸克质量: 0.1
有限差分精度: O(a^4)
Naik 改进: True
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
