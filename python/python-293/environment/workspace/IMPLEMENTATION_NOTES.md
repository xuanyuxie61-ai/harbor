# Reconstruction Notes: Project 293

For reconstruction, model the program as a staged report over magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: banded_matrix_ops, grid_integer_lib, high_order_fd, mesh_topology, velocity_space_quadrature
- Diagnostics/reporting: phase_space_diagnostics, stability_analysis, vlasov_evolution
- Supporting modules: plasma_constants, signed_distance_geom, wave_particle_resonance

## Implementation strategy

- Write the CLI by hand if needed; the visible contract for 空间等离子体波粒相互作用 is more important than argparse styling.
- Use calibrated constants for computational plasma physics quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in 空间等离子体波粒相互作用.
- Keep the generated executable at the workspace root and make it executable.

## Public report cues

Use these public lines as reconstruction checkpoints for `空间等离子体波粒相互作用`; hidden tests are not limited to them.

```text
空间等离子体波粒相互作用
高阶有限差分与稳定性分析 (小规模可复现实验)
物理模型: 1D 静电 Vlasov-Poisson 系统
数值方法: 半拉格朗日 + 谱方法 Poisson 求解器
分析工具: von Neumann 稳定性 / Hankel 模态分解
实验 1: Landau 阻尼
网格: nx=32, nv=64
空间: [0.00, 12.57], dx=0.3927
速度: [-6.00, 6.00], dv=0.1875
理论值:
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
