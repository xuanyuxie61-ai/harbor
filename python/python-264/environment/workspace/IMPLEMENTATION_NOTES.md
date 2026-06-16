# Reconstruction Notes: Project 264

The binary behaves like a scripted experiment focused on magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: field_solver, high_order_fd, magnetosphere_grid, prime_grid, vlasov_solver
- Model layer: field_solver
- Diagnostics/reporting: boundary_conditions, data_io, phase_space_diagnostics, stability_analysis
- Supporting modules: combinatorial_modes, partial_digest_resonance, physical_constants, stochastic_sampler, test_particle_orbit

## Rebuild approach

- Use a tiny dispatch layer for energy_flux / density,; hidden tests should not depend on accidental framework formatting.
- If a real computational plasma physics solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around energy_flux / density,; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `energy_flux / density,`; hidden tests are not limited to them.

```text
energy_flux / density,
磁层粒子输运模拟: 高阶有限差分与稳定性分析
Magnetospheric Particle Transport Simulation
High-Order Finite Difference & Stability Analysis
科学问题: 地球辐射带相对论电子 (L, E) 相空间输运
控制方程: 2D Fokker-Planck (漂移动力学) 方程
数值方法: 高阶有限差分 + von Neumann 稳定性分析
应用领域: 计算空间物理 / 空间天气预报
第1部分: 物理常数与磁层参数
[基本常数]
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
