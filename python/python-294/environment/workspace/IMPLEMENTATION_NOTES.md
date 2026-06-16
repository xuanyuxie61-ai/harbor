# Reconstruction Notes: Project 294

A practical clone can treat the scientific core as sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: collision_operator, dispersion_solver, fd_operators, matrix_update
- Diagnostics/reporting: boundary_conditions, collision_operator, diagnostics_bayesian, dispersion_solver, distribution_sampler, mode_decomposition
- Supporting modules: boundary_geometry, config, dos_plasma, numerical_core, vlasov_maxwell

## Rebuild approach

- Let `main.py` own the public behavior for 激光等离子体相互作用: 高阶有限差分与稳定性分析, while helper modules hold constants and small calculations.
- For 激光等离子体相互作用: 高阶有限差分与稳定性分析, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for uncertainty quantification summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `激光等离子体相互作用: 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
激光等离子体相互作用: 高阶有限差分与稳定性分析
High-Order Finite Difference for Laser-Plasma Interaction
配置创建完成: N_x=256, N_v=128, FD_order=4
阶段 1: 物理参数设置与完整性校验
参考密度 n_0 = 1.00e+25 m^-3
电子温度 T_e = 1.60e-16 J (1000 eV)
等离子体频率 omega_p0 = 1.78e+14 rad/s
趋肤深度 c/omega_p0 = 1.68e-06 m
热速度 v_th = 1.33e+07 m/s (v_th/c = 0.0442)
德拜长度 lambda_D = 7.43e-08 m
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
