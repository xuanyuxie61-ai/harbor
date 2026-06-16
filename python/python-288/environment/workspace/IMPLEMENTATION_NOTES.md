# Reconstruction Notes: Project 288

From the public behavior, the natural decomposition is magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: dg_parallel_solver, iterative_solver, legendre_fd
- Model layer: plasma_physics
- Diagnostics/reporting: output_diagnostics, perp_diffusion, stability_analysis, wavelet_diagnostics
- Supporting modules: divertor_geometry, parameter_scan, profile_tools

## Rebuild approach

- Handle `--help` and `--version` explicitly for 边界等离子体输运与偏滤器热负荷模拟系统; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 288; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 288, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `边界等离子体输运与偏滤器热负荷模拟系统`; hidden tests are not limited to them.

```text
边界等离子体输运与偏滤器热负荷模拟系统
Edge Plasma Transport & Divertor Heat Flux Simulation
High-Order Finite Difference & Stability Analysis
(小规模可复现实验)
[阶段 1/8] 物理参数初始化与基本物理量计算
Coulomb对数 ln(Λ)      = 13.089
电子-离子碰撞频率 ν_ei  = 1.517e+06 s^-1
Spitzer平行热传导 κ_∥   = 1.324e+00 W/(m·eV)
Bohm扩散系数 D_Bohm     = 1.179e+00 m²/s
离子Larmor半径 ρ_i      = 3.854695e-04 m
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
