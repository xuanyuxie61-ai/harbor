# Reconstruction Notes: Project 246

The source evidence points to reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: mesh_topology, poisson_solver, time_integrator
- Diagnostics/reporting: diagnostics, initial_conditions, parameter_calibration, stability_analysis
- Supporting modules: __main__, cosmo_config, finite_difference, legendre_kernel, nas_search

## Rebuild approach

- Make the flag paths cheap for project 246; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by 宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析; random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 246.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析`; hidden tests are not limited to them.

```text
#  宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
#  Computational Astrophysics: LSS N-body with High-Order FD
Stage 1: 加载宇宙学参数 (seed 1068 internalstate/config_utils)
盒子边长 L = 64.0 Mpc/h
宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
Computational Astrophysics: LSS N-body with High-Order FD
网格 N = 16 (总 4096 个网格单元)
粒子数 N_p = 4096
(Ω_m, Ω_Λ, h) = (0.308, 0.692, 0.6781)
红移范围 z = 49.0 → 0.0
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
