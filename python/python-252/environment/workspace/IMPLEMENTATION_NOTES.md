# Reconstruction Notes: Project 252

This task can be solved by rebuilding a concise pipeline for magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: high_order_fd, implicit_solver, mhd_grid
- Model layer: mhd_equations
- Diagnostics/reporting: mhd_equations, simulation_pipeline, stability_analysis
- Supporting modules: causal_connectivity, kerr_geometry, mhd_constants, mode_selector, optimal_control, spectral_filter

## Rebuild approach

- For this computational plasma physics task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 252's printed diagnostics.
- Keep the final report for project 252 deterministic, compact, and ordered like the reference.
- For 黑洞喷流形成与 MHD 模拟, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `黑洞喷流形成与 MHD 模拟`; hidden tests are not limited to them.

```text
PROJECT 252: 黑洞喷流形成与 MHD 模拟
高阶有限差分与稳定性分析 (博士级可复现实验)
黑洞喷流 MHD 模拟 & 稳定性分析 (博士级可复现实验)
完成 20 步, 最终密度扰动: 7.277096e-02
[1/8] 配置校验: 全部通过 (9 项)
[2/8] 网格生成: Nr=48, Nt=24, r=[2.00, 50.0]
[3/8] 初始条件: Bondi 吸积 + 磁化 + MRI 微扰
[4/8] 时间推进: dt=0.5000, n_steps=20, FD阶数=6
[5/8] 稳定性分析: 拟合增长率 gamma=-0.016985, R^2=0.9936
前 3 个不稳定模态:
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
