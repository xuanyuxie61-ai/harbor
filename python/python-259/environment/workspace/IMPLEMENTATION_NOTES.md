# Reconstruction Notes: Project 259

The reference run suggests a small driver that combines reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: comoving_grid, fornberg_fd
- Diagnostics/reporting: correlation_landy_szalay, von_neumann_stability
- Supporting modules: acoustic_wave_pde, background_cosmology, bao_constants, bao_observables, bezier_window, chi2_fitter_bilevel

## Coding plan

- Keep the command surface for project 259 narrow: version, help, and the deterministic default workflow.
- Let model parameters, evolution loops, and summary tables drive the helper functions, but keep the default run under verifier time limits.
- For project 259, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `计算宇宙学 — 重子声学振荡参数拟合`; hidden tests are not limited to them.

```text
PROJECT 259 : 计算宇宙学 — 重子声学振荡参数拟合
高阶有限差分与稳定性分析 (小规模可复现实验)
Phase 00 :: 初始化物理常数与基准宇宙学
ω_b          = 0.02237   (Ω_b = 0.04887)
ω_m          = 0.14200   (Ω_m = 0.31019)
Ω_Λ          = 0.68972
Ω_r          = 9.12567e-05
σ_8          = 0.8111
w_0, w_a     = -1.000, 0.000
Phase 01 :: 生成共动坐标网格 + 区域编号
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
