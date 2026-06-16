# 计算宇宙学 — 重子声学振荡参数拟合

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in astrophysics simulation.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Program sketch

- 高阶有限差分与稳定性分析 (小规模可复现实验)
- Phase 00 :: 初始化物理常数与基准宇宙学
- ω_b          = 0.02237   (Ω_b = 0.04887)
- ω_m          = 0.14200   (Ω_m = 0.31019)

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Version probe:

```text
synthesis-python-259 1.0
```

## Observable anchors

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
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
