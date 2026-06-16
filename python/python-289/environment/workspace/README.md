# 1D slab gyrokinetic turbulence simulator

The task is to rebuild a small Python implementation that behaves like a reference spectral physics executable.

The program reads like a research demonstration: it sets up operators, spectra, and formatted numerical landmarks, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Program sketch

- PROJECT 289 — 1D 平板位形回旋动力学湍流模拟器
- > **方法**：高阶有限差分 + 隐式时间推进 + 线性稳定性分析（小规模可复现实验）
- 1. 项目概述
- 2. 核心科学公式

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Version probe:

```text
synthesis-python-289 1.0
```

## Observable anchors

```text
PROJECT 289 -- 1D slab gyrokinetic turbulence simulator
Domain:  计算等离子体 / 湍流输运 / gyrokinetic 模拟
高阶有限差分 + 稳定性分析 (小规模可复现实验)
Step 0 -- Equilibrium construction
ion sound speed  c_s      = 2.1885e+05 m/s
ion cyclotron    omega_ci = 1.1974e+08 rad/s
gyroradius       rho_s    = 1.8278e-03 m
ITG threshold (rough)  (R/L_Ti)_c ~ 1111.111
n0 profile (5 pts)  = [1.    0.975 0.95  0.925 0.9  ]
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
