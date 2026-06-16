# 暗能量状态方程约束: 高阶有限差分与稳定性分析

The reference executable is the oracle for a reduced astrophysics simulation experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## Scientific role

- main.py -- 暗能量状态方程约束: 高阶有限差分与稳定性分析
- 统一入口 (零参数运行)
- Project 260: 计算宇宙学 -- 暗能量状态方程约束
- 高阶有限差分与稳定性分析 (小规模可复现实验)

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Stable version text:

```text
synthesis-python-260 1.0
```

## Report signatures

```text
暗能量状态方程约束: 高阶有限差分与稳定性分析
High-Order FD + von Neumann Stability for DE EoS
1. CPL 暗能量状态方程与 FLRW 背景
基准模型 (LCDM):
CPL 模型 (w0=-0.9, wa=-0.3):
w(z=0) = -0.9000
w(z=1) = -1.0500
Phantom crossing: True
Phantom crossing a: 0.6666666666666667
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
