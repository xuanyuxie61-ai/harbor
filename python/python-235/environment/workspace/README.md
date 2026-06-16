# 计算高能物理: 异常事件检测与新物理信号搜索

This case exposes the observable surface of a research-style lattice field theory driver rather than its source.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Program sketch

- main.py — 计算高能物理: 异常事件检测与新物理信号搜索
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 统一入口, 零参数可运行.
- 科学问题:

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Version probe:

```text
synthesis-python-235 1.0
```

## Observable anchors

```text
计算高能物理: 异常事件检测与新物理信号搜索
高阶有限差分与稳定性分析 (小规模可复现实验)
Python 3.11.11, NumPy 1.26.4
Phase 0: 物理常数与实验配置
√s = 13 TeV
格点: N_x=128, h=0.1, m=2.0
BSM: mZ'=500.0 GeV, g'=0.3
事件: 200, 质量窗口: (60.0, 120.0)
Phase 1: 高阶有限差分稳定性分析
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
