# 阿尔芬波-高能粒子相互作用: 高阶有限差分与稳定性分析

This case exposes the observable surface of a research-style computational plasma physics driver rather than its source.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Scientific role

- main.py - 阿尔芬波-高能粒子相互作用模拟统一入口
- 本项目实现计算等离子体物理中的阿尔芬波与高能粒子相互作用，
- 使用高阶有限差分方法和稳定性分析。
- 物理问题描述:

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Stable version text:

```text
synthesis-python-290 1.0
```

## Report signatures

```text
阿尔芬波-高能粒子相互作用: 高阶有限差分与稳定性分析
PROJECT_290 - 计算等离子体物理博士级合成项目
Step 1: 等离子体参数配置
阿尔芬波-高能粒子相互作用模拟参数摘要
磁场强度 B₀ = 5.00 T
大半径 R₀ = 1.65 m
小半径 a = 0.50 m
反转比 ε = a/R₀ = 0.3030
安全因子范围 q₀ = 1.00 ~ q_a = 3.00
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
