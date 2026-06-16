# ╔══════════════════════════════════════════════════════════════════════╗

The task is to rebuild a small Python implementation that behaves like a reference numerical-methods benchmark executable.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Program sketch

- main.py — 计算高能物理: 喷注聚类与 jet substructure 分析
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 统一入口, 零参数可运行.
- 融合 15 个种子项目的核心算法:

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Version probe:

```text
synthesis-python-223 1.0
```

## Observable anchors

```text
╔══════════════════════════════════════════════════════════════════════╗
║  计算高能物理: 喷注聚类与 jet substructure 分析                    ║
║  高阶有限差分与稳定性分析 — 小规模可复现实验                        ║
║  Computational HEP: Jet Clustering & Substructure                  ║
║  High-Order Finite Differences & Stability Analysis                 ║
╚══════════════════════════════════════════════════════════════════════╝
阶段 1: 多喷注事件生成
生成 10 个初始部分子
质心系能量: √s = 13000 GeV
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
