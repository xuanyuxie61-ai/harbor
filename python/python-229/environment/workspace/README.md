# 计算高能物理 — 探测器响应矩阵与 unfolding 反演

Project 229 asks for a faithful external clone of a synthesized numerical-methods benchmark report generator.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Program sketch

- PROJECT_229 : 计算高能物理 — 探测器响应矩阵与 unfolding 反演
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 统一入口, 零参数运行.
- 流程:

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Version probe:

```text
synthesis-python-229 1.0
```

## Observable anchors

```text
PROJECT_229: 计算高能物理
探测器响应矩阵与 unfolding 反演
高阶有限差分与稳定性分析 (小规模可复现实验)
1. 相空间网格与能量 bin 构造
动量空间网格: 半径 = 50.0 GeV, n_per_axis = 6
网格点数: 实际 1189, 解析估计 1150
探测器接受 (pT>0.5 GeV, |η|<2.5): 1176/1189
能量 bin: 15 (true) × 15 (rec), E ∈ [1.0, 100.0] GeV
2. 探测器网格与富化 (来自 789 + 1353)
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
