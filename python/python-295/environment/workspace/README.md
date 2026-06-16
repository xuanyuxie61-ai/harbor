# 惯性约束聚变内爆对称性模拟

Project 295 asks for a faithful external clone of a synthesized fusion and radiation transport report generator.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Program sketch

- 惯性约束聚变内爆对称性模拟 — 高阶有限差分与稳定性分析
- 博士级合成项目说明
- 项目概述
- 种子项目到科学问题的映射

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Version probe:

```text
synthesis-python-295 1.0
```

## Observable anchors

```text
惯性约束聚变内爆对称性模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
Inertial Confinement Fusion Implosion Symmetry Simulation
High-Order Finite Difference & Stability Analysis
阶段 1: ICF 物理参数设置
多方指数 γ = 1.6667
平均粒子质量 = 4.1816e-24 g (DT)
靶丸半径 = 1.0000e-02 cm
初始燃料密度 = 2.5000e-01 g/cm³
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
