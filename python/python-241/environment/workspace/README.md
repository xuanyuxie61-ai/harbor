# 核反应光学模型: 高阶有限差分与稳定性分析

The benchmark centers on a scientific driver in numerical-methods benchmark, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## Program sketch

- 核反应光学模型截面预测与高阶有限差分稳定性分析
- — 统一入口 (零参数可运行)
- 科学问题:
- 求解核子-重核散射的径向薛定谔方程, 使用复数光学模型势 (Woods-Saxon

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Version probe:

```text
synthesis-python-241 1.0
```

## Observable anchors

```text
核反应光学模型: 高阶有限差分与稳定性分析
Nuclear Optical Model: High-Order Finite Difference
and Stability Analysis
(DD 聚变中子能量)
计算方法: Numerov 四阶 + 传递矩阵 + 序贯检验
阶段 1: 光学模型势构造
靶核: 208Pb82
入射粒子: 中子
实验室能量: 14.1 MeV
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
