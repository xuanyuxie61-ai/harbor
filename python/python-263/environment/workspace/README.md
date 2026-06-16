# Coronal Heating & Solar Wind Acceleration

The benchmark centers on a scientific driver in numerical-methods benchmark, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## Workflow outline

- PROJECT 263 — 计算太阳物理：日冕加热与太阳风加速
- 项目概述
- 本项目是一个面向前沿科学问题的博士级计算太阳物理研究平台，聚焦**日冕加热机制**与**太阳风加速过程**，采用**高阶有限差分方法**并系统开展**数值稳定性分析**。整个项目从小规模可复现实验出发，融合了 15 个种子项目的核心算法，构建了一个从光球到太阳风的多尺度耦合...
- 一、原项目到科学问题的映射

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Version output:

```text
synthesis-python-263 1.0
```

## Stable landmarks

```text
PROJECT 263 - Coronal Heating & Solar Wind Acceleration
High-Order Finite Differences with Stability Analysis
Python 3.11.11, NumPy 1.26.4
Random seed: 263
Corona loop grid (nodes=64)
n_nodes                 : 64
h_min                   :   1.391037e+06
h_max                   :   4.677348e+06
h_mean                  :   3.145125e+06
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
