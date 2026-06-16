# Reverse-Engineer Project 241: 核反应光学模型: 高阶有限差分与稳定性分析

## Workspace objective

You are in `/app/workspace` with a reference executable for project `241`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 241 as:

```text
synthesis-python-241 1.0
```

The benchmark centers on a scientific driver in numerical-methods benchmark, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## Public interface

The default run presents the following workflow title:

```text
核反应光学模型: 高阶有限差分与稳定性分析
```

Initial public cues for 核反应光学模型: 高阶有限差分与稳定性分析:

```text
核反应光学模型: 高阶有限差分与稳定性分析
Nuclear Optical Model: High-Order Finite Difference
and Stability Analysis
(DD 聚变中子能量)
计算方法: Numerov 四阶 + 传递矩阵 + 序贯检验
阶段 1: 光学模型势构造
```

Create source files for 核反应光学模型: 高阶有限差分与稳定性分析 and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for 核反应光学模型: 高阶有限差分与稳定性分析 must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
