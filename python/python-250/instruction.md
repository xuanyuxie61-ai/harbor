# Reverse-Engineer Project 250: 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析

## Build goal

You are in `/app/workspace` with a reference executable for project `250`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 250:

```text
synthesis-python-250 1.0
```

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## What must match

Project 250 should be rebuilt around:

```text
超新星爆发辐射流体模拟：高阶有限差分与稳定性分析
```

Useful observation anchors from the oracle:

```text
1. 球对称网格构造
最小 dr = 1.01e+05 cm,  最大 dr = 4.63e+07 cm
2. 物态、不透明度、离散纵标
物态: γ=1.667, μ=0.617, Y_e=0.42
单元数: 64,  r ∈ [1.00e+06, 5.00e+08] cm
不透明度: 10 × 8 切比雪夫节点
```

Finish by making a standalone workspace executable for 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 250, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
