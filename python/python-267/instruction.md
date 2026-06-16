# Reverse-Engineer Project 267: 拓扑绝缘体边界态高阶有限差分求解器

## Black-box target

You are in `/app/workspace` with a reference executable for project `267`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-267 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference spectral physics executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## Reference behavior

The binary's scientific topic is:

```text
拓扑绝缘体边界态高阶有限差分求解器
```

The transcript begins to define the target through:

```text
拓扑绝缘体边界态高阶有限差分求解器
Topological Insulator Boundary State FD Solver
计算凝聚态: 拓扑绝缘体边界态数值求解
高阶有限差分与稳定性分析 (小规模可复现实验)
NumPy 版本: 1.26.4
████████████████████████████████████████████████████████████████████████
```

Your rebuilt spectral physics program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 拓扑绝缘体边界态高阶有限差分求解器; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
