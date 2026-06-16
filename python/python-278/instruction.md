# Reverse-Engineer Project 278: 计算材料 — 相图计算与 CALPHAD 建模

## Task target

You are in `/app/workspace` with a reference executable for project `278`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for 计算材料 — 相图计算与 CALPHAD 建模 is:

```text
synthesis-python-278 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference numerical-methods benchmark executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## External contract

Use the following label as the center of the reconstruction:

```text
计算材料 — 相图计算与 CALPHAD 建模
```

Reference lines worth preserving for project 278:

```text
PROJECT 278: 计算材料 — 相图计算与 CALPHAD 建模
高阶有限差分与稳定性分析 (小规模可复现实验)
Fe-C 二元合金体系
气体常数 R = 8.3145 J/(mol·K)
成分范围: [1.00e-08, 0.25]
温度范围: [700.0, 1800.0] K
```

For project 278, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 278, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
