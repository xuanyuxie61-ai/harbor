# Reverse-Engineer Project 234: 博士级 B 物理衰变链重建与 CP 破坏分析

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `234`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 博士级 B 物理衰变链重建与 CP 破坏分析 is:

```text
synthesis-python-234 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference scientific computing executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## Observable behavior

Reconstruct the command-line behavior for this scientific computing target:

```text
博士级 B 物理衰变链重建与 CP 破坏分析
```

Start by matching these lines:

```text
博士级 B 物理衰变链重建与 CP 破坏分析
高阶有限差分与稳定性分析 (小规模可复现实验)
统一入口: 零参数可运行
Python: 3.11.11
NumPy:  1.26.4
[1] CKM 矩阵与么正三角形
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 234 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
