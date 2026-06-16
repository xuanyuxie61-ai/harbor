# Reverse-Engineer Project 272: Weyl 半金属 Berry Curvature 计算系统

## Task target

You are in `/app/workspace` with a reference executable for project `272`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for Weyl 半金属 Berry Curvature 计算系统 is:

```text
synthesis-python-272 1.0
```

This is a ProgramBench-style task: infer and reproduce a spectral physics CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## External contract

Use the following label as the center of the reconstruction:

```text
Weyl 半金属 Berry Curvature 计算系统
```

Reference lines worth preserving for project 272:

```text
Weyl 半金属 Berry Curvature 计算系统
高阶有限差分与稳定性分析 (小规模可复现实验)
第 1 部分：Weyl Hamiltonian 构建与能带结构
本征值: E_± = -0.2450, 0.3550
[Hamiltonian] Γ 点 Hamiltonian 矩阵:
H(Γ) =
```

For project 272, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 272, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
