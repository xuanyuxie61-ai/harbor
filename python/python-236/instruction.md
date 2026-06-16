# Reverse-Engineer Project 236: 格点 QCD: 强子谱关联函数拟合

## Task target

You are in `/app/workspace` with a reference executable for project `236`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for 格点 QCD: 强子谱关联函数拟合 is:

```text
synthesis-python-236 1.0
```

The program under observation is a deterministic lattice field theory demonstrator packaged as a single executable. Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

## External contract

Use the following label as the center of the reconstruction:

```text
格点 QCD: 强子谱关联函数拟合
```

Reference lines worth preserving for project 236:

```text
格点 QCD: 强子谱关联函数拟合
高阶有限差分与稳定性分析 (小规模可复现实验)
1. 格点几何构建
格点: Ls=4, Lt=12, V=768
形状: (4, 4, 4, 12)
距离矩阵: max=6.928, mean=3.825
```

For project 236, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 236, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
