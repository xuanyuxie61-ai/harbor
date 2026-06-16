# Reverse-Engineer Project 296: Fast Ignition ICF

## Task target

You are in `/app/workspace` with a reference executable for project `296`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for Fast Ignition ICF is:

```text
synthesis-python-296 1.0
```

The benchmark centers on a scientific driver in computational plasma physics, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## External contract

Use the following label as the center of the reconstruction:

```text
Fast Ignition ICF
```

Reference lines worth preserving for project 296:

```text
PROJECT 296: Fast Ignition ICF
电子束能量沉积 高阶有限差分与稳定性分析
小规模可复现实验 (博士级)
Fast Ignition ICF : 电子束能量沉积高阶有限差分与稳定性分析
[阶段 1/14] 物理参数初始化...
临界密度 n_c      = 1.011e+27 /m^3
```

For project 296, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 296, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
