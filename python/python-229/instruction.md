# Reverse-Engineer Project 229: 计算高能物理 — 探测器响应矩阵与 unfolding 反演

## Workspace objective

You are in `/app/workspace` with a reference executable for project `229`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 229 as:

```text
synthesis-python-229 1.0
```

Project 229 asks for a faithful external clone of a synthesized numerical-methods benchmark report generator. The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

## Public interface

The default run presents the following workflow title:

```text
计算高能物理 — 探测器响应矩阵与 unfolding 反演
```

Initial public cues for 计算高能物理 — 探测器响应矩阵与 unfolding 反演:

```text
PROJECT_229: 计算高能物理
探测器响应矩阵与 unfolding 反演
高阶有限差分与稳定性分析 (小规模可复现实验)
1. 相空间网格与能量 bin 构造
动量空间网格: 半径 = 50.0 GeV, n_per_axis = 6
网格点数: 实际 1189, 解析估计 1150
```

Create source files for 计算高能物理 — 探测器响应矩阵与 unfolding 反演 and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for 计算高能物理 — 探测器响应矩阵与 unfolding 反演 must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
