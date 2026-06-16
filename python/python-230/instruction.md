# Reverse-Engineer Project 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定

## Task target

You are in `/app/workspace` with a reference executable for project `230`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for 计算高能物理 Profile Likelihood 与 CLs 上限设定 is:

```text
synthesis-python-230 1.0
```

The benchmark centers on a scientific driver in numerical-methods benchmark, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## External contract

Use the following label as the center of the reconstruction:

```text
计算高能物理 Profile Likelihood 与 CLs 上限设定
```

Reference lines worth preserving for project 230:

```text
#  PROJECT 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定
#  高阶有限差分与稳定性分析 (小规模可复现实验)
#  融合 15 个科研种子项目的核心算法
模块 1: 物理模型构建 (H -> γγ 双光子搜索)
PROJECT 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定
高阶有限差分与稳定性分析 (小规模可复现实验)
```

For project 230, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 230, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
