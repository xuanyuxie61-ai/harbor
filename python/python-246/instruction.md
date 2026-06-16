# Reverse-Engineer Project 246: 宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `246`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析 is:

```text
synthesis-python-246 1.0
```

This case exposes the observable surface of a research-style astrophysics simulation driver rather than its source. Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

## Observable behavior

Reconstruct the command-line behavior for this astrophysics simulation target:

```text
宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
```

Start by matching these lines:

```text
#  宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
#  Computational Astrophysics: LSS N-body with High-Order FD
Stage 1: 加载宇宙学参数 (seed 1068 internalstate/config_utils)
盒子边长 L = 64.0 Mpc/h
宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
Computational Astrophysics: LSS N-body with High-Order FD
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 246 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
