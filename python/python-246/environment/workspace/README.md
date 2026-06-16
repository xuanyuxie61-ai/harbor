# 宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析

This case exposes the observable surface of a research-style astrophysics simulation driver rather than its source.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Observed behavior

- 宇宙大尺度结构 N 体模拟: 高阶有限差分与稳定性分析
- Computational Astrophysics — LSS N-body with High-Order FD & Stability Analysis
- 科学领域**: 计算天体物理 — 宇宙大尺度结构形成
- 1. 科学问题陈述

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Identity check:

```text
synthesis-python-246 1.0
```

## Transcript anchors

```text
#  宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
#  Computational Astrophysics: LSS N-body with High-Order FD
Stage 1: 加载宇宙学参数 (seed 1068 internalstate/config_utils)
盒子边长 L = 64.0 Mpc/h
宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析
Computational Astrophysics: LSS N-body with High-Order FD
网格 N = 16 (总 4096 个网格单元)
粒子数 N_p = 4096
(Ω_m, Ω_Λ, h) = (0.308, 0.692, 0.6781)
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
