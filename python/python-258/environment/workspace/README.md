# 弱引力透镜质量重建: 高阶有限差分与稳定性分析

The program under observation is a deterministic astrophysics simulation demonstrator packaged as a single executable.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. Exact internal algorithms are less important than externally stable scientific summaries and return codes.

## Observed behavior

- main.py — 弱引力透镜质量重建: 高阶有限差分与稳定性分析
- 统一入口, 零参数可运行。
- 本项目融合 15 个种子项目的核心算法, 在计算宇宙学领域实现:
- 核心物理:

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

Identity check:

```text
synthesis-python-258 1.0
```

## Transcript anchors

```text
弱引力透镜质量重建: 高阶有限差分与稳定性分析
Weak Lensing Mass Reconstruction with High-Order FD
PROJECT 258 — 计算宇宙学博士级可复现实验
阶段 1: 宇宙学距离与临界密度计算
透镜红移 z_L = 0.3
源红移   z_S = 1.0
Σ_crit = 2.750e+15 M_sun/Mpc²
共动距离 χ(z_S) = 3397.82 Mpc
E(z) at z=[0.  0.5 1.  2. ]: [1.         1.3224362  1.79083779 3.0327875 ]
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.
