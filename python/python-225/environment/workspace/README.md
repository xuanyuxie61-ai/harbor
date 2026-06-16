# 暗物质直接探测 recoil spectrum 建模

The program under observation is a deterministic uncertainty quantification demonstrator packaged as a single executable.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Exact internal algorithms are less important than externally stable scientific summaries and return codes.

## Visible purpose

- PROJECT 225: 暗物质直接探测 recoil spectrum 高阶有限差分建模与稳定性分析
- > **计算高能物理 · 博士级合成项目
- >
- 一、项目概述

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

Reference version:

```text
synthesis-python-225 1.0
```

## Reference cues

```text
PROJECT 225: 暗物质直接探测 recoil spectrum 建模
High-order finite difference & stability analysis
计算高能物理 · 博士级合成项目
Step 1: 浮点机器常数测定 (Malcolm-Gentleman-Maroney)
[OK] 双精度 eps = 9.997e-13, 有效位 4
Step 2: 探测器模块几何 (GRF 图 I/O)
[OK] 构建探测器图: 20 个模块
坐标范围: r ∈ [0.50, 1.70]
[OK] GRF I/O: 节点数 = 20, 边数 = 62
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.
