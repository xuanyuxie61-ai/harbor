# Reverse-Engineer Project 227: 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `227`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 227's identity check should return:

```text
synthesis-python-227 1.0
```

The reference executable is the oracle for a reduced computational plasma physics experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Behavioral target

The public-facing subject of project 227 is:

```text
计算高能物理: 粒子径迹重建与 Kalman 滤波拟合
```

Visible cues for the computational plasma physics workflow:

```text
计算高能物理: 粒子径迹重建与 Kalman 滤波拟合
高阶有限差分与稳定性分析 (小规模可复现实验)
PROJECT_227 — 博士级合成项目
1. 探测器配置
TrackingDetector: 8 layers
Radius range: [33.0, 580.0] mm
```

The final workspace for project 227 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
