# 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合

The reference executable is the oracle for a reduced computational plasma physics experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## Workflow outline

- main.py — 计算高能物理: 粒子径迹重建与 Kalman 滤波拟合
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 统一入口，零参数可运行。
- 本项目融合 15 个种子项目的核心算法，构建完整的带电粒子径迹重建流水线:

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Version output:

```text
synthesis-python-227 1.0
```

## Stable landmarks

```text
计算高能物理: 粒子径迹重建与 Kalman 滤波拟合
高阶有限差分与稳定性分析 (小规模可复现实验)
PROJECT_227 — 博士级合成项目
1. 探测器配置
TrackingDetector: 8 layers
Radius range: [33.0, 580.0] mm
Total material budget: 0.2400 X₀
Layer 0: r=33.0mm, t/X₀=0.0200, σ_r=10.0μm, σ_z=115.0μm, mat=silicon
Layer 1: r=50.5mm, t/X₀=0.0200, σ_r=10.0μm, σ_z=115.0μm, mat=silicon
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
