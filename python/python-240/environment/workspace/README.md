# 重离子碰撞椭圆流与初始条件涨落建模

Project 240 asks for a faithful external clone of a synthesized uncertainty quantification report generator.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Observed behavior

- 主程序入口：重离子碰撞椭圆流与初始条件涨落建模
- Main entry point: Heavy-ion collision elliptic flow and initial condition
- fluctuation modeling with high-order finite differences and stability analysis.
- This program performs:

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Identity check:

```text
synthesis-python-240 1.0
```

## Transcript anchors

```text
重离子碰撞椭圆流与初始条件涨落建模
Heavy-Ion Collision: Elliptic Flow & Initial Condition Fluctuations
High-Order Finite Differences & Stability Analysis
1. 蒙特卡洛Glauber初始条件 (Monte Carlo Glauber)
碰撞系统: Au+Au, √s = 200 GeV
碰撞参数: b = 7.0 fm
每核子数: A = 197
高斯展宽: σ = 0.5 fm
事件 1: N_part = 85, N_coll = 55
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
