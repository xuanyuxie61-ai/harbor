# 随机热传导方程的不确定性量化与置信/预测带构建

Project 207 asks for a faithful external clone of a synthesized uncertainty quantification report generator.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Visible purpose

- main.py — 统一入口: 随机热传导方程的不确定性量化与置信/预测带构建
- 本项目解决的前沿科学问题
- 对于具有随机扩散系数的一维热传导方程:
- ∂u/∂t = ∂/∂x [ κ(x,ω) · ∂u/∂x ],  x ∈ [0,L], t ∈ [0,T]

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Reference version:

```text
synthesis-python-207 1.0
```

## Reference cues

```text
随机热传导方程的不确定性量化与置信/预测带构建
Scientific Problem: UQ for Stochastic Parabolic PDEs
Focus: Confidence Intervals & Prediction Intervals
SHE-UQ 问题参数摘要
空间域:        [0, 1.0] m,  nx=51,  dx=0.020000
时间域:        [0, 0.5] s,  nt=101,  dt=0.005000
基准扩散系数:  κ₀ = 1.0000e-02 m²/s
随机扰动强度:  σ_κ = 3.0000e-03 m²/s
相关长度:      ℓ = 0.1000 m
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
