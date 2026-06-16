# 边界等离子体输运与偏滤器热负荷模拟系统

Project 288 is framed as a cleanroom reproduction task around computational plasma physics.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## Observed behavior

- main.py - 边界等离子体输运与偏滤器热负荷模拟统一入口
- 本项目研究边界等离子体在SOL（Scrape-Off Layer）区域的输运过程，
- 使用高阶有限差分/DG方法求解平行与垂直方向的输运方程，
- 并通过稳定性分析确保数值格式的可靠性。

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Identity check:

```text
synthesis-python-288 1.0
```

## Transcript anchors

```text
边界等离子体输运与偏滤器热负荷模拟系统
Edge Plasma Transport & Divertor Heat Flux Simulation
High-Order Finite Difference & Stability Analysis
(小规模可复现实验)
[阶段 1/8] 物理参数初始化与基本物理量计算
Coulomb对数 ln(Λ)      = 13.089
电子-离子碰撞频率 ν_ei  = 1.517e+06 s^-1
Spitzer平行热传导 κ_∥   = 1.324e+00 W/(m·eV)
Bohm扩散系数 D_Bohm     = 1.179e+00 m²/s
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
