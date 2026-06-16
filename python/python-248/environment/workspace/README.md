# Galaxy Formation Hydrodynamics Test Problem

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in astrophysics simulation.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Scientific role

- PROJECT 248: 星系形成流体动力学模拟
- 博士级科学合成项目 — 高阶有限差分与稳定性分析（小规模可复现实验）
- 一、项目概述
- 本项目融合 **15 个种子科研项目** 的核心算法，围绕 **计算天体物理** 领域中的 **星系形成流体动力学模拟** 这一前沿博士级科学问题，构建了一个完整的数值模拟框架。

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Stable version text:

```text
synthesis-python-248 1.0
```

## Report signatures

```text
PROJECT_248 :: Galaxy Formation Hydrodynamics Test Problem
Domain: high-order finite differences + stability analysis
(small-scale reproducible experiment)
N     dx           L_inf (order 4)   L_inf (order 6)   rate_4  rate_6
[ok] all 16 Python modules imported successfully
High-order FD convergence check: f(x) = sin(2 pi x) on [0, 1]
32   3.125e-02         3.099e-04          2.553e-06      0.00    0.00
64   1.562e-02         1.943e-05          4.011e-08      4.00    5.99
128   7.812e-03         1.216e-06          6.276e-10      4.00    6.00
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
