# 计算凝聚态 DFT 能带计算

Project 266 is framed as a cleanroom reproduction task around uncertainty quantification.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## Scientific role

- PROJECT 266 — 计算凝聚态: 密度泛函理论能带计算
- 高阶有限差分与稳定性分析（小规模可复现实验）
- 一、项目概述
- 在 1D 周期晶格（Mathieu 势模型）上，实现了从实空间离散化、Kohn-Sham 方程求解、自洽场迭代、

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Stable version text:

```text
synthesis-python-266 1.0
```

## Report signatures

```text
PROJECT 266: 计算凝聚态 DFT 能带计算
高阶有限差分与稳定性分析 (小规模可复现实验)
Phase 1: 物理常数与晶格设置
晶格常数 a = 10.0000 Bohr = 5.2920 Å
倒格矢 G = 0.628319 1/Bohr
BZ 边界 k_max = 0.314159 1/Bohr
1 Hartree = 27.211386 eV
k 点数: 32
k 范围: [-0.3043, 0.3043] 1/Bohr
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
