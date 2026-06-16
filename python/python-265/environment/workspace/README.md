# Cosmic-Ray Transport in the Heliosphere

The visible binary represents a synthesized numerical-methods benchmark workflow with a deterministic command-line transcript.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. The hidden tests may look beyond the examples, so preserve the general report shape rather than only one line.

## Program sketch

- PROJECT 265 — 宇宙线在日球层中的传播：高阶有限差分与稳定性分析
- 一、项目概述
- 本项目面向**计算空间物理**前沿方向 —— **宇宙线在日球层中的传播与扩散**。具体科学问题为：
- 项目的代码架构、变量命名、物理量纲全部与宇宙线输运问题深度耦合（如 `kappa_par`、`D_mumu`、`parker_imf`、`focusing_length`、`BarenblattCosmicRay` 等），绝非通用数值库的换皮。

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

Version probe:

```text
synthesis-python-265 1.0
```

## Observable anchors

```text
PROJECT 265 -- Cosmic-Ray Transport in the Heliosphere
High-order finite differences & stability analysis
EXPERIMENT 1 -- heliospheric mesh & Parker IMF
radial grid : Nr = 48, r_min = 0.050 AU, r_max = 120.0 AU
mu grid     : Nmu = 16, min = -0.9950, max = +0.9950
EXPERIMENT 2 -- high-order finite-difference stencils
upwind1 max error = 9.899e-14
upwind2 max error = 3.230e-14
compact4 max error = 1.176e-14
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.
