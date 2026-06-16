# 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis

The reference executable is the oracle for a reduced lattice field theory experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## Program sketch

- PROJECT 271 — 一维横场 Ising 模型量子相变的高阶有限差分/有限尺寸标度/稳定性分析
- > 计算凝聚态 · 博士级合成项目 · 15 个种子项目核心算法融合
- >
- 0. 项目定位

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Version probe:

```text
synthesis-python-271 1.0
```

## Observable anchors

```text
(small-scale reproducible experiment, Python)
Stage 1 : NAS-style benchmark kernels
btrix                  residual = 1.619e-16
cfft2d                 residual = 1.361e-15
PROJECT 271 : 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis
cholsky                residual = 1.349e-16
emit                   residual = 5.554e-16
dct                    residual = 8.882e-16
su2_chain              residual = 1.333e-15
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
