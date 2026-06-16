# 二维材料异质结能带对齐

Project 284 asks for a faithful external clone of a synthesized numerical-methods benchmark report generator.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Scientific role

- PROJECT_284: 二维材料异质结能带对齐 — 高阶有限差分与稳定性分析
- 博士级合成说明文档
- 一、科学问题背景
- 1.1 前沿科学问题

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Stable version text:

```text
synthesis-python-284 1.0
```

## Report signatures

```text
二维材料异质结能带对齐
高阶有限差分与稳定性分析
(小规模可复现实验)
第 1 部分: 材料参数与异质结拓扑
[材料参数]
MoS2  : Eg(300K)=1.864eV, a_B=15.79Å, E_bind=60.8meV, chi=4.00eV
WSe2  : Eg(300K)=1.679eV, a_B=25.64Å, E_bind=31.9meV, chi=4.20eV
MoSe2 : Eg(300K)=1.578eV, a_B=14.85Å, E_bind=69.3meV, chi=4.15eV
WS2   : Eg(300K)=1.936eV, a_B=22.42Å, E_bind=40.1meV, chi=3.90eV
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
