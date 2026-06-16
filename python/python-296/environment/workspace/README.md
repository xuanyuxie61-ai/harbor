# Fast Ignition ICF

The benchmark centers on a scientific driver in computational plasma physics, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## Scientific role

- PROJECT 296: Fast Ignition ICF 电子束能量沉积高阶有限差分与稳定性分析
- > **博士级计算等离子体物理合成项目
- 一、项目概述
- 1.1 科学问题

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Stable version text:

```text
synthesis-python-296 1.0
```

## Report signatures

```text
PROJECT 296: Fast Ignition ICF
电子束能量沉积 高阶有限差分与稳定性分析
小规模可复现实验 (博士级)
Fast Ignition ICF : 电子束能量沉积高阶有限差分与稳定性分析
[阶段 1/14] 物理参数初始化...
临界密度 n_c      = 1.011e+27 /m^3
日冕密度 n_e      = 1.000e+25 /m^3
日冕温度 T_e      = 5.000e+03 eV
稠密芯密度        = 1.000e+31 /m^3
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
