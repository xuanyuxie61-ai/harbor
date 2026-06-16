# 计算高能物理: 量能器 Shower Profile 快速模拟

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Observed behavior

- main.py — 计算高能物理: 量能器 Shower Profile 快速模拟
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 统一入口: 零参数运行即可完成从参数输入到结果输出的完整流程。
- 科学问题:

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Identity check:

```text
synthesis-python-228 1.0
```

## Transcript anchors

```text
计算高能物理: 量能器 Shower Profile 快速模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
种子项目融合: 15 个科学计算项目 → 统一物理模拟框架
1. 实验参数配置
入射粒子: electron, E0 = 10000.0 MeV = 10.0 GeV
主材料: PbWO4
辐射长度 X0 = 0.8900 cm
临界能量 Ec = 7.97 MeV
Molière 半径 R_M = 2.3674 cm
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
