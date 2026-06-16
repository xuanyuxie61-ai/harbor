# 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## What to reproduce

- 超新星爆发辐射流体模拟：高阶有限差分与稳定性分析
- 统一入口 (零参数). 运行流程：
- 1. 设置物理常数和球对称网格
- 2. 构造前身星初始剖面 (解析近似)

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Expected `--version` text:

```text
synthesis-python-250 1.0
```

## Output cues

```text
1. 球对称网格构造
最小 dr = 1.01e+05 cm,  最大 dr = 4.63e+07 cm
2. 物态、不透明度、离散纵标
物态: γ=1.667, μ=0.617, Y_e=0.42
单元数: 64,  r ∈ [1.00e+06, 5.00e+08] cm
不透明度: 10 × 8 切比雪夫节点
κ(T=5e9, ρ=1e9) = 9.766e+18 cm^2/g
S_N 求积: 18 方向, 权重和 = 1.5708e+00
S_4 level-symmetric: 8 方向
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
