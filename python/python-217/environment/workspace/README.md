# 鲁棒优化与不确定约束 —— 博士级科学计算项目 217

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Program sketch

- 项目概述
- 科学问题
- 模拟移动床色谱是一种连续分离技术，广泛应用于制药、化工等领域。其优化面临以下挑战：
- 1. **参数不确定性**：扩散系数、流速、吸附等温线参数存在测量误差和空间变化

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Version probe:

```text
synthesis-python-217 1.0
```

## Observable anchors

```text
步骤 1: 构造不确定性集合
椭球集合维度: 4
样本数量: 20
最坏情况线性目标: 2.4928
鲁棒优化与不确定约束 —— 博士级科学计算项目 217
应用：模拟移动床色谱过程的鲁棒优化
样本均值: [0.45875611 0.44097848 0.46637004 0.44288446]
超球正象限样本: [0.01397656 0.37183484 0.08101746 0.77924355]
非中心 t CDF(2.0, df=10, delta=0.5) = 0.7182, ifault=0
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
