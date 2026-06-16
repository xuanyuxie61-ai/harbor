# main.py  --  贝叶斯模型校准统一入口

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Scientific role

- main.py  --  贝叶斯模型校准统一入口
- 科学问题:
- Gray-Scott 反应扩散系统的贝叶斯参数校准.
- 给定稀疏带噪观测 y_obs, 推断反应速率 (f, k)

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Stable version text:

```text
synthesis-python-206 1.0
```

## Report signatures

```text
阶段 1: 数值基底初始化
阶段 2: 合成观测数据生成
真实参数: {'Du': 0.16, 'Dv': 0.08, 'f': 0.04, 'k': 0.06}
sim 观测 (n=8): 0.0282, 0.0036, 0.0276, 0.0187 ...
PROJECT 206: 贝叶斯模型校准
Gray-Scott 反应扩散系统的不确定性量化
[NumericalBase] ibeta=2, it=53, irnd=1
real 观测 (n=8): 0.0392, 0.0089, 0.0076, 0.0095 ...
阶段 3: 先验几何构建
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
