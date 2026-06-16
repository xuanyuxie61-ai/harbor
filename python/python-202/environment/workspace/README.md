# 不确定性量化: 随机配置方法 (Stochastic Collocation UQ)

This case exposes the observable surface of a research-style uncertainty quantification driver rather than its source.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## What to reproduce

- 不确定性量化: 随机配置方法 — 统一入口
- Uncertainty Quantification via Stochastic Collocation Methods
- 本项目实现了一个完整的随机配置法框架, 用于求解参数化 PDE 中的
- 不确定性量化问题。核心科学问题:

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Expected `--version` text:

```text
synthesis-python-202 1.0
```

## Output cues

```text
不确定性量化: 随机配置方法 (Stochastic Collocation UQ)
PROJECT 202 — 博士级科研代码合成项目
NumPy 版本: 1.26.4
随机种子: 42 (可复现)
第一部分: Smolyak 稀疏网格构造
维度 D=3, Level q=4
多指标数: 19
非零系数数: 19
稀疏网格原始点数: 189
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
