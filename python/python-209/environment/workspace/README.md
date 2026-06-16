# 随机参数 PDE 模型的不确定性量化

This benchmark instance is a black-box reconstruction exercise for a compact uncertainty quantification program.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## Workflow outline

- main.py — 统一入口: 随机参数 PDE 的不确定性量化全流水线
- 本项目解决的前沿科学问题:
- 随机参数椭圆 PDE 的不确定性量化 (Uncertainty Quantification)
- 其中 a(x,ω) = exp(Y(x,ω)) 为对数正态随机扩散系数,

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Version output:

```text
synthesis-python-209 1.0
```

## Stable landmarks

```text
PROJECT 209: 随机参数 PDE 模型的不确定性量化
Uncertainty Quantification for Stochastic Parameter PDEs
Python 版本: 3.11.11
NumPy 版本: 1.26.4
Step 1: 随机场建模与 Karhunen-Loève 展开
空间点: 400, 相关长度: ℓ=0.3, 方差: σ²=1.0
KL 模态数: 15
能量捕获比: 0.988455
前3个特征值: [130.36380529  67.43047606  67.43047606]
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
