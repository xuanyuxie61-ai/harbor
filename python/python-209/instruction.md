# Reverse-Engineer Project 209: 随机参数 PDE 模型的不确定性量化

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `209`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 209's identity check should return:

```text
synthesis-python-209 1.0
```

This benchmark instance is a black-box reconstruction exercise for a compact uncertainty quantification program. Treat the default execution as the primary specification and preserve the order of its visible sections.

## Behavioral target

The public-facing subject of project 209 is:

```text
随机参数 PDE 模型的不确定性量化
```

Visible cues for the uncertainty quantification workflow:

```text
PROJECT 209: 随机参数 PDE 模型的不确定性量化
Uncertainty Quantification for Stochastic Parameter PDEs
Python 版本: 3.11.11
NumPy 版本: 1.26.4
Step 1: 随机场建模与 Karhunen-Loève 展开
空间点: 400, 相关长度: ℓ=0.3, 方差: σ²=1.0
```

The final workspace for project 209 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving 随机参数 PDE 模型的不确定性量化; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
