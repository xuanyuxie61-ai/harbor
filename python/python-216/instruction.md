# Reverse-Engineer Project 216: Stochastic Helmholtz Optimization via SAA

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `216`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for Stochastic Helmholtz Optimization via SAA is:

```text
synthesis-python-216 1.0
```

The reference executable is the oracle for a reduced scientific computing experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Observable behavior

Reconstruct the command-line behavior for this scientific computing target:

```text
Stochastic Helmholtz Optimization via SAA
```

Start by matching these lines:

```text
阶段 1: 伪随机数发生器初始化 (seed: 763_middle_square + 1393_vin)
主种子 = 2160607
校验 seed 完整性: True
前 10 个 U[0,1) 样本:
PROJECT 216: Stochastic Helmholtz Optimization via SAA
随机亥姆霍兹方程的样本平均近似优化
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 216 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
