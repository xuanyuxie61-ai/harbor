# Non-Hermitian Spectral Structure & Exceptional

This benchmark instance is a black-box reconstruction exercise for a compact spectral physics program.

The program reads like a research demonstration: it sets up operators, spectra, and formatted numerical landmarks, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## Workflow outline

- main.py - PROJECT_275 统一入口
- 非厄米 SSH 模型的谱结构、例外点与高阶有限差分稳定性分析.
- 流程:
- 1. 构造非厄米 SSH 哈密顿量

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Version output:

```text
synthesis-python-275 1.0
```

## Stable landmarks

```text
PROJECT 275: Non-Hermitian Spectral Structure & Exceptional
Points - High-Order Finite Difference & Stability Analysis
1. Non-Hermitian SSH Hamiltonian Construction
SSH parameters: t1=1.0, t2=0.6, gamma=0.35, k=1.0472
H_SSH(k) =
[[0. +0.35j       1.3-0.51961524j]
[1.3+0.51961524j 0. -0.35j      ]]
Discriminant Delta(k) = 7.350000 + 0.000000i
|Delta| = 7.350000e+00
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
