# 博士级 B 物理衰变链重建与 CP 破坏分析

The task is to rebuild a small Python implementation that behaves like a reference scientific computing executable.

The program reads like a research demonstration: it sets up configuration values, numerical kernels, and formatted report sections, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Observed behavior

- 统一入口: 博士级 B 物理衰变链重建与 CP 破坏分析的高阶有限差分及稳定性研究。
- 零参数可运行。
- 完整流程:
- [1] 加载 B 物理常数与 CKM 矩阵参数;

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Identity check:

```text
synthesis-python-234 1.0
```

## Transcript anchors

```text
博士级 B 物理衰变链重建与 CP 破坏分析
高阶有限差分与稳定性分析 (小规模可复现实验)
统一入口: 零参数可运行
Python: 3.11.11
NumPy:  1.26.4
[1] CKM 矩阵与么正三角形
Wolfenstein 参数: lambda=0.22650, A=0.7900, rhobar=0.1590, etabar=0.3500
CKM 矩阵 (近似到 O(lambda^4)):
|V_{u}|:  0.97402  exp(i 0.0000)  0.22650  exp(i 0.0000)  0.00353  exp(i -1.1444)
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
