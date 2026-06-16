# 计算材料 — 相图计算与 CALPHAD 建模

The task is to rebuild a small Python implementation that behaves like a reference numerical-methods benchmark executable.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Scientific role

- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 统一入口: 零参数可运行。
- Fe-C 二元合金体系的完整 CALPHAD 工作流:
- 1. CVT 非均匀网格生成

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Stable version text:

```text
synthesis-python-278 1.0
```

## Report signatures

```text
PROJECT 278: 计算材料 — 相图计算与 CALPHAD 建模
高阶有限差分与稳定性分析 (小规模可复现实验)
Fe-C 二元合金体系
气体常数 R = 8.3145 J/(mol·K)
成分范围: [1.00e-08, 0.25]
温度范围: [700.0, 1800.0] K
随机种子: 278
Step 1: CVT 非均匀成分网格生成 (Lloyd 算法)
CVT 生成点数: 30
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
