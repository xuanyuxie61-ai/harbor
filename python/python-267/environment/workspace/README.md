# 拓扑绝缘体边界态高阶有限差分求解器

The task is to rebuild a small Python implementation that behaves like a reference spectral physics executable.

The program reads like a research demonstration: it sets up operators, spectra, and formatted numerical landmarks, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## Visible purpose

- main.py — 拓扑绝缘体边界态高阶有限差分求解器 (统一入口)
- 计算凝聚态: 拓扑绝缘体边界态数值求解
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 本项目实现 BHZ (Bernevig-Hughes-Zhang) 模型的实空间有限差分离散化,

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Reference version:

```text
synthesis-python-267 1.0
```

## Reference cues

```text
拓扑绝缘体边界态高阶有限差分求解器
Topological Insulator Boundary State FD Solver
计算凝聚态: 拓扑绝缘体边界态数值求解
高阶有限差分与稳定性分析 (小规模可复现实验)
NumPy 版本: 1.26.4
████████████████████████████████████████████████████████████████████████
阶段 1: 高阶有限差分模板构造与验证
有限差分模板系数汇总
一阶导数系数 c_j:
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
