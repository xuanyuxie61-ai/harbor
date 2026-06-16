# 分布式 ADMM 多物理场 PDE 约束优化框架

This benchmark instance is a black-box reconstruction exercise for a compact scientific computing program.

The program reads like a research demonstration: it sets up configuration values, numerical kernels, and formatted report sections, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## What to reproduce

- main.py — 分布式 ADMM 多物理场 PDE 约束优化框架
- 统一入口 (零参数可运行)
- 科学问题:
- 将多种物理场 (反应扩散、神经动力学、大气化学、流体力学)

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Expected `--version` text:

```text
synthesis-python-220 1.0
```

## Output cues

```text
#  分布式 ADMM 多物理场 PDE 约束优化框架
#  科学领域: 数学优化 — 分布式优化与 ADMM 方法
#  博士级科学计算合成项目 (PROJECT 220)
实验 1: 分布式 Poisson 反问题 (扩散系数辨识)
分布式 ADMM 多物理场 PDE 约束优化框架
科学领域: 数学优化 — 分布式优化与 ADMM 方法
博士级科学计算合成项目 (PROJECT 220)
网格节点数: 1601
单元数: 512
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
