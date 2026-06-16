# 强关联 Hubbard 模型量子蒙特卡洛模拟系统

This case exposes the observable surface of a research-style lattice field theory driver rather than its source.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## What to reproduce

- PROJECT_268 博士级合成说明
- 计算凝聚态: 强关联 Hubbard 模型量子蒙特卡洛
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 一、科学问题与物理背景

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Expected `--version` text:

```text
synthesis-python-268 1.0
```

## Output cues

```text
强关联 Hubbard 模型量子蒙特卡洛模拟系统
高阶有限差分与稳定性分析 (小规模可复现实验)
PROJECT_268 - 博士级合成项目
阶段 1: 晶格几何与布里渊区构造
融合种子项目: [02] wedge_grid, [05] triangle_to_fem,
[08] disk_grid, [11] triangulation, [15] metis_graph
>> 1.1 三角晶格构造 (4x4 簇)
簇尺寸: 4 x 4
总格点数: Ns = 16
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
