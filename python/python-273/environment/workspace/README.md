# 声子谱与热输运计算: 高阶有限差分与稳定性分析

Project 273 asks for a faithful external clone of a synthesized lattice field theory report generator.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Visible purpose

- main.py — 声子谱与热输运计算: 高阶有限差分与稳定性分析
- 统一入口, 零参数可运行。
- 项目融合 15 个种子项目的核心算法:
- 1. 1239_DurationModulatedDynamics -> slds_phonon.py (SLDS状态跟踪)

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Reference version:

```text
synthesis-python-273 1.0
```

## Reference cues

```text
声子谱与热输运计算: 高阶有限差分与稳定性分析
计算凝聚态物理 —— 博士级前沿数值实验
目标材料: FCC 铜 (Cu)
晶格常数: 3.615 Angstrom
原子质量: 63.546 amu
第1步: 晶格几何构建
FCC 超胞: 3x3x3 = 108 个原子
超胞边长: 10.845 Angstrom
近邻壳层结构:
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
