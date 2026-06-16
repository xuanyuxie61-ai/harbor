# 计算材料基因组高通量筛选框架

This case exposes the observable surface of a research-style fusion and radiation transport driver rather than its source.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Visible purpose

- main.py - 统一入口
- 高阶有限差分与稳定性分析的材料基因组高通量筛选框架。
- 科学问题:
- 在掺杂 LLZO (Li7La3Zr2O12) 固态电解质中, 通过高通量计算筛选

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Reference version:

```text
synthesis-python-279 1.0
```

## Reference cues

```text
PROJECT 279: 计算材料基因组高通量筛选框架
高阶有限差分与稳定性分析 (LLZO 固态电解质)
Phase 1: 晶体结构编码与描述符生成
LLZO 立方晶格常数: a = 12.970 Å
代表性原子数: 13
晶胞体积: V = 2181.83 Å³
度量张量 det(G) = 4.7604e-54
倒格矢度量张量 G* (0,0) = 2.3468e+19
Coulomb matrix 前 5 特征值: [2.87745625e+15 4.14361203e+12 1.33517278e+11 0.00000000e+00
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
