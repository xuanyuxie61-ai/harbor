# PDF 全局拟合与误差传播流水线

This benchmark instance is a black-box reconstruction exercise for a compact lattice field theory program.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## Visible purpose

- main.py — PDF 全局拟合与误差传播的统一入口
- 计算高能物理：PDF 全局拟合与误差传播
- —— 高阶有限差分与稳定性分析（小规模可复现实验）
- 本项目融合 15 个种子项目的核心算法, 实现:

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Reference version:

```text
synthesis-python-231 1.0
```

## Reference cues

```text
PDF 全局拟合与误差传播流水线
计算高能物理：高阶有限差分与稳定性分析
数据点: DIS=77, DY=96, 总计=173
Λ_QCD (LO, nf=5) = 0.087827 GeV
[Stage 1] 初始化参数、网格与合成数据...
x 网格: 24 点, x ∈ [0.001000, 0.990000]
Q² 网格: 12 点, Q² ∈ [2.00, 10000.00] GeV²
[Stage 2] DGLAP 演化 (IMEX 二阶)...
Q² =       4.3381 GeV², 动量 = 0.147051
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
