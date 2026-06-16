# Stellar evolution with coupled nuclear networks

The reference executable is the oracle for a reduced astrophysics simulation experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## Visible purpose

- PROJECT_249 - 博士级科学计算合成项目
- 计算天体物理：恒星演化与核反应网络——高阶有限差分与稳定性分析
- 科学领域**：计算天体物理，恒星演化，核合成
- 难度**：博士级 / 前沿科学计算

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Reference version:

```text
synthesis-python-249 1.0
```

## Reference cues

```text
PROJECT_249 - Stellar evolution with coupled nuclear networks
High-order finite difference + adaptive mesh + stiff integrators
Stage 1 : Adaptive mass grid construction (1D CVT)
max/min spacing ratio = 2.650
smoothness measure    = 0.635
first 6 mass coords   = ['0.000e+00', '4.208e+32', '1.143e+33', '1.887e+33', '2.660e+33', '3.530e+33']
Stage 2 : Initial stellar model construction
constructed 48 stellar zones
centre  r = 5.191e+09 cm   T = 3.300e+07 K   P = 4.816e+18
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
