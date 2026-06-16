# 格点 QCD 有限温相变模拟配置

The reference executable is the oracle for a reduced lattice field theory experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## What to reproduce

- 格点 QCD 有限温相变模拟的统一入口.
- 科学问题:
- 本项目研究纯 SU(3) 规范理论在有限温度下的退禁闭相变.
- 通过 HMC (混合蒙特卡罗) 算法采样规范场构型, 测量 Polyakov loop

## Command surface

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Expected `--version` text:

```text
synthesis-python-238 1.0
```

## Output cues

```text
格点 QCD 有限温相变模拟配置
格点尺寸: 3^3 × 4
作用量类型: tree_level_symanzik
积分器: leapfrog
β 值扫描: [5.0, 5.5, 5.7, 6.0]
轨迹数: 5
MD 步数: 8, 步长: 0.02
夸克质量: 0.1
有限差分精度: O(a^4)
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
