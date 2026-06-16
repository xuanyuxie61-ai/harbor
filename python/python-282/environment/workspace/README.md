# 固态电解质界面反应建模 — 高阶有限差分与稳定性分析

The reference executable is the oracle for a reduced fusion and radiation transport experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## Observed behavior

- 统一入口：固态电解质界面（SEI）高阶有限差分反应-扩散-电迁移耦合建模
- 与稳定性分析（小规模可复现实验）。
- 零参数运行方式：
- 本项目融合 15 个种子项目算法，面向计算材料领域前沿问题：

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Identity check:

```text
synthesis-python-282 1.0
```

## Transcript anchors

```text
固态电解质界面反应建模 — 高阶有限差分与稳定性分析
Solid Electrolyte Interphase (SEI) High-Order FD Modeling
阶段 1：参数校验与实验时间戳
参数自洽性校验: 通过
时间戳字符串: 20260608
时间戳哈希（用于 RNG 种子）: 20260608
阶段 2：SEI 网格构造与纳米孔隙密堆积
节点数 N = 81, 域长 L = 50.0 nm
空间步长 dx = 0.6250 nm
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
