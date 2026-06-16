# Reverse-Engineer Project 282: 固态电解质界面反应建模 — 高阶有限差分与稳定性分析

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `282`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 固态电解质界面反应建模 — 高阶有限差分与稳定性分析 is:

```text
synthesis-python-282 1.0
```

The reference executable is the oracle for a reduced fusion and radiation transport experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Observable behavior

Reconstruct the command-line behavior for this fusion and radiation transport target:

```text
固态电解质界面反应建模 — 高阶有限差分与稳定性分析
```

Start by matching these lines:

```text
固态电解质界面反应建模 — 高阶有限差分与稳定性分析
Solid Electrolyte Interphase (SEI) High-Order FD Modeling
阶段 1：参数校验与实验时间戳
参数自洽性校验: 通过
时间戳字符串: 20260608
时间戳哈希（用于 RNG 种子）: 20260608
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 282 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
