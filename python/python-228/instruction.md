# Reverse-Engineer Project 228: 计算高能物理: 量能器 Shower Profile 快速模拟

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `228`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 计算高能物理: 量能器 Shower Profile 快速模拟 is:

```text
synthesis-python-228 1.0
```

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## Observable behavior

Reconstruct the command-line behavior for this uncertainty quantification target:

```text
计算高能物理: 量能器 Shower Profile 快速模拟
```

Start by matching these lines:

```text
计算高能物理: 量能器 Shower Profile 快速模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
种子项目融合: 15 个科学计算项目 → 统一物理模拟框架
1. 实验参数配置
入射粒子: electron, E0 = 10000.0 MeV = 10.0 GeV
主材料: PbWO4
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 228 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
