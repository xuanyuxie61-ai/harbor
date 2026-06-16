# Reverse-Engineer Project 217: 鲁棒优化与不确定约束 —— 博士级科学计算项目 217

## Workspace objective

You are in `/app/workspace` with a reference executable for project `217`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 217 as:

```text
synthesis-python-217 1.0
```

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## Public interface

The default run presents the following workflow title:

```text
鲁棒优化与不确定约束 —— 博士级科学计算项目 217
```

Initial public cues for 鲁棒优化与不确定约束 —— 博士级科学计算项目 217:

```text
步骤 1: 构造不确定性集合
椭球集合维度: 4
样本数量: 20
最坏情况线性目标: 2.4928
鲁棒优化与不确定约束 —— 博士级科学计算项目 217
应用：模拟移动床色谱过程的鲁棒优化
```

Create source files for 鲁棒优化与不确定约束 —— 博士级科学计算项目 217 and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for 鲁棒优化与不确定约束 —— 博士级科学计算项目 217 must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
