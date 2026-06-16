# Reverse-Engineer Project 206: main.py  --  贝叶斯模型校准统一入口

## Task target

You are in `/app/workspace` with a reference executable for project `206`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for main.py  --  贝叶斯模型校准统一入口 is:

```text
synthesis-python-206 1.0
```

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## External contract

Use the following label as the center of the reconstruction:

```text
main.py  --  贝叶斯模型校准统一入口
```

Reference lines worth preserving for project 206:

```text
阶段 1: 数值基底初始化
阶段 2: 合成观测数据生成
真实参数: {'Du': 0.16, 'Dv': 0.08, 'f': 0.04, 'k': 0.06}
sim 观测 (n=8): 0.0282, 0.0036, 0.0276, 0.0187 ...
PROJECT 206: 贝叶斯模型校准
Gray-Scott 反应扩散系统的不确定性量化
```

For project 206, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 206, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
