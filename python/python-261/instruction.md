# Reverse-Engineer Project 261: 计算宇宙学: 再电离历史数值模拟

## Black-box target

You are in `/app/workspace` with a reference executable for project `261`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-261 1.0
```

This is a ProgramBench-style task: infer and reproduce a astrophysics simulation CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## Reference behavior

The binary's scientific topic is:

```text
计算宇宙学: 再电离历史数值模拟
```

The transcript begins to define the target through:

```text
#  计算宇宙学: 再电离历史数值模拟
#  高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
reion_cosmology.py       : 物理与宇宙学常数
reion_grid.py            : 计算网格生成
计算宇宙学: 再电离历史数值模拟
高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
```

Your rebuilt astrophysics simulation program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 计算宇宙学: 再电离历史数值模拟; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
