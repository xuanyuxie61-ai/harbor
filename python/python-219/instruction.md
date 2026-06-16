# Reverse-Engineer Project 219: Pontryagin 多阶段随机最优控制框架

## Black-box target

You are in `/app/workspace` with a reference executable for project `219`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-219 1.0
```

The benchmark centers on a scientific driver in scientific computing, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## Reference behavior

The binary's scientific topic is:

```text
Pontryagin 多阶段随机最优控制框架
```

The transcript begins to define the target through:

```text
本框架融合 15 个种子项目的核心算法, 解决能量受限航天器在温度场
PDE 约束下的多目标轨迹优化问题。
科学领域: 数学优化 - 最优控制与 Pontryagin 原理
难度等级: 博士级
Pontryagin 多阶段随机最优控制框架
Stochastic Pontryagin Multi-Stage Optimal Control Framework
```

Your rebuilt scientific computing program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for Pontryagin 多阶段随机最优控制框架; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
