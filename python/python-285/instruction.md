# Reverse-Engineer Project 285: 多铁性材料磁电耦合模拟

## Black-box target

You are in `/app/workspace` with a reference executable for project `285`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-285 1.0
```

The benchmark centers on a scientific driver in computational plasma physics, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## Reference behavior

The binary's scientific topic is:

```text
多铁性材料磁电耦合模拟
```

The transcript begins to define the target through:

```text
PROJECT 285: 多铁性材料磁电耦合模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
材料体系: BiFeO₃ (BFO) 钙钛矿多铁性材料
理论框架: Landau-Ginzburg-Devonshire 自由能泛函
数值方法: 6阶有限差分 + 半隐式时间积分 + ADI 分解
1. 数值方法验证
```

Your rebuilt computational plasma physics program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 多铁性材料磁电耦合模拟; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
