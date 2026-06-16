# Reverse-Engineer Project 291: 等离子体鞘层高阶有限差分与稳定性分析

## Black-box target

You are in `/app/workspace` with a reference executable for project `291`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-291 1.0
```

The program under observation is a deterministic spectral physics demonstrator packaged as a single executable. Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

## Reference behavior

The binary's scientific topic is:

```text
等离子体鞘层高阶有限差分与稳定性分析
```

The transcript begins to define the target through:

```text
等离子体鞘层高阶有限差分与稳定性分析
融合 15 个种子项目的计算等离子体博士级研究框架
方向: 等离子体鞘层-壁面相互作用
方法: 高阶紧致有限差分 + 谱稳定性分析
阶段 1: 等离子体参数设置与网格生成
等离子体鞘层参数摘要
```

Your rebuilt spectral physics program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 等离子体鞘层高阶有限差分与稳定性分析; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
