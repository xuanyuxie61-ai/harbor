# Reverse-Engineer Project 279: 计算材料基因组高通量筛选框架

## Black-box target

You are in `/app/workspace` with a reference executable for project `279`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-279 1.0
```

This case exposes the observable surface of a research-style fusion and radiation transport driver rather than its source. Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

## Reference behavior

The binary's scientific topic is:

```text
计算材料基因组高通量筛选框架
```

The transcript begins to define the target through:

```text
PROJECT 279: 计算材料基因组高通量筛选框架
高阶有限差分与稳定性分析 (LLZO 固态电解质)
Phase 1: 晶体结构编码与描述符生成
LLZO 立方晶格常数: a = 12.970 Å
代表性原子数: 13
晶胞体积: V = 2181.83 Å³
```

Your rebuilt fusion and radiation transport program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 计算材料基因组高通量筛选框架; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
