# Reverse-Engineer Project 231: PDF 全局拟合与误差传播流水线

## Black-box target

You are in `/app/workspace` with a reference executable for project `231`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-231 1.0
```

This benchmark instance is a black-box reconstruction exercise for a compact lattice field theory program. Treat the default execution as the primary specification and preserve the order of its visible sections.

## Reference behavior

The binary's scientific topic is:

```text
PDF 全局拟合与误差传播流水线
```

The transcript begins to define the target through:

```text
PDF 全局拟合与误差传播流水线
计算高能物理：高阶有限差分与稳定性分析
数据点: DIS=77, DY=96, 总计=173
Λ_QCD (LO, nf=5) = 0.087827 GeV
[Stage 1] 初始化参数、网格与合成数据...
x 网格: 24 点, x ∈ [0.001000, 0.990000]
```

Your rebuilt lattice field theory program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for PDF 全局拟合与误差传播流水线; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
