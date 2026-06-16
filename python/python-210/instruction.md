# Reverse-Engineer Project 210: 不确定性量化 — 可靠性分析与失效概率计算

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `210`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 不确定性量化 — 可靠性分析与失效概率计算 is:

```text
synthesis-python-210 1.0
```

The visible binary represents a synthesized uncertainty quantification workflow with a deterministic command-line transcript. Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

## Observable behavior

Reconstruct the command-line behavior for this uncertainty quantification target:

```text
不确定性量化 — 可靠性分析与失效概率计算
```

Start by matching these lines:

```text
PROJECT 210: 不确定性量化 — 可靠性分析与失效概率计算
Uncertainty Quantification: Reliability Analysis &
Failure Probability Computation
随机变量数        = 2
问题定义
X1_Strength         : mean = 1.0000, std = 0.2000
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 210 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
