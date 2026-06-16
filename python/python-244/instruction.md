# Reverse-Engineer Project 244: 中子星核物质状态方程约束

## Build goal

You are in `/app/workspace` with a reference executable for project `244`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 244:

```text
synthesis-python-244 1.0
```

Project 244 is framed as a cleanroom reproduction task around uncertainty quantification. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## What must match

Project 244 should be rebuilt around:

```text
中子星核物质状态方程约束
```

Useful observation anchors from the oracle:

```text
中子星核物质状态方程约束
高阶有限差分与稳定性分析 (小规模可复现实验)
Python: 3.11.11
NumPy: 1.26.4
阶段 1: 数值基础验证
机器精度 eps = 2.220446e-16
```

Finish by making a standalone workspace executable for 中子星核物质状态方程约束; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 244, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
