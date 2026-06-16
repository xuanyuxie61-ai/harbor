# Reverse-Engineer Project 232: 计算高能物理散射振幅数值计算与截面积分

## Build goal

You are in `/app/workspace` with a reference executable for project `232`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 232:

```text
synthesis-python-232 1.0
```

The visible binary represents a synthesized lattice field theory workflow with a deterministic command-line transcript. Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

## What must match

Project 232 should be rebuilt around:

```text
计算高能物理散射振幅数值计算与截面积分
```

Useful observation anchors from the oracle:

```text
PROJECT_232: 计算高能物理散射振幅数值计算与截面积分
高阶有限差分与稳定性分析 (小规模可复现实验)
物理过程: ππ → ππ 弹性散射 (I=2 通道)
粒子质量: m_π± = 0.139570 GeV, m_π⁰ = 0.134977 GeV
阈值: √s_thr = 2m_π = 0.279141 GeV
1. 运动学网格构造
```

Finish by making a standalone workspace executable for 计算高能物理散射振幅数值计算与截面积分; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 232, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
