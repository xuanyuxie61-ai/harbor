# Reverse-Engineer Project 238: 格点 QCD 有限温相变模拟配置

## Build goal

You are in `/app/workspace` with a reference executable for project `238`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 238:

```text
synthesis-python-238 1.0
```

The reference executable is the oracle for a reduced lattice field theory experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## What must match

Project 238 should be rebuilt around:

```text
格点 QCD 有限温相变模拟配置
```

Useful observation anchors from the oracle:

```text
格点 QCD 有限温相变模拟配置
格点尺寸: 3^3 × 4
作用量类型: tree_level_symanzik
积分器: leapfrog
β 值扫描: [5.0, 5.5, 5.7, 6.0]
轨迹数: 5
```

Finish by making a standalone workspace executable for 格点 QCD 有限温相变模拟配置; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 238, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
