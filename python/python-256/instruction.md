# Reverse-Engineer Project 256: 计算天体物理——恒星振动模式与星震学反演

## Build goal

You are in `/app/workspace` with a reference executable for project `256`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 256:

```text
synthesis-python-256 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference astrophysics simulation executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## What must match

Project 256 should be rebuilt around:

```text
计算天体物理——恒星振动模式与星震学反演
```

Useful observation anchors from the oracle:

```text
PROJECT_256: 计算天体物理——恒星振动模式与星震学反演
高阶有限差分与稳定性分析 (小规模可复现实验)
阶段 1: 恒星平衡模型构建
恒星质量: 1.0000 M_sun
恒星半径: 1.0000 R_sun
多方指数: n = 3.0
```

Finish by making a standalone workspace executable for 计算天体物理——恒星振动模式与星震学反演; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 256, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
