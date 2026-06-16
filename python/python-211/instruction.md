# Reverse-Engineer Project 211: 多尺度无约束非线性优化

## Workspace objective

You are in `/app/workspace` with a reference executable for project `211`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 211 as:

```text
synthesis-python-211 1.0
```

Project 211 is framed as a cleanroom reproduction task around scientific computing. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## Public interface

The default run presents the following workflow title:

```text
多尺度无约束非线性优化
```

Initial public cues for 多尺度无约束非线性优化:

```text
PROJECT_211: 多尺度无约束非线性优化
量子-经典混合能量景观全局寻优
博士级科学计算项目 (15 种子项目融合)
Python: 3.11.11
NumPy: 1.26.4
模块 1: 特殊函数库验证
```

Create source files for 多尺度无约束非线性优化 and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for 多尺度无约束非线性优化 must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
