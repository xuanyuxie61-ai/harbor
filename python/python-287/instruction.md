# Reverse-Engineer Project 287: MHD 不稳定性数值模拟

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `287`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 287's identity check should return:

```text
synthesis-python-287 1.0
```

The visible binary represents a synthesized uncertainty quantification workflow with a deterministic command-line transcript. Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

## Behavioral target

The public-facing subject of project 287 is:

```text
MHD 不稳定性数值模拟
```

Visible cues for the uncertainty quantification workflow:

```text
#  PROJECT 287: MHD 不稳定性数值模拟
#  高阶有限差分与稳定性分析 (小规模可复现实验)
Python 版本: 3.11.11
NumPy 版本:  1.26.4
PROJECT 287: MHD 不稳定性数值模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
```

The final workspace for project 287 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving MHD 不稳定性数值模拟; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
