# Reverse-Engineer Project 223: ╔══════════════════════════════════════════════════════════════════════╗

## Workspace objective

You are in `/app/workspace` with a reference executable for project `223`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 223 as:

```text
synthesis-python-223 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference numerical-methods benchmark executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## Public interface

The default run presents the following workflow title:

```text
╔══════════════════════════════════════════════════════════════════════╗
```

Initial public cues for ╔══════════════════════════════════════════════════════════════════════╗:

```text
╔══════════════════════════════════════════════════════════════════════╗
║  计算高能物理: 喷注聚类与 jet substructure 分析                    ║
║  高阶有限差分与稳定性分析 — 小规模可复现实验                        ║
║  Computational HEP: Jet Clustering & Substructure                  ║
║  High-Order Finite Differences & Stability Analysis                 ║
╚══════════════════════════════════════════════════════════════════════╝
```

Create source files for ╔══════════════════════════════════════════════════════════════════════╗ and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for ╔══════════════════════════════════════════════════════════════════════╗ must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
