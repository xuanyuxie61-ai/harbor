# Binary Neutron Star Merger & Kilonova

The visible binary represents a synthesized fusion and radiation transport workflow with a deterministic command-line transcript.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. The hidden tests may look beyond the examples, so preserve the general report shape rather than only one line.

## Scientific role

- PROJECT_254 — 计算天体物理:双中子星并合与 kilonova 辐射转移
- 高阶有限差分与稳定性分析(小规模可复现实验)
- 博士级合成代码 · 中文文档 · Python 3.10+
- 1. 科学问题定位

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

Stable version text:

```text
synthesis-python-254 1.0
```

## Report signatures

```text
#                                                                      #
#   PROJECT_254 : Binary Neutron Star Merger & Kilonova         #
#   High-Order Finite Differences & Stability Analysis          #
Python executable : /usr/local/bin/python3
PROJECT_254 : Binary Neutron Star Merger & Kilonova
High-Order Finite Differences & Stability Analysis
Python version    : 3.11.11
Stage 1 / Physical Constants & Derived Quantities
[self-check] passed = True
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.
