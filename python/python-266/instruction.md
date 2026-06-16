# Reverse-Engineer Project 266: 计算凝聚态 DFT 能带计算

## Task target

You are in `/app/workspace` with a reference executable for project `266`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for 计算凝聚态 DFT 能带计算 is:

```text
synthesis-python-266 1.0
```

Project 266 is framed as a cleanroom reproduction task around uncertainty quantification. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## External contract

Use the following label as the center of the reconstruction:

```text
计算凝聚态 DFT 能带计算
```

Reference lines worth preserving for project 266:

```text
PROJECT 266: 计算凝聚态 DFT 能带计算
高阶有限差分与稳定性分析 (小规模可复现实验)
Phase 1: 物理常数与晶格设置
晶格常数 a = 10.0000 Bohr = 5.2920 Å
倒格矢 G = 0.628319 1/Bohr
BZ 边界 k_max = 0.314159 1/Bohr
```

For project 266, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 266, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
