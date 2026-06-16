# Reverse-Engineer Project 233: 计算高能物理 — 顶夸克质量测量

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `233`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 233's identity check should return:

```text
synthesis-python-233 1.0
```

Project 233 is framed as a cleanroom reproduction task around lattice field theory. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## Behavioral target

The public-facing subject of project 233 is:

```text
计算高能物理 — 顶夸克质量测量
```

Visible cues for the lattice field theory workflow:

```text
PROJECT 233: 计算高能物理 — 顶夸克质量测量
Top Quark Mass Statistical Modeling
High-Order Finite Difference & Stability Analysis
科学目标: 从 tt̄ ne变质量谱提取顶夸克 pole 质量
方法: 轮廓似然比 + NRQCD 阈值截面 + 高阶有限差分
实验条件: LHC Run 2, √s = 13 TeV, L = 139 fb⁻¹
```

The final workspace for project 233 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving 计算高能物理 — 顶夸克质量测量; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
