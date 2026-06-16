# 计算高能物理 — 顶夸克质量测量

Project 233 is framed as a cleanroom reproduction task around lattice field theory.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## Workflow outline

- PROJECT 233 — 计算高能物理：顶夸克质量测量统计建模
- 高阶有限差分与稳定性分析（小规模可复现实验）
- 一、项目概述
- 实现了在 LHC (大型强子对撞机) pp 碰撞实验中，通过 $t\bar{t}$ ne变质量谱

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Version output:

```text
synthesis-python-233 1.0
```

## Stable landmarks

```text
PROJECT 233: 计算高能物理 — 顶夸克质量测量
Top Quark Mass Statistical Modeling
High-Order Finite Difference & Stability Analysis
科学目标: 从 tt̄ ne变质量谱提取顶夸克 pole 质量
方法: 轮廓似然比 + NRQCD 阈值截面 + 高阶有限差分
实验条件: LHC Run 2, √s = 13 TeV, L = 139 fb⁻¹
顶夸克质量测量 — 完整分析流水线
Computational High Energy Physics
Top Quark Mass Statistical Analysis
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
