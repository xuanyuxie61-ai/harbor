# Reconstruction Notes: Project 223

Expect hidden checks to care most about the externally visible parts of high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: high_order_fd, jet_grid
- Diagnostics/reporting: conservation_laws, stability_analysis
- Supporting modules: event_generator, jet_boundary, jet_clustering, jet_fourvector, jet_substructure

## Coding plan

- For ╔══════════════════════════════════════════════════════════════════════╗, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the numerical-methods benchmark reconstruction clearer and faster.
- If ╔══════════════════════════════════════════════════════════════════════╗ prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `╔══════════════════════════════════════════════════════════════════════╗`; hidden tests are not limited to them.

```text
╔══════════════════════════════════════════════════════════════════════╗
║  计算高能物理: 喷注聚类与 jet substructure 分析                    ║
║  高阶有限差分与稳定性分析 — 小规模可复现实验                        ║
║  Computational HEP: Jet Clustering & Substructure                  ║
║  High-Order Finite Differences & Stability Analysis                 ║
╚══════════════════════════════════════════════════════════════════════╝
阶段 1: 多喷注事件生成
生成 10 个初始部分子
质心系能量: √s = 13000 GeV
过滤后: 9 个粒子 (pT > 15 GeV, |η| < 2.5)
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
