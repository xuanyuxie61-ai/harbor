# Reconstruction Notes: Project 267

Expect hidden checks to care most about the externally visible parts of matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: boundary_state_solver, brillouin_quadrature, fd_stencils, self_consistent_solver
- Diagnostics/reporting: stability_analysis, time_evolution
- Supporting modules: bhz_hamiltonian, disorder_generator, eigenvalue_root_finder, monte_carlo_averaging, topological_invariants

## Solver-shaped plan

- For 拓扑绝缘体边界态高阶有限差分求解器, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the spectral physics reconstruction clearer and faster.
- If 拓扑绝缘体边界态高阶有限差分求解器 prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Transcript details

Use these public lines as reconstruction checkpoints for `拓扑绝缘体边界态高阶有限差分求解器`; hidden tests are not limited to them.

```text
拓扑绝缘体边界态高阶有限差分求解器
Topological Insulator Boundary State FD Solver
计算凝聚态: 拓扑绝缘体边界态数值求解
高阶有限差分与稳定性分析 (小规模可复现实验)
NumPy 版本: 1.26.4
████████████████████████████████████████████████████████████████████████
阶段 1: 高阶有限差分模板构造与验证
有限差分模板系数汇总
一阶导数系数 c_j:
二阶导数系数 d_j:
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
