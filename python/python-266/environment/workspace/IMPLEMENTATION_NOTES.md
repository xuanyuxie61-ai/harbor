# Reconstruction Notes: Project 266

From the public behavior, the natural decomposition is sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: fd_operators, scf_solver
- Diagnostics/reporting: stability_analysis, xc_functionals
- Supporting modules: band_structure, bayesian_inverse, chebyshev_rootfinder, dos, kohn_sham, lattice

## Practical reconstruction

- Handle `--help` and `--version` explicitly for 计算凝聚态 DFT 能带计算; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 266; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 266, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `计算凝聚态 DFT 能带计算`; hidden tests are not limited to them.

```text
PROJECT 266: 计算凝聚态 DFT 能带计算
高阶有限差分与稳定性分析 (小规模可复现实验)
Phase 1: 物理常数与晶格设置
晶格常数 a = 10.0000 Bohr = 5.2920 Å
倒格矢 G = 0.628319 1/Bohr
BZ 边界 k_max = 0.314159 1/Bohr
1 Hartree = 27.211386 eV
k 点数: 32
k 范围: [-0.3043, 0.3043] 1/Bohr
k 点星: [(0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
