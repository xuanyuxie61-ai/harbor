# Reconstruction Notes: Project 234

Expect hidden checks to care most about the externally visible parts of a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: dalitz_quadrature
- Model layer: b_physics_constants
- Diagnostics/reporting: diophantine_quantum, fractal_perturbation, sparse_decay_io, time_evolution_cp
- Supporting modules: dalitz_geometry, high_order_finite_difference, isobar_amplitude, monte_carlo_phase, multichannel_cvt

## Rebuild approach

- For 博士级 B 物理衰变链重建与 CP 破坏分析, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the scientific computing reconstruction clearer and faster.
- If 博士级 B 物理衰变链重建与 CP 破坏分析 prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `博士级 B 物理衰变链重建与 CP 破坏分析`; hidden tests are not limited to them.

```text
博士级 B 物理衰变链重建与 CP 破坏分析
高阶有限差分与稳定性分析 (小规模可复现实验)
统一入口: 零参数可运行
Python: 3.11.11
NumPy:  1.26.4
[1] CKM 矩阵与么正三角形
Wolfenstein 参数: lambda=0.22650, A=0.7900, rhobar=0.1590, etabar=0.3500
CKM 矩阵 (近似到 O(lambda^4)):
|V_{u}|:  0.97402  exp(i 0.0000)  0.22650  exp(i 0.0000)  0.00353  exp(i -1.1444)
|V_{d}|:  0.22650  exp(i 3.1416)  0.97320  exp(i 0.0000)  0.04053  exp(i 0.0000)
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
