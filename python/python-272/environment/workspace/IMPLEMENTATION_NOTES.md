# Reconstruction Notes: Project 272

A practical clone can treat the scientific core as matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: brillouin_mesh, ode_integrator, spectral_solver
- Diagnostics/reporting: stability_analysis, statistics_analyzer
- Supporting modules: berry_curvature, chern_number, finite_difference, polynomial_basis, weyl_hamiltonian

## Practical reconstruction

- Let `main.py` own the public behavior for Weyl 半金属 Berry Curvature 计算系统, while helper modules hold constants and small calculations.
- For Weyl 半金属 Berry Curvature 计算系统, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for spectral physics summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `Weyl 半金属 Berry Curvature 计算系统`; hidden tests are not limited to them.

```text
Weyl 半金属 Berry Curvature 计算系统
高阶有限差分与稳定性分析 (小规模可复现实验)
第 1 部分：Weyl Hamiltonian 构建与能带结构
本征值: E_± = -0.2450, 0.3550
[Hamiltonian] Γ 点 Hamiltonian 矩阵:
H(Γ) =
[[+0.0550 -0.3000 + +0.0000i],
[-0.3000 +0.0550 + +0.0000i]]
[Hamiltonian] Γ 点能隙: Δ = 0.6000
[Hamiltonian] 高对称路径: Gamma → X → M → Gamma → R → X → R → M
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
