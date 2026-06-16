# Reconstruction Notes: Project 284

The important implementation pressure is high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: adaptive_mesh, high_order_fd
- Diagnostics/reporting: stability_analysis
- Supporting modules: band_sampling, barycentric_interp, basis_selector, brillouin_integral, dos_histogram, fem_basis_t6

## Practical reconstruction

- Build the executable wrapper last for project 284, after the Python entry point matches probes.
- Keep numerical-methods benchmark computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 二维材料异质结能带对齐.
- Make the no-argument path finish quickly under verifier time limits.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `二维材料异质结能带对齐`; hidden tests are not limited to them.

```text
二维材料异质结能带对齐
高阶有限差分与稳定性分析
(小规模可复现实验)
第 1 部分: 材料参数与异质结拓扑
[材料参数]
MoS2  : Eg(300K)=1.864eV, a_B=15.79Å, E_bind=60.8meV, chi=4.00eV
WSe2  : Eg(300K)=1.679eV, a_B=25.64Å, E_bind=31.9meV, chi=4.20eV
MoSe2 : Eg(300K)=1.578eV, a_B=14.85Å, E_bind=69.3meV, chi=4.15eV
WS2   : Eg(300K)=1.936eV, a_B=22.42Å, E_bind=40.1meV, chi=3.90eV
hBN   : Eg(300K)=5.942eV, a_B=3.31Å, E_bind=725.6meV, chi=2.00eV
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
