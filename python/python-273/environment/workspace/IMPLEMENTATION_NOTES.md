# Reconstruction Notes: Project 273

The important implementation pressure is small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: dynamical_matrix, eigen_solver, fd_stencil, time_integration
- Model layer: thermal_transport
- Diagnostics/reporting: stability_analysis, time_integration
- Supporting modules: bz_sampling, dos_calc, interatomic_potential, lattice_geometry, monte_carlo, optimize_phonon

## Solver-shaped plan

- Build the executable wrapper last for project 273, after the Python entry point matches probes.
- Keep lattice field theory computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 声子谱与热输运计算: 高阶有限差分与稳定性分析.
- Make the no-argument path finish quickly under verifier time limits.

## Transcript details

Use these public lines as reconstruction checkpoints for `声子谱与热输运计算: 高阶有限差分与稳定性分析`; hidden tests are not limited to them.

```text
声子谱与热输运计算: 高阶有限差分与稳定性分析
计算凝聚态物理 —— 博士级前沿数值实验
目标材料: FCC 铜 (Cu)
晶格常数: 3.615 Angstrom
原子质量: 63.546 amu
第1步: 晶格几何构建
FCC 超胞: 3x3x3 = 108 个原子
超胞边长: 10.845 Angstrom
近邻壳层结构:
壳层 1: r = 2.5562 A, Z = 12
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
