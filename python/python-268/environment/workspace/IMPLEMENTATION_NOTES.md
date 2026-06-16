# Reconstruction Notes: Project 268

The source evidence points to small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Diagnostics/reporting: greens_function, stability_analysis, thermalization_diagnostics
- Supporting modules: determinant_qmc, finite_difference, hubbard_stratonovich, lattice_geometry, observable_estimator, parameter_optimizer

## Suggested coding path

- Make the flag paths cheap for project 268; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by 强关联 Hubbard 模型量子蒙特卡洛模拟系统; random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 268.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `强关联 Hubbard 模型量子蒙特卡洛模拟系统`; hidden tests are not limited to them.

```text
强关联 Hubbard 模型量子蒙特卡洛模拟系统
高阶有限差分与稳定性分析 (小规模可复现实验)
PROJECT_268 - 博士级合成项目
阶段 1: 晶格几何与布里渊区构造
融合种子项目: [02] wedge_grid, [05] triangle_to_fem,
[08] disk_grid, [11] triangulation, [15] metis_graph
>> 1.1 三角晶格构造 (4x4 簇)
簇尺寸: 4 x 4
总格点数: Ns = 16
跳跃连接数: 48
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
