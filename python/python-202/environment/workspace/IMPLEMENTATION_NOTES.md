# Reconstruction Notes: Project 202

The source evidence points to sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: advection_solver, elliptic_solver, sparse_grid, sparse_operators
- Model layer: particle_transport, stochastic_field
- Diagnostics/reporting: advection_solver, basis_conversion, classification_surrogate, moment_computation, sensitivity_analysis
- Supporting modules: adaptive_refinement, calendar_converter, domain_mapper, polynomial_utils, response_surface, stiff_stochastic_ode

## Suggested coding path

- Make the flag paths cheap for project 202; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by 不确定性量化: 随机配置方法 (Stochastic Collocation UQ); random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 202.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `不确定性量化: 随机配置方法 (Stochastic Collocation UQ)`; hidden tests are not limited to them.

```text
不确定性量化: 随机配置方法 (Stochastic Collocation UQ)
PROJECT 202 — 博士级科研代码合成项目
NumPy 版本: 1.26.4
随机种子: 42 (可复现)
第一部分: Smolyak 稀疏网格构造
维度 D=3, Level q=4
多指标数: 19
非零系数数: 19
稀疏网格原始点数: 189
全张量积点数: 343
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
