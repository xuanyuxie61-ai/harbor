# Reconstruction Notes: Project 218

The important implementation pressure is a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: complementarity_solver, grid_manager, laplacian_operator, nonlocal_operator
- Model layer: bioconvection_dynamics, stochastic_field
- Diagnostics/reporting: bioconvection_dynamics, hermite_functionals, hypercube_projection, variational_inequality
- Supporting modules: active_set_manager, cellular_state_space, hamming_encoder, jacobian_estimator, lagrange_reconstructor

## Practical reconstruction

- Build the executable wrapper last for project 218, after the Python entry point matches probes.
- Keep scientific computing computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 随机变分不等式与非局部互补问题.
- Make the no-argument path finish quickly under verifier time limits.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `随机变分不等式与非局部互补问题`; hidden tests are not limited to them.

```text
PROJECT 218: 随机变分不等式与非局部互补问题
Mathematical Optimization: Variational Inequalities
and Complementarity Problems
Python 版本: 3.11.11
NumPy 版本:  1.26.4
模块 1: 变分不等式问题与互补求解器
问题信息: VariationalInequalityProblem(n=30, min_eig_sym=-2.4162e-01, κ=inf)
维度 n = 30
对称部分最小特征值 = -2.4162e-01
P0-矩阵: False
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
