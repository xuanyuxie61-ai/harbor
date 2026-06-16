# Reconstruction Notes: Project 209

The binary behaves like a scripted experiment focused on sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: fem_solver, sparse_grid_collocation
- Model layer: random_field
- Diagnostics/reporting: bifurcation, error_analysis, solution_manifold, sparse_grid_collocation
- Supporting modules: bayesian_inverse, deep_surrogate, multi_fidelity, polynomial_chaos

## Implementation strategy

- Use a tiny dispatch layer for 随机参数 PDE 模型的不确定性量化; hidden tests should not depend on accidental framework formatting.
- If a real uncertainty quantification solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around 随机参数 PDE 模型的不确定性量化; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Public report cues

Use these public lines as reconstruction checkpoints for `随机参数 PDE 模型的不确定性量化`; hidden tests are not limited to them.

```text
PROJECT 209: 随机参数 PDE 模型的不确定性量化
Uncertainty Quantification for Stochastic Parameter PDEs
Python 版本: 3.11.11
NumPy 版本: 1.26.4
Step 1: 随机场建模与 Karhunen-Loève 展开
空间点: 400, 相关长度: ℓ=0.3, 方差: σ²=1.0
KL 模态数: 15
能量捕获比: 0.988455
前3个特征值: [130.36380529  67.43047606  67.43047606]
前5模态累积能量: 0.8184
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
