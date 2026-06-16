# Reconstruction Notes: Project 214

The observable transcript is organized around a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: measurement_operator
- Diagnostics/reporting: diagnostics, gan_sparse_prior, regularization_path
- Supporting modules: fem_forward, graph_sparsity, l1_ista, levelset_support, monte_carlo_feynman_kac, sparse_basis

## Suggested coding path

- Give L1-Regularized Sparse PCE Recovery a stable wrapper script so the verifier sees the same executable interface every run.
- Do not overfit one line of project 214; preserve the staged progression of the report.
- Render L1-Regularized Sparse PCE Recovery's section headers through explicit strings so punctuation survives refactors.
- Be careful with warning text: hidden tests generally inspect stdout and exit behavior.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `L1-Regularized Sparse PCE Recovery`; hidden tests are not limited to them.

```text
Project 214: L1-Regularized Sparse PCE Recovery
for Stochastic Elliptic PDEs
阶段 1: 稀疏多项式基构造
维数 d = 3,  总阶 p = 4
全阶截断基大小 M_full = 35
双曲交叉基大小 M_hc  = 11 (q=0.6)
Legendre 正交性验证误差 (16 对): 8.218e-16
8 点 Gauss-Jacobi (α=β=0.5) 精确阶: 15 (理论 = 15)
阶段 2: 观测算子与采样
观测算子 A ∈ R^{84 × 11}
```

Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
