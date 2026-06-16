# Reconstruction Notes: Project 211

From the public behavior, the natural decomposition is a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: linalg_solver, mesh_basis
- Model layer: correlation_field
- Diagnostics/reporting: continuation, correlation_field, special_functions
- Supporting modules: chebyshev_surrogate, global_search, ode_optimizers, quantum_objective, quasi_newton, test_problems

## Coding plan

- Handle `--help` and `--version` explicitly for 多尺度无约束非线性优化; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 211; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 211, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `多尺度无约束非线性优化`; hidden tests are not limited to them.

```text
PROJECT_211: 多尺度无约束非线性优化
量子-经典混合能量景观全局寻优
博士级科学计算项目 (15 种子项目融合)
Python: 3.11.11
NumPy: 1.26.4
模块 1: 特殊函数库验证
T_0(cos(π/4)) = 1.0000000000, cos(0π/4) = 1.0000000000, err = 0.00e+00
T_1(cos(π/4)) = 0.7071067812, cos(1π/4) = 0.7071067812, err = 0.00e+00
T_2(cos(π/4)) = 0.0000000000, cos(2π/4) = 0.0000000000, err = 1.61e-16
T_3(cos(π/4)) = -0.7071067812, cos(3π/4) = -0.7071067812, err = 2.22e-16
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
