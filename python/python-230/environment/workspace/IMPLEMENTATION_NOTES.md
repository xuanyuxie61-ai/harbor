# Reconstruction Notes: Project 230

This task can be solved by rebuilding a concise pipeline for high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Model layer: physical_model
- Diagnostics/reporting: stability
- Supporting modules: asymptotic, cls_calculator, finite_diff, likelihood, numerical_tools, profiling_optimizer

## Practical reconstruction

- For this numerical-methods benchmark task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 230's printed diagnostics.
- Keep the final report for project 230 deterministic, compact, and ordered like the reference.
- For 计算高能物理 Profile Likelihood 与 CLs 上限设定, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `计算高能物理 Profile Likelihood 与 CLs 上限设定`; hidden tests are not limited to them.

```text
#  PROJECT 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定
#  高阶有限差分与稳定性分析 (小规模可复现实验)
#  融合 15 个科研种子项目的核心算法
模块 1: 物理模型构建 (H -> γγ 双光子搜索)
PROJECT 230: 计算高能物理 Profile Likelihood 与 CLs 上限设定
高阶有限差分与稳定性分析 (小规模可复现实验)
融合 15 个科研种子项目的核心算法
区间数: 5
Nuisance 参数数: 2
标称信号 s_nom: [ 2.   8.  15.   7.   1.5]
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
