# Reconstruction Notes: Project 220

The binary behaves like a scripted experiment focused on a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: fem_operators, linear_solvers, mesh, quadrature
- Supporting modules: adaptive_penalty, admm_core, burgers, config, convergence, domain_decomp

## Suggested coding path

- Use a tiny dispatch layer for 分布式 ADMM 多物理场 PDE 约束优化框架; hidden tests should not depend on accidental framework formatting.
- If a real scientific computing solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around 分布式 ADMM 多物理场 PDE 约束优化框架; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `分布式 ADMM 多物理场 PDE 约束优化框架`; hidden tests are not limited to them.

```text
#  分布式 ADMM 多物理场 PDE 约束优化框架
#  科学领域: 数学优化 — 分布式优化与 ADMM 方法
#  博士级科学计算合成项目 (PROJECT 220)
实验 1: 分布式 Poisson 反问题 (扩散系数辨识)
分布式 ADMM 多物理场 PDE 约束优化框架
科学领域: 数学优化 — 分布式优化与 ADMM 方法
博士级科学计算合成项目 (PROJECT 220)
网格节点数: 1601
单元数: 512
最小单元质量: 0.2500
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
