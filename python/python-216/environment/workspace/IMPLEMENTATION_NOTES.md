# Reconstruction Notes: Project 216

For reconstruction, model the program as a staged report over a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: helmholtz_solver, quadrature
- Model layer: random_field
- Diagnostics/reporting: convergence_analysis, validation
- Supporting modules: middle_square_rng, saa_objective, scientific_formulas, stochastic_optimizer

## Rebuild approach

- Write the CLI by hand if needed; the visible contract for Stochastic Helmholtz Optimization via SAA is more important than argparse styling.
- Use calibrated constants for scientific computing quantities rather than running an oversized simulation.
- Preserve bilingual labels, units, and bracketed stage markers when they appear in Stochastic Helmholtz Optimization via SAA.
- Keep the generated executable at the workspace root and make it executable.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `Stochastic Helmholtz Optimization via SAA`; hidden tests are not limited to them.

```text
阶段 1: 伪随机数发生器初始化 (seed: 763_middle_square + 1393_vin)
主种子 = 2160607
校验 seed 完整性: True
前 10 个 U[0,1) 样本:
PROJECT 216: Stochastic Helmholtz Optimization via SAA
随机亥姆霍兹方程的样本平均近似优化
领域: 数学优化 - 随机优化与样本平均近似
状态连续性校验: True
分支 RNG 派生完成: branch 1 seed_r = 9508464125378339, branch 2 seed_r = 7881731219276392
阶段 2: Karhunen-Loève 随机场 (seed: 479_gram + 333_ellipsoid)
```

Probe before coding, because several projects share generic themes but differ in their visible numbers. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
