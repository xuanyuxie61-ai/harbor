# Reconstruction Notes: Project 215

The reference run suggests a small driver that combines a deterministic research-style workflow with staged numerical diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: adaptive_rk_integrator, quadrature_engine
- Diagnostics/reporting: objective_functions
- Supporting modules: coupled_pde_system, cvt_design_sampler, nsga2_optimizer, pareto_core, pram_parallel, stochastic_robustness

## Implementation strategy

- Keep the command surface for project 215 narrow: version, help, and the deterministic default workflow.
- Let configuration values, numerical kernels, and formatted report sections drive the helper functions, but keep the default run under verifier time limits.
- For project 215, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Public report cues

Use these public lines as reconstruction checkpoints for `设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,`; hidden tests are not limited to them.

```text
f1: 结构柔度    (最小化 → 最大化刚度)
f2: 热应力方差  (最小化 → 均匀温度分布)
f3: 负转化率    (最小化 → 最大化生化转化效率)
约束: 体积分数 ≤ 0.5
科学问题:
设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,
同时优化:
方法: NSGA-II + 顺序耦合 PDE + 随机鲁棒性
[1.1] 数值积分精度验证:
[1.2] Gauss-Legendre 节点 (5阶):
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
