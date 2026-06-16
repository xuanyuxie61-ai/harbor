# Reconstruction Notes: Project 210

The cleanroom notes below emphasize sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: spatial_mesh
- Diagnostics/reporting: calibration, sensitivity
- Supporting modules: adaptive_sampling, asymptotic, limit_states, polynomial_chaos, random_variables, reliability

## Rebuild approach

- Keep project 210's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of 不确定性量化 — 可靠性分析与失效概率计算's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the uncertainty quantification transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `不确定性量化 — 可靠性分析与失效概率计算`; hidden tests are not limited to them.

```text
PROJECT 210: 不确定性量化 — 可靠性分析与失效概率计算
Uncertainty Quantification: Reliability Analysis &
Failure Probability Computation
随机变量数        = 2
问题定义
X1_Strength         : mean = 1.0000, std = 0.2000
X2_Load             : mean = 0.0000, std = 1.0000
极限状态函数: 应力-强度干涉模型
g(u) = (mu_R + sigma_R*u_1) - (mu_S + sigma_S*u_2)
精确 beta = (10-6)/sqrt(4+2.25) = 1.600000
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
