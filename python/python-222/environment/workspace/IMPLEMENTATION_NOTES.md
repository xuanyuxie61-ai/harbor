# Reconstruction Notes: Project 222

From the public behavior, the natural decomposition is high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: fd_operators
- Model layer: rapidity_transport
- Diagnostics/reporting: fragmentation_optimizer, hadronization_string, stability_analysis
- Supporting modules: color_flow_graph, constants, fluid_moments, parton_cascade, phase_space_quality, quantum_constraints

## Rebuild approach

- Handle `--help` and `--version` explicitly for PartonShowerHD v1.0; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 222; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 222, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `PartonShowerHD v1.0`; hidden tests are not limited to them.

```text
PartonShowerHD v1.0
高阶有限差分稳定性分析下的 Parton Shower 与强子化
计算高能物理 · 博士级科学计算合成项目
Stage 1: 物理参数初始化
beta_0(nf=5) = 0.610094
跑动耦合常数 alpha_s(mu):
Stage 2: DGLAP 分裂核多项式投影
Pqq(z) ~ -14.7937 + 536.444*z + -4403.56*z^2 + 13658.7*z^3 + -17661.3*z^4 + 8066.67*z^5
Pqg(z) ~ 0.5 + -1*z + 1*z^2 + -1.55188e-13*z^3 + 1.22402e-13*z^4 + -3.36606e-14*z^5
Pgq(z) ~ 182.095 + -2392*z + 11271.1*z^2 + -23680*z^3 + 22672*z^4 + -8066.67*z^5
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
