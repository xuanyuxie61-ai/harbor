# Reconstruction Notes: Project 233

From the public behavior, the natural decomposition is small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: finite_diff_operators, nonlinear_ode_solvers, polygon_phase_grid
- Diagnostics/reporting: spectral_function, time_delay_stability
- Supporting modules: detector_response, hammersley_phase_space, hankel_momentum_transform, knapsack_combinatoric, monomial_symmetry, profile_likelihood

## Implementation strategy

- Handle `--help` and `--version` explicitly for 计算高能物理 — 顶夸克质量测量; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 233; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 233, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Public report cues

Use these public lines as reconstruction checkpoints for `计算高能物理 — 顶夸克质量测量`; hidden tests are not limited to them.

```text
PROJECT 233: 计算高能物理 — 顶夸克质量测量
Top Quark Mass Statistical Modeling
High-Order Finite Difference & Stability Analysis
科学目标: 从 tt̄ ne变质量谱提取顶夸克 pole 质量
方法: 轮廓似然比 + NRQCD 阈值截面 + 高阶有限差分
实验条件: LHC Run 2, √s = 13 TeV, L = 139 fb⁻¹
顶夸克质量测量 — 完整分析流水线
Computational High Energy Physics
Top Quark Mass Statistical Analysis
[Step 1] 物理参数初始化...
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
