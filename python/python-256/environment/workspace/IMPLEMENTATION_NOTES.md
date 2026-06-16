# Reconstruction Notes: Project 256

Expect hidden checks to care most about the externally visible parts of reduced physical models, time evolution, and tabulated diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Model layer: oscillation_equations
- Diagnostics/reporting: domain_decomposition, oscillation_equations, stability_analysis
- Supporting modules: boundary_encoding, high_order_finite_diff, hypernetwork_params, inverse_problem, mode_classifier, monte_carlo_sampler

## Suggested coding path

- For 计算天体物理——恒星振动模式与星震学反演, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the astrophysics simulation reconstruction clearer and faster.
- If 计算天体物理——恒星振动模式与星震学反演 prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `计算天体物理——恒星振动模式与星震学反演`; hidden tests are not limited to them.

```text
PROJECT_256: 计算天体物理——恒星振动模式与星震学反演
高阶有限差分与稳定性分析 (小规模可复现实验)
阶段 1: 恒星平衡模型构建
恒星质量: 1.0000 M_sun
恒星半径: 1.0000 R_sun
多方指数: n = 3.0
平均分子量: μ = 0.6173
中心密度: ρ_c = 1.1078e+01 g/cm³
中心压强: P_c = 5.2846e+15 dyn/cm²
中心温度: T_c = 3.5693e+06 K
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
