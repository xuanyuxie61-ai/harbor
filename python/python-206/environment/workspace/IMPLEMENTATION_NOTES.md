# Reconstruction Notes: Project 206

A practical clone can treat the scientific core as sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: evidence_quadrature, map_solver
- Model layer: forward_model, surrogate_model
- Diagnostics/reporting: bayesian_calibration, constrained_calibration, prior_geometry
- Supporting modules: domain_adapter, experimental_design, numerical_base, proposal_engine, synthetic_data

## Practical reconstruction

- Let `main.py` own the public behavior for main.py  --  贝叶斯模型校准统一入口, while helper modules hold constants and small calculations.
- For main.py  --  贝叶斯模型校准统一入口, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for uncertainty quantification summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `main.py  --  贝叶斯模型校准统一入口`; hidden tests are not limited to them.

```text
阶段 1: 数值基底初始化
阶段 2: 合成观测数据生成
真实参数: {'Du': 0.16, 'Dv': 0.08, 'f': 0.04, 'k': 0.06}
sim 观测 (n=8): 0.0282, 0.0036, 0.0276, 0.0187 ...
PROJECT 206: 贝叶斯模型校准
Gray-Scott 反应扩散系统的不确定性量化
[NumericalBase] ibeta=2, it=53, irnd=1
real 观测 (n=8): 0.0392, 0.0089, 0.0076, 0.0095 ...
阶段 3: 先验几何构建
参数维度: 4
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
