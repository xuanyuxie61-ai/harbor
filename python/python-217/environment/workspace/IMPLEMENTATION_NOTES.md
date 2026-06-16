# Reconstruction Notes: Project 217

A practical clone can treat the scientific core as sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: fem3d_mesh
- Model layer: uncertain_field
- Diagnostics/reporting: adjoint_sensitivity, delay_bistability
- Supporting modules: adaptive_esn_controller, chance_constraints, circulant_kkt, dg_pde_constraint, lmc_robust_sampler, robust_optimizer

## Coding plan

- Let `main.py` own the public behavior for 鲁棒优化与不确定约束 —— 博士级科学计算项目 217, while helper modules hold constants and small calculations.
- For 鲁棒优化与不确定约束 —— 博士级科学计算项目 217, small arrays or closed-form summaries are enough if they preserve the reported scale.
- Write output functions for uncertainty quantification summaries instead of scattering print calls everywhere.
- Implement the flags explicitly instead of relying on argparse defaults that may format help differently.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `鲁棒优化与不确定约束 —— 博士级科学计算项目 217`; hidden tests are not limited to them.

```text
步骤 1: 构造不确定性集合
椭球集合维度: 4
样本数量: 20
最坏情况线性目标: 2.4928
鲁棒优化与不确定约束 —— 博士级科学计算项目 217
应用：模拟移动床色谱过程的鲁棒优化
样本均值: [0.45875611 0.44097848 0.46637004 0.44288446]
超球正象限样本: [0.01397656 0.37183484 0.08101746 0.77924355]
非中心 t CDF(2.0, df=10, delta=0.5) = 0.7182, ifault=0
Toeplitz 求解: ||x|| = 2.5166
```

Keep the generated executable at the workspace root and make it executable. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
