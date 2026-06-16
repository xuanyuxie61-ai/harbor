# Reconstruction Notes: Project 207

The important implementation pressure is sampling, surrogate modelling, and reproducible statistical summaries. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: cube_exactness_quadrature, mesh_vertex_to_element
- Model layer: fem2d_scalar_field
- Diagnostics/reporting: bifurcation_time_delay, kl_expansion, lax_wendroff_advection, prediction_band_builder
- Supporting modules: boundary_word_extractor, cdf_discrete_sampler, confidence_band_calibrator, covariance_cholesky, luhn_checksum_validator, maple_boundary_geometry

## Solver-shaped plan

- Build the executable wrapper last for project 207, after the Python entry point matches probes.
- Keep uncertainty quantification computations deterministic; fixed seeds and fixed iteration counts are preferable.
- Use the anchors below as smoke checks while rebuilding 随机热传导方程的不确定性量化与置信/预测带构建.
- Make the no-argument path finish quickly under verifier time limits.

## Transcript details

Use these public lines as reconstruction checkpoints for `随机热传导方程的不确定性量化与置信/预测带构建`; hidden tests are not limited to them.

```text
随机热传导方程的不确定性量化与置信/预测带构建
Scientific Problem: UQ for Stochastic Parabolic PDEs
Focus: Confidence Intervals & Prediction Intervals
SHE-UQ 问题参数摘要
空间域:        [0, 1.0] m,  nx=51,  dx=0.020000
时间域:        [0, 0.5] s,  nt=101,  dt=0.005000
基准扩散系数:  κ₀ = 1.0000e-02 m²/s
随机扰动强度:  σ_κ = 3.0000e-03 m²/s
相关长度:      ℓ = 0.1000 m
Fourier 数:    Fo = 0.1250
```

Implement the flags explicitly instead of relying on argparse defaults that may format help differently. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
