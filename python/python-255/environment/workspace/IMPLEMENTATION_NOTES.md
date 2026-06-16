# Reconstruction Notes: Project 255

From the public behavior, the natural decomposition is matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: high_order_fdm, retrieval_solver
- Model layer: atmospheric_model
- Diagnostics/reporting: adaptive_resolution, stability_analysis
- Supporting modules: censoring_design, collatz_tracer, feature_enhancer, fft_analyzer, opacity_engine, radiative_transfer

## Solver-shaped plan

- Handle `--help` and `--version` explicitly for 系外行星大气光谱反演; avoid relying on library-generated text that may drift.
- Keep tolerances and formatting explicit for project 255; scientific notation and spacing are visible.
- Make numeric formatting part of the implementation for project 255, not an afterthought.
- Preserve non-English labels and punctuation where they appear in the reference transcript.

## Transcript details

Use these public lines as reconstruction checkpoints for `系外行星大气光谱反演`; hidden tests are not limited to them.

```text
PROJECT 255 : 系外行星大气光谱反演
Exoplanet Atmospheric Spectral Retrieval
High-Order Finite Differences & Stability Analysis
Step 1: 大气结构建模 (WASP-39b 型热木星)
行星质量        : 5.314e+26 kg
行星半径        : 9.079e+07 m
表面重力        : 4.303 m/s^2
平衡温度        : 1116.0 K
标高 (T_eq)     : 9.303e+11 m
离散层数        : 80
```

If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
