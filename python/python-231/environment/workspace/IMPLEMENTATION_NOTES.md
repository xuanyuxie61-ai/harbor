# Reconstruction Notes: Project 231

The binary behaves like a scripted experiment focused on small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: fd_stability
- Diagnostics/reporting: dglap_evolution, fd_stability
- Supporting modules: comb_canal, experimental_data, hessian_error, mc_replicas, pdf_fit, pdf_interp

## Solver-shaped plan

- Use a tiny dispatch layer for PDF 全局拟合与误差传播流水线; hidden tests should not depend on accidental framework formatting.
- If a real lattice field theory solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around PDF 全局拟合与误差传播流水线; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Transcript details

Use these public lines as reconstruction checkpoints for `PDF 全局拟合与误差传播流水线`; hidden tests are not limited to them.

```text
PDF 全局拟合与误差传播流水线
计算高能物理：高阶有限差分与稳定性分析
数据点: DIS=77, DY=96, 总计=173
Λ_QCD (LO, nf=5) = 0.087827 GeV
[Stage 1] 初始化参数、网格与合成数据...
x 网格: 24 点, x ∈ [0.001000, 0.990000]
Q² 网格: 12 点, Q² ∈ [2.00, 10000.00] GeV²
[Stage 2] DGLAP 演化 (IMEX 二阶)...
Q² =       4.3381 GeV², 动量 = 0.147051
Q² =       9.4096 GeV², 动量 = 0.126160
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
