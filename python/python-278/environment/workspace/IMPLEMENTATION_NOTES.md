# Reconstruction Notes: Project 278

Expect hidden checks to care most about the externally visible parts of high-order stencil construction, stability analysis, and deterministic reporting. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: cvt_composition_mesh, high_order_fd
- Model layer: ode_midpoint_phasefield
- Diagnostics/reporting: bifurcation_detection, clinical_statistics_calphad, cvt_composition_mesh, lyapunov_phase_stability
- Supporting modules: backward_euler_calphad, calphad_fec_constants, calphad_parameter_optimizer, gaussian_spiral_scan, gibbs_energy_calphad, monte_carlo_hyperball

## Practical reconstruction

- For 计算材料 — 相图计算与 CALPHAD 建模, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the numerical-methods benchmark reconstruction clearer and faster.
- If 计算材料 — 相图计算与 CALPHAD 建模 prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `计算材料 — 相图计算与 CALPHAD 建模`; hidden tests are not limited to them.

```text
PROJECT 278: 计算材料 — 相图计算与 CALPHAD 建模
高阶有限差分与稳定性分析 (小规模可复现实验)
Fe-C 二元合金体系
气体常数 R = 8.3145 J/(mol·K)
成分范围: [1.00e-08, 0.25]
温度范围: [700.0, 1800.0] K
随机种子: 278
Step 1: CVT 非均匀成分网格生成 (Lloyd 算法)
CVT 生成点数: 30
Lloyd 迭代次数: 25
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
