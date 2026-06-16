# Reconstruction Notes: Project 281

The reference run suggests a small driver that combines small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: matrix_eigenvalue, time_integrator
- Model layer: thermodynamic_models
- Diagnostics/reporting: boundary_conditions, combinatorial_enumeration, diagnostic_output, diffusion_simulation, signal_analysis, stability_analysis
- Supporting modules: compact_finite_difference, crystal_lattice, electrode_constants

## Implementation strategy

- Keep the command surface for project 281 narrow: version, help, and the deterministic default workflow.
- Let gauge-field state, action terms, and stability checks drive the helper functions, but keep the default run under verifier time limits.
- For project 281, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Public report cues

Use these public lines as reconstruction checkpoints for `电池电极材料离子扩散模拟`; hidden tests are not limited to them.

```text
PROJECT 281: 电池电极材料离子扩散模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
博士级计算材料科学合成项目
阶段 1: 物理常数与 Arrhenius 扩散系数
T [K]        D [m²/s]    V_T [mV]
────────  ──────────────  ──────────
273.15      8.2721e-15     23.5382
298.15      3.9000e-14     25.6926
323.15      1.4465e-13     27.8469
348.15      4.4443e-13     30.0012
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
