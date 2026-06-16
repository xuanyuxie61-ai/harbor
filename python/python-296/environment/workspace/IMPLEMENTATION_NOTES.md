# Reconstruction Notes: Project 296

This task can be solved by rebuilding a concise pipeline for magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Likely implementation pieces

- Numerical layer: plasma_mesh, vandermonde_fd
- Diagnostics/reporting: conservation_monitor, energy_deposition_ftcs, error_analysis, phase_space_io, plasma_diagnostics, von_neumann_stability
- Supporting modules: convergence_benchmark, electron_beam_source, electron_trajectory_rk4, plasma_parameters, spectral_tools

## Practical reconstruction

- For this computational plasma physics task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 296's printed diagnostics.
- Keep the final report for project 296 deterministic, compact, and ordered like the reference.
- For Fast Ignition ICF, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Observable checkpoints

Use these public lines as reconstruction checkpoints for `Fast Ignition ICF`; hidden tests are not limited to them.

```text
PROJECT 296: Fast Ignition ICF
电子束能量沉积 高阶有限差分与稳定性分析
小规模可复现实验 (博士级)
Fast Ignition ICF : 电子束能量沉积高阶有限差分与稳定性分析
[阶段 1/14] 物理参数初始化...
临界密度 n_c      = 1.011e+27 /m^3
日冕密度 n_e      = 1.000e+25 /m^3
日冕温度 T_e      = 5.000e+03 eV
稠密芯密度        = 1.000e+31 /m^3
快电子温度        = 1.500e+06 eV  (~ 1.50 MeV)
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
