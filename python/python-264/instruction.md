# Reverse-Engineer Project 264: energy_flux / density,

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `264`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for energy_flux / density, is:

```text
synthesis-python-264 1.0
```

This benchmark instance is a black-box reconstruction exercise for a compact computational plasma physics program. Treat the default execution as the primary specification and preserve the order of its visible sections.

## Observable behavior

Reconstruct the command-line behavior for this computational plasma physics target:

```text
energy_flux / density,
```

Start by matching these lines:

```text
energy_flux / density,
磁层粒子输运模拟: 高阶有限差分与稳定性分析
Magnetospheric Particle Transport Simulation
High-Order Finite Difference & Stability Analysis
科学问题: 地球辐射带相对论电子 (L, E) 相空间输运
控制方程: 2D Fokker-Planck (漂移动力学) 方程
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 264 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
