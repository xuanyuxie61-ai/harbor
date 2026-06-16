# Edwards-Anderson Spin Glass Simulation

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in uncertainty quantification.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Observed behavior

- 博士级科学合成项目说明
- 项目名称
- 自旋玻璃 Monte Carlo 模拟：高阶有限差分与稳定性分析（小规模可复现实验）
- 英文：Edwards-Anderson Spin Glass: Monte Carlo Simulation with High-Order Finite Difference and Stability Analysis (Small-Scale Reproducible...

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Identity check:

```text
synthesis-python-270 1.0
```

## Transcript anchors

```text
Edwards-Anderson Spin Glass Simulation
High-Order Finite Difference + Stability Analysis
(Small-Scale Reproducible Experiment)
Configuration:
Lattice: 4x4x4 (N=64 spins, z=6)
Boundary: periodic
Coupling: gaussian, J_var=1.0
Disorder realizations: 3
Temperature list: [3.0, 2.0, 1.5, 1.0, 0.7]
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
