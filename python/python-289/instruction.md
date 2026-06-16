# Reverse-Engineer Project 289: 1D slab gyrokinetic turbulence simulator

## Workspace objective

You are in `/app/workspace` with a reference executable for project `289`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 289 as:

```text
synthesis-python-289 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference spectral physics executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## Public interface

The default run presents the following workflow title:

```text
1D slab gyrokinetic turbulence simulator
```

Initial public cues for 1D slab gyrokinetic turbulence simulator:

```text
PROJECT 289 -- 1D slab gyrokinetic turbulence simulator
Domain:  计算等离子体 / 湍流输运 / gyrokinetic 模拟
高阶有限差分 + 稳定性分析 (小规模可复现实验)
Step 0 -- Equilibrium construction
ion sound speed  c_s      = 2.1885e+05 m/s
ion cyclotron    omega_ci = 1.1974e+08 rad/s
```

Create source files for 1D slab gyrokinetic turbulence simulator and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for 1D slab gyrokinetic turbulence simulator must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
