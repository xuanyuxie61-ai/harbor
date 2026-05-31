# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project Description: Multi-Scale Synaptic Plasticity Simulation

This project is a computational neuroscience framework that simulates synaptic plasticity across multiple spatial and temporal scales. It combines mathematical models of protein diffusion, wave propagation, stochastic weight dynamics, metabolic resource allocation, cortical mesh analysis, homeostatic regulation, spectral analysis, vesicle release, and nonlinear synaptic currents. The entry point is `main.py`, which orchestrates all modules and prints summary statistics.

After this description, all `.py` files **except `main.py`** will be removed. Your task is to re‑implement the missing modules based solely on the information below. The `main.py` file will remain as‑is and will serve as the specification of the module interfaces (imports, function signatures, and expected return types).

---

## Modules

### 1. `numerical_integrator.py`
**Purpose**  
Provides fundamental numerical integration methods for ordinary differential equations (ODEs) used by other modules.

**Key functions**  
- `rk1_integrate(f, tspan, y0, n_steps)` – forward Euler (RK1) integration.  
- `rk4_integrate(f, tspan, y0, n_steps)` – classical Runge‑Kutta 4 integration.  
- `adaptive_rk12(f, tspan, y0, tol, h0, h_min, h_max)` – adaptive step‑size integration using an embedded RK1/RK2 pair.  
- `estimate_stability_jacobian(f, t, y, eps)` – finite‑difference Jacobian computation.  
- `compute_stiffness_ratio(J)` – stiffness ratio from eigenvalues of the Jacobian.

**Dependencies**  
- numpy

**Implementation notes**  
The integrators must handle vector‑valued ODEs, return arrays of time points and solution values, and validate inputs (e.g., positive step count, valid time span). The `rk4_integrate` function is used heavily by `plasticity_wave.py` and `homeostatic_dynamics.py`. The adaptive method adjusts step size based on a local error estimate and tolerance, with minimum and maximum step bounds.

---

### 2. `cable_diffusion.py`
**Purpose**  
Models diffusion of plasticity‑related proteins (PRPs) along a dendritic cable using discrete Laplacian operators and explicit time stepping.

**Key functions**  
- `build_laplacian_1d(n, h, bc)` – returns the `(n,n)` matrix of the 1D discrete Laplacian for boundary conditions `'DD'`, `'DN'`, `'ND'`, `'NN'`, `'PP'`.  
- `apply_laplacian_1d(n, h, u, bc)` – matrix‑free application of the Laplacian to a vector.  
- `cable_diffusion_step(c, D, h, dt, gamma, source, bc)` – single forward Euler step of the cable equation with degradation and optional source term; clips negative concentrations to zero.  
- `laplacian_eigenvalues(n, h, bc)` – computes eigenvalues of the Laplacian matrix for stability analysis.
