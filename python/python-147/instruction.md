# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# Project python-147: PINN Solver for the Kuramoto–Sivashinsky Equation

## 1. Project Overview

This project builds a Physics‑Informed Neural Network (PINN) to solve the one‑dimensional Kuramoto–Sivashinsky (KS) partial differential equation on a periodic domain.  The main script (`main.py`) orchestrates a full scientific computing pipeline that includes:

- Generating a reference solution using a spectral Exponential Time Differencing (ETDRK4) method.
- Training a PINN with a composite physics‑informed loss.
- Comparing the PINN against manufactured solutions and classical RBF interpolation.
- Analysing chaotic initial conditions, quadrature rules, mesh utilities, and adaptive collocation point sampling.

The remaining Python files implement the building blocks.  Your task is to recreate all missing source files (everything except `main.py`) to restore the full functionality.

## 2. File‑by‑File Description

### 2.1 `ks_pde_solver.py` – Reference Spectral Solver

Provides a numerically accurate reference for the KS equation  
\(u_t + u u_x + u_{xx} + u_{xxxx} = 0\) on the periodic domain \(x \in [0, 32\pi]\).

**Core functions:**

- `solve_ks_etdrk4(nx, tmax, dt, n_snapshots)`  
  Runs an ETDRK4 time‑stepping scheme in Fourier space.  Returns spatial grid `x`, time snapshots `t`, solution matrix `u` (spatial × temporal), wavenumbers `k`, and the linear Fourier operator `L_op`.  The initial condition is a fixed trigonometric function.  ETDRK4 coefficients are pre‑computed via a roots‑of‑unity contour integral (Kassam–Trefethen approach).

- `ks_reference_residual(u, x, t, k)`  
  Computes the PDE residual \(u_t + u u_x + u_{xx} + u_{xxxx}\) for a given field using spectral differentiation (FFT).

### 2.2 `pinn_network.py` – Neural Network Architecture

Defines a custom fully‑connected feed‑forward network `PINNNetwork` that maps \((t, x)\) to \(u(t,x)\).

**Key class and method responsibilities:**

- `PINNNetwork(input_dim, hidden_dims, output_dim, activation, rbf_scale, seed)`  
  Initialises weight matrices and biases using Xavier/Glorot scaling.  Supported activations: `'gaussian_rbf'`, `'squircle'`, and `'tanh'`; each has a corresponding derivative implementation.

- `forward(X, store_cache)` / `predict(X)`  
  Compute the network output; forward can optionally cache intermediate values.

- `finite_difference_derivatives(X, var_idx)` – first‑order partial derivative w.r.t. a chosen input variable.

- `second_derivative(X, var_idx)` / `fourth_derivative(X, var_idx)` – higher‑order derivatives via centred finite‑difference stencils.

- `get_params_flat()` / `set_params_flat(params)` – flatten/unflatten all weights and biases for gradient‑based optimisation.

- `parameter_count()` – returns total number of trainable parameters.

### 2.3 `physics_loss.py` – Physics‑Informed Loss Functions

Computes the loss terms that enforce the KS dynamics, initial conditions, and periodic boundary conditions.

**Core functions:**

- `compute_pde_residual(network, X_f)`  
  Evaluates \(u_t + u u_x + u_{xx} + u_{xxxx}\) at a set of collocation points using the network’s finite‑difference derivatives.

- `compute_ic_loss(network, X_ic, u_ic_target)`  
  Mean squared error against a target initial condition.

- `compute_bc_loss(network, X_bc_0, X_bc_L)`  
  Enforces \(u(t,0) = u(t,L)\) by comparing outputs at paired boundary points.

- `compute_total_loss(network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L, lambda_pde, lambda_ic, lambda_bc)`  
  Weighted sum of the three loss terms.  Returns the total loss and a dictionary with individual components.

- `compute_loss_gradient(network, ...)`  
  Numerical gradient of the total loss via central differences (full batch).  An additional minibatch variant is available.

### 2.4 `stochastic_optimizer.py` – Training Optimisers

Implements optimisation routines tailored for the PINN loss landscape.

**Key classes:**

- `SGDWithMomentum(params_dim, lr, momentum, lr_decay, min_lr)`  
  Nesterov‑accelerated gradient descent with step‑wise learning‑rate decay.

- `StochasticCoordinateDescent(params_dim, block_size, lr, lr_decay)`  
  Randomly selects a block of coordinates and performs a gradient‑based update, inspired by Gauss–Seidel ideas.

- `CosineAnnealingScheduler(eta_max, eta_min, T_max)`  
  Produces a learning rate that follows a cosine decay schedule.

- `CombinedOptimizer(...)`  
  Switches from SGD with momentum to coordinate descent after a fixed number of iterations.

### 2.5 `domain_mesh.py` – Collocation Point and Mesh Utilities

Generates and manages the discrete points where the physics loss is evaluated.

**Core functions:**

- `generate_collocation_grid(tmax, L_domain, nt, nx)` – structured space‑time grid.
- `generate_boundary_points(tmax, L_domain, n_bc)` – paired points for periodic BCs.
- `generate_initial_condition_points(L_domain, nx_ic)` – points at \(t=0\).
- `triangulation_boundary_edges(triangle_nodes)` – extracts boundary edges from a triangulation.
- `boundary_edge_to_path(boundary_edges)` – orders boundary edges into a closed polygon.
- `find_nearest_neighbors(points_ref, points_query)` – brute‑force nearest neighbour search.
- `cluster_points_by_distance(points, threshold)` – distance‑based clustering.

### 2.6 `rbf_kernel.py` – Radial Basis Functions and RBF Layers

Provides classical RBF interpolation as a baseline, plus an RBF layer that can be integrated into a network.

**Core functions and class:**

- `compute_pairwise_distance(X1, X2)` – Euclidean distance matrix.
- `rbf_phi1` to `rbf_phi4` – multiquadric, inverse multiquadric, thin‑plate spline, and Gaussian kernels.
- `rbf_interpolation_weights(X_data, f_data, r0, phi_type)` – solves the linear system for RBF weights and returns weights and a condition number.
- `rbf_interpolate(X_data, w, r0, X_query, phi_type)` – evaluates the interpolant.
- `RBFKernelLayer(n_centers, input_dim, r0, phi_type, learnable_centers, seed)` – a standalone layer with centres, output weights, and bias.

### 2.7 `chaos_utils.py` – Chaotic Dynamics and Initial Conditions

Generates non‑trivial initial conditions that capture spatiotemporal complexity of the KS equation.

**Core functions:**

- `squircle_trajectory(s, t0, y0, tstop, n_points)` – solves the squircle ODE (a generalisation of harmonic motion) with RK4.
- `squircle_activation_basis(x, s, n_modes)` – creates a periodic basis from time‑shifted squircle solutions.
- `cross_chaos_ifs(n_points, seed)` – iterates an IFS to produce points on a fractal cross.
- `cellular_automaton_rule30(cell_num, step_num, seed_center)` – evolves Rule‑30 CA.
- `generate_chaotic_initial_condition(L_domain, nx, chaos_type, amplitude)` – selects one of the above sources to build a spatial initial profile \(u(0,x)\).

### 2.8 `quadrature_rules.py` – Numerical Integration

Supplies high‑order quadrature rules, potentially
