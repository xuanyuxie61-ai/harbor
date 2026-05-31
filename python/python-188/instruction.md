# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project python-188: Semantic Embedding Space Analysis and Optimization

This project combines several physics-inspired numerical methods with data-science techniques to analyze, transform, and optimize semantic embedding spaces (e.g., NLP embeddings). The entry point is `main.py`, which orchestrates a pipeline of independent computational modules. The task is to re-implement all modules **except** `main.py` so that the system runs correctly. The descriptions below provide the necessary blueprint without revealing low-level code or exact implementation details.

---

## Main entry: `main.py`
This file is kept unchanged. It imports classes and functions from all other modules, then calls a series of analysis steps (orthogonal basis decomposition, FEM projection, heat diffusion, reaction dynamics, period analysis, CVT quantization, TSP path optimization, subset selection, fractal analysis, structured decomposition, sampling, numerical verification and robustness checks). It prints summary statistics for each step. A successful implementation of the missing modules must make `main.py` execute without errors and produce plausible numeric output.

---

## Files to implement

### 1. `embedding_bases.py`
**Purpose:** Build an orthogonal basis derived from the eigenfunctions of the Helmholtz equation on a disk (Bessel functions). Represent semantic density fields as linear combinations of these bases and perform projection/reconstruction.

**Key components:**
- `SemanticEmbeddingBases` class:
  - Constructor: accepts disk radius, maximum radial mode index `max_mode_m`, and maximum angular mode index `max_mode_n`. It precomputes the zeros of Bessel functions `J_n` and creates a list of (m, n, type) basis indices.
  - `evaluate_basis(r, theta, m, n, angular_type)` – evaluate a single basis function at polar coordinates.
  - `evaluate_all_bases(r, theta)` – return a matrix of all basis function evaluations.
  - `project_semantic_vector(semantic_field, r, theta)` – compute spectral coefficients by L2 projection onto the bases, using a quadrature rule in polar coordinates.
  - `reconstruct_semantic_field(coeffs, r, theta)` – linear combination of bases to recover the field.
  - `basis_orthogonality_check(r, theta)` – compute the normalized Gram matrix to verify orthogonality.

**Implementation notes:** The bases are indexed by radial mode `m` and angular mode `n`. For `n=0` only cosine terms exist, for `n>0` both cosine and sine terms are used. The Helmholtz equation’s radial part yields Bessel functions, and boundary condition `Z=0` on the rim of the disk determines the eigenvalues via zeros of `J_n`. The projection uses a Riemann-sum-like approximation of the L2 inner product that accounts for the polar Jacobian `r`. All numerical Bessel utilities can be taken from `scipy.special`.

### 2. `fem_projection.py`
**Purpose:** 2D finite element method for L2 projection of a semantic density function onto a piecewise-linear triangular mesh. This discretizes a continuous function onto a rectangular grid for further PDE-based processing.

**Key components:**
- `FEM2DSemanticProjection` class:
  - Constructor: defines the rectangular domain (x left/right, y bottom/top) and number of nodes in each direction (`nx`, `ny`). Builds a regular grid of nodes and a triangular element connectivity list (two triangles per rectangle).
  - `project(semantic_func)` – assemble the mass matrix and right-hand side using a quadrature rule (e.g., mid-edge points) over each triangle, enforce Dirichlet boundary conditions (projection matches the function on the boundary), and solve the sparse linear system `A U = b` to obtain nodal values.
  - `compute_l2_error(U, semantic_func)` – estimate the L2 error between the FEM solution and the original function using a higher-order triangular quadrature (e.g., 6-point rule).
  - Various helper methods for element areas and boundary detection.

**Implementation notes:** The matrix assembly uses a standard continuous Galerkin approach with linear basis functions. The sparse system is solved with `scipy.sparse.linalg.spsolve`. The function `semantic_func` is expected to be callable on coordinates `(x, y)`.

### 3. `heat_diffusion.py`
**Purpose:** Solve the 1D steady-state heat equation with spatially varying conductivity and source term, modeling semantic information diffusion along a line segment.

**Key components:**
- `SemanticHeatDiffusion` class:
  - Constructor: number of grid points `n`, interval endpoints `a`, `b`. Computes the spatial step `dx` and the node coordinate array.
  - `solve(ua, ub, conductivity_func, source_func)` – set up a tridiagonal finite-difference system using harmonic averages of conductivity at half-points; enforce Dirichlet boundary conditions `ua`, `ub`; solve the linear system and return the temperature profile `U`.
  - `compute_flux(U, conductivity_func)` – compute the negative gradient times conductivity (heat flux) using central differences in the interior and one-sided differences at boundaries.
  - `solve_nonuniform_conductivity(ua, ub, K_values, F_values)` – variant where conductivity and source are given as arrays of node values (averages used at interfaces).

**Implementation notes:** The ODE is `-d/dx (K dU/dx) = F`. The discrete equations preserve mass conservation (net flux equals integrated source). All linear systems are small and can be solved with direct methods.

### 4. `reaction_dynamics.py`
**Purpose:** Model the bidirectional conversion of semantic concepts as a chemical reaction and its extension to multi-concept reaction networks.

**Key components:**
- `SemanticReactionDynamics` class:
  - Constructor: forward and backward rate constants `k1`, `k2` (both ≥ 0), initial concentrations `w10`, `w20`, time interval.
  - `exact_solution(t)` – returns the analytic solution for `w1(t)`, `w2(t)` (exponential relaxation).
  - `derivative(t, y)` – right-hand side of the ODE `dw1/dt = -k1 w1 + k2 w2`, `dw2/dt = k1 w1 - k2 w2`.
  - `solve_numerical(num_points)` – integrate the ODE using `scipy.integrate.solve_ivp` and return time points and concentrations.
  - `equilibrium()` – return steady states `w1_eq`, `w2_eq`.
  - `relaxation_time()` – characteristic timescale `1/(k1+k2)`.
  - `conserved_quantity(w1, w2)` – compute `w1 + w2` (should remain constant).
- `MultiConceptReactionNetwork` class:
  - Constructor: number of concepts `n`, rate matrix `K` (n×n, each column sums to zero to conserve total mass), initial state vector.
  - `derivative(t, y)` – use matrix-vector product `K @ y`.
  - `solve(t_span, num_points)` – integrate multi-concept ODE.
  - `equilibrium_state()` – solve for steady state under mass conservation.

**Implementation notes:** The rate matrix `K[i,j]` is the rate from concept `j` to `i`. The conservation law is `sum_i y_i = constant`. The solver should use standard ODE integrators.

### 5. `period_analysis.py`
**Purpose:** Analyze nonlinear oscillators relevant to semantic stability: the Van der Pol oscillator and the Lotka‑Volterra predator–prey system. Provide analytic period estimates and numerical measurements.

**Key components:**
- `VanDerPolSemanticOscillator` class:
  - Constructor: damping parameter `mu` (>0).
  - `_derivative(t, y)` – returns `[x', x'']` for `x'' - mu(1-x^2)x' + x = 0`.
  - `period_estimate()` – returns a period estimate using an asymptotic formula (Urabe’s formula) involving `mu` and some empirical constants.
  - `period_cartwright()`, `period_cook()` – simpler period approximations.
  - `solve(t_span, y0, num_points)` – integrate the ODE.
  - `measure_period_numerical(t_span, num_points)` – compute the average period from zero-crossings
