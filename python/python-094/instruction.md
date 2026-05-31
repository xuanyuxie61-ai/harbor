# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

```markdown
# Nonlinear Acoustic Shock-Wave Simulation – Module Specification

This document describes all Python modules required by the main simulation orchestrator
(`main.py`). The modules must be implemented according to the following descriptions so that
`main.py` runs without error and performs a complete multi‑stage nonlinear acoustics
simulation. The specification focuses on module responsibilities, public APIs, key
algorithms, and inter‑module dependencies. Full implementation details, closed‑form
formulas, and exact numerical constants are intentionally omitted; you need to design the
implementations yourself while preserving the interfaces and overall algorithm logic.

---

## Module 1: `shock_physics.py`

**Purpose:**  
Encapsulates the physical model for nonlinear acoustic shock‑wave propagation.  
It provides medium properties, derived quantities (Mach number, absorption, shock‑formation
distance, Goldberg number), and methods to compute the right‑hand side of the Burgers and
KZK equations, as well as auxiliary physical functions.

**Key elements:**

- **Class `NonlinearAcousticsPhysics`**  
  Constructor parameters: `medium` (`'water'` or `'air'`), `f0` (center frequency),
  `p0` (source pressure), `geometry` (`'planar'`, `'spherical'`, `'cylindrical'`).  
  Must populate instance attributes such as `c0`, `rho0`, `nu`, `beta`, `k0`,
  `wavelength`, `M0`, `classical_absorption`, `shock_formation_distance`,
  `Goldberg_number`, etc.  
  Includes methods:
  - `_classical_absorption()` – simplified Stokes‑Kirchhoff estimate scaled by
    `nu * omega0^2 / c0^2`.
  - `_compute_shock_formation_distance()` – from Fubini solution.
  - `_compute_goldberg_number()` – ratio of absorption length to shock‑formation distance.
  - `taite_equation(p)` – Tait equation of state for water/air.
  - `nonlinear_wave_speed(p)` – pressure‑dependent speed of sound.
  - `burgers_rhs(u, x, nu_eff)` – right‑hand side of the viscous Burgers equation with
    upwind‑biased convection and central‑difference diffusion.
  - `kzk_rhs(p, r_grid, z, diffraction, absorption)` – axisymmetric KZK right‑hand side
    (diffraction and absorption parts).
  - `entropy_production_rate(u, x)` – entropy estimate from velocity gradients.
  - `shock_mach_number(u_post, u_pre)` – Rankine‑Hugoniot‑based Mach number.
  - `spectral_cascade_energy(u_hat, k_vec)` – energy spectrum.
  - `validate_physical_state(u, p)` – sanity checks (finite values, velocity/pressure bounds).

---

## Module 2: `spectral_solver.py`

**Purpose:**  
Spectral differentiation tools based on Vandermonde matrices and polynomial interpolation
nodes (Legendre / Chebyshev families). Used for high‑order spatial discretisation of PDEs.

**Key elements:**

- **Class `VandermondeSolver`**  
  Takes 1‑D node array (no duplicates). Provides:
  - `solve(b)` – solve Vandermonde linear system (e.g. Björck‑Pereyra algorithm, O(n²)).
  - `determinant()` – product of pairwise differences.
  - `to_dense()` – dense matrix.
  - `apply_mv(v)` – matrix‑vector product via Horner’s method.

- **Class `SpectralDifferentiator`**  
  Constructor: number of points `n`, node type (`'legendre_gauss_lobatto'`,
  `'chebyshev_gauss_lobatto'`, `'legendre_gauss'`).  
  Must compute:
  - `nodes` (collocation points in [-1,1]).
  - `differentiation_matrix` (first‑order spectral derivative matrix).
  - `second_derivative_matrix()`.
  - `differentiate(u)` – apply matrix to function values.
  Differentiation matrix construction follows standard formulas (e.g. barycentric weights
  for Legendre‑Gauss‑Lobatto, explicit formula for Chebyshev). For Legendre nodes, you may
  refine internal points by solving for zeros of the derivative of the Legendre polynomial
  with Newton’s method.

- **Standalone functions:**
  - `map_nodes_to_interval(nodes, a, b)` – affine mapping and Jacobian.
  - `solve_burgers_spectral_1d(...)` – solve 1‑D Burgers equation with RK4 time stepping and
    spectral spatial differentiation. Applies homogeneous Dirichlet boundary conditions and
    clipping for stability.

---

## Module 3: `mesh_generator.py`

**Purpose:**  
Generate and optimise 2‑D acoustic domain meshes using hexagonal grids and Centroidal
Voronoi Tessellation (CVT) iterations.

**Key elements:**

- `hex_grid_points(nodes_per_layer, layers, box)` – lay down a staggered hexagonal point
  set inside a rectangular box.
- `hex_grid_approximate_n(...)` – estimate total number of points.
- `find_closest(ndim, ...)` – vectorised nearest‑neighbour assignment for CVT.
- `cvt_iterate_2d(generators, ratio, region_box)` – one Lloyd iteration of CVT:
  random sampling, assignment, centroid update, computes move distance and energy.
- `cvt_optimize_2d(...)` – repeated CVT iteration until convergence.
- `adaptive_density_function(x, y, ...)` – Gaussian bump density used for refinement near
  shock fronts.
- `rejection_sampling_adaptive(...)` – generate points according to adaptive density.

- **Class `AcousticMesh`**  
  Constructor: box, method (`'hex'`, `'cvt'`, `'adaptive_cvt'`), grid parameters and
  CVT iteration count.  
  Generates the final point set in `self.points` (N×2 array).  
  Methods:
  - `compute_element_size()` – approximate average nearest‑neighbour distance.
  - `compute_mesh_quality()` – CVT energy via random sampling.

---

## Module 4: `nonlinear_pde_solver.py`

**Purpose:**  
Time‑advancement solvers for nonlinear acoustic PDEs using Strang splitting (KZK) and
finite‑volume Godunov method (Burgers).

**Key elements:**

- **Class `StrangSplittingSolver`**  
  For the axisymmetric KZK equation. Constructor receives `NonlinearAcousticsPhysics`,
  grid parameters (`dr`, `dtau`, `Nr`, `Ntau`, `r_max`, `tau_max`), and flags for
  diffraction/absorption/nonlinearity. Internally builds spectral differentiation matrices
  for the `tau` direction and estimates CFL‑limited step size.  
  Main method: `propagate(p_initial, z_max, dz)` – advances pressure field in `z` using
  Strang splitting: diffraction half‑step, nonlinear+absorption full step, diffraction
  half‑step. Diffraction step uses explicit Euler or a simple implicit scheme on the
  radial Laplacian. Nonlinear/absorption step is integrated per radial location with a
  Runge‑Kutta method on the spectral ODE system.

- **Class `FiniteVolumeShockCapturing`**  
  First‑order Godunov solver for the viscous Burgers equation. Constructor: `Nx`, domain
  limits, viscosity `nu`.  
  Key method: `solve(u0, t_final, dt)` – time loop calling `step(u, dt)`, which computes
  Godunov numerical flux (explicit formula for Burgers flux) and adds viscous central
  differences. Uses Dirichlet zero boundaries and CFL‑based step adjustment.

---

## Module 5: `geometry_utils.py`

**Purpose:**  
Utility functions for NACA airfoil geometry, triangle metrics, and acoustic boundary
management. Also provides a class for handling NACA boundaries and mesh quality.

**Key elements:**

- NACA functions:
  - `naca4_symmetric(t, c, x)` – thickness
