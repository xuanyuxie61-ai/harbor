# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Project Description: Swarm Robotics Emergent Behaviour Simulation

This project implements a multi‑scale simulation of self‑organizing robot collectives. It combines spatial meshing, environmental scalar fields, coverage optimisation (Centroidal Voronoi Tessellation), sensor noise models, sparse graph Laplacians, stochastic collision‑avoidance potentials, hybrid robot dynamics with internal chaotic states, and macroscopic density field evolution. The final benchmark script (`main.py`) orchestrates all modules into a single, zero‑parameter workflow that computes several emergence metrics.

When reconstructing the missing code, only `main.py` is provided; all other files listed below must be re‑implemented from this description.

---

## File inventory and responsibilities

### `spatial_mesh.py`
- Defines the class **`TetMesh`** that represents a tetrahedral mesh in 3D.
  - Construction from node coordinates and element connectivity (0‑based, 4‑node tetrahedra).
  - Method **`refine()`** performs one level of 8‑to‑1 subdivision: each tetrahedron is split into eight children by inserting mid‑edge nodes.
  - Method **`point_in_tet()`** tests point containment using barycentric coordinates.
  - Method **`locate_point()`** brute‑force locates which tetrahedron contains a query point.
  - Method **`interpolate_nodal_field()`** evaluates a scalar field defined on nodes at an arbitrary point by barycentric interpolation inside the containing tetrahedron.
- Factory function **`generate_simple_tet_mesh(scale)`** returns a `TetMesh` covering the cube `[-scale, scale]^3` with a Kuhn triangulation (6 tetrahedra).
- Function **`read_mesh_medit(filename)`** reads a minimal MEDIT `*.mesh` file and returns a `TetMesh` or `None`.

### `environment_field.py`
- Defines the class **`EnvironmentField`** that holds a scalar field on a `TetMesh`.
  - Constructor takes a `TetMesh` and a nodal‑values array.
  - Method **`evaluate(p)`** returns the interpolated field value at a 3D point; returns `NaN` if outside the mesh.
  - Method **`gradient(p, eps)`** approximates the gradient by central finite differences.
- Functions:
  - **`generate_gradient_field(mesh, direction, magnitude)`** creates a linear field `φ(x) = magnitude · ⟨direction, x⟩`.
  - **`generate_gaussian_bump_field(mesh, center, sigma, amplitude)`** creates a Gaussian bump.
  - **`sample_field_at_positions(field, positions)`** evaluates the field at multiple robot positions.

### `sensor_noise.py`
- Implements noise injection models for scalar measurements (values in `[0,1]`):
  - **`salt_and_pepper_noise(measurements, level)`** randomly sets a fraction `level/2` to 0.0 and another `level/2` to 1.0.
  - **`uniform_noise(measurements, level)`** replaces a fraction `level` of entries with uniform random values.
  - **`gaussian_sensor_noise(measurements, sigma)`** adds zero‑mean Gaussian noise with standard deviation `sigma` and clips to `[0,1]`.
  - **`apply_sensor_noise(measurements, config)`** applies a combination of the above noises according to a configuration dictionary (keys `salt_pepper_level`, `uniform_level`, `gaussian_sigma`).

### `coverage_optimization.py`
- Implements Centroidal Voronoi Tessellation (CVT) for area coverage.
  - **`cvt_lloyd_2d(generators, density_func, bounds, n_samples, n_iterations)`** runs Lloyd’s algorithm in a rectangular 2D domain. It uses Monte‑Carlo sampling, computes weighted centroids of Voronoi cells weighted by the density function, handles empty cells by random re‑initialisation, and returns the optimised generators and an energy history list.
  - **`cvt_circle_nonuniform_density(n, radius, n_iterations, n_samples)`** initialises generators inside a circle by rejection sampling, defines a non‑uniform density `1 + 5·exp(−5·r²/R²)`, calls `cvt_lloyd_2d`, projects points back onto the disk, and returns the generators and energy history.
  - **`coverage_metric(positions, density_func, bounds, n_samples)`** computes the normalised CVT energy for a given configuration of points.

### `interaction_matrix.py`
- Builds the sparse graph Laplacian of a geometric proximity graph.
  - **`build_sparse_laplacian(positions, sensing_radius, weight_func, block_size)`** assembles a weighted adjacency matrix `W` for pairs within the sensing radius, using a distance‑dependent weight function (default: squared tent). It works in blocks to limit memory and returns both the Laplacian `L = D − W` and the adjacency `W` as `scipy.sparse.csr_matrix` objects.
  - **`fiedler_value(L)`** computes the algebraic connectivity (second smallest eigenvalue) of the Laplacian using sparse eigensolver, returning 0 if disconnected.
  - **`consensus_dynamics_step(x, L, dt)`** performs one explicit Euler step of the linear consensus protocol `x − dt*L·x`.

### `stochastic_control.py`
- Provides Feynman‑Kac collision‑avoidance potentials.
  - **`potential(a, x)`** evaluates a quadratic potential used in the 1‑D reference problem.
  - **`feynman_kac_1d_solve(a, h, n_paths, n_grid)`** solves a 1‑D Poisson problem via Monte‑Carlo Feynman‑Kac, as a validation example.
  - **`feynman_kac_collision_potential(positions, obstacles, obstacle_radius, domain_radius)`** computes a 2‑D heuristic collision‑avoidance potential at robot positions. It combines a boundary term (exponential decay with distance to origin) and an obstacle term based on distance to the nearest obstacle.
  - **`gradient_fk_potential(positions, obstacles, obstacle_radius, domain_radius, eps)`** returns the numerical gradient (finite differences) of the above potential.

### `swarm_dynamics.py`
- Defines the hybrid robot dynamics.
  - **Arneodo chaotic system**: constants `ARNEODO_DEFAULTS` and **`arneodo_deriv(t, xyz, alpha, beta, delta)`** returning the right‑hand side of the 3‑D Arneodo attractor.
  - ODE integrators:
    - **`solve_rk4(f, tspan, y0, n)`** – classic 4‑stage Runge‑Kutta.
    - **`solve_bdf3(f, tspan, y0, n)`** – BDF3 (backward differentiation formula of order 3) with RK3 start‑up and Newton‑like solve (`scipy.optimize.fsolve`) for each implicit step.
    - **`solve_theta_method(f, tspan, y0, n, theta)`** – theta method (e.g. Crank‑Nicolson for θ=0.5), solved implicitly with `fsolve`.
  - Class **`SwarmRobot`**:
    - Attributes: `position` (ndarray, 3), `velocity` (ndarray, 3), `internal` (ndarray, 3, Arneodo state).
    - Property `state` returns the concatenated vector `[position, velocity, internal]`; the setter unpacks a flat vector back into the attributes.
  - Helper **`repulsion_force(pi, pj, repulsion_range, repulsion_strength)`** computes a short‑range Lennard‑Jones‑like repulsion between two robots.
  - **`swarm_rhs(t, z, robots, control_gains, env_gradient_func, consensus_target)`** computes the time derivative of the flat swarm state vector, combining:
    - position → velocity,
    - velocity → PD‑like consensus term, velocity damping, environmental gradient following, and repulsion forces,
    - internal → Arne
