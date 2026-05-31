# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Non-Hermitian Physics & Exceptional Points – Project Description

This project implements a computational framework for non‑Hermitian quantum systems, with a focus on exceptional points, biorthogonal topology, and related numerical methods. The project consists of a main driver script `main.py` and a set of supporting modules. Only `main.py` will be preserved; all other Python files must be re‑implemented. The following sections describe the required modules and their interfaces.

---

## Module Overview

### 1. `hamiltonian_builder.py`
Constructs non‑Hermitian tight‑binding Hamiltonians in momentum space for several common models.
- **2×2 Pauli matrices** are used as building blocks.
- **Functions to implement:**
  - `build_pt_symmetric_hamiltonian_1d(k, t, m, gamma)` – 1D PT‑symmetric two‑band model.
  - `build_pt_symmetric_hamiltonian_2d(kx, ky, t, m, gamma, a)` – 2D square‑lattice non‑Hermitian model.
  - `build_nonhermitian_ssh_hamiltonian(k, t1, t2, gamma)` – Non‑Hermitian SSH model.
  - `build_nonhermitian_hofstadter_hamiltonian(kx, ky, phi, t, gamma, q)` – Hofstadter model with flux and gain/loss.
  - `characteristic_polynomial_2x2(H)` – returns coefficients of the characteristic polynomial of a 2×2 matrix.
  - `discriminant_2x2(H)` – evaluates the discriminant of a 2×2 Hamiltonian.

### 2. `exceptional_point_solver.py`
Finds exceptional points (EPs) where eigenvectors coalesce.
- **Laguerre root‑finding** on the discriminant (or its derivatives) is used to locate EPs in the complex plane.
- **Functions to implement:**
  - `laguerre_root_find(f, x0, degree, abserr, kmax)` – complex root‑finding of a function and its first two derivatives.
  - `find_exceptional_points_1d(t, m, gamma, k_guess_grid)` – EPs of the 1D PT‑symmetric Hamiltonian.
  - `find_exceptional_points_ssh(t1, t2, gamma, k_guess_grid)` – EPs of the non‑Hermitian SSH model.
  - `local_exceptional_point_order(H, param, dH_dparam, eps)` – estimates the order of an EP from the discriminant.

### 3. `biorthogonal_topology.py`
Computes biorthogonal Berry phases, curvatures, and winding numbers for non‑Hermitian Hamiltonians.
- **Biorthogonal normalization** of left and right eigenvectors is used.
- **Functions to implement:**
  - `compute_biorthogonal_eigenvectors(H)` – returns eigenvalues, right eigenvectors, and left eigenvectors with biorthogonal normalization.
  - `berry_connection_1d(H_func, k, dk)` – Berry connection for a 1D model using finite differences.
  - `berry_curvature_2d(H_func, kx, ky, dk)` – Berry curvature in 2D.
  - `zak_phase_1d(H_func, k_points, a)` – numerical Zak phase via integration of the connection.
  - `chern_number_2d(H_func, kx_points, ky_points, dk)` – integrated Berry curvature over the 2D Brillouin zone.
  - `winding_number_complex_energy(H_func, k_points)` – winding number of the complex energy band around the origin.

### 4. `brillouin_integrator.py`
Performs numerical integration over the 3D Brillouin zone using tetrahedral decomposition and symmetric quadrature rules.
- **Precomputed NCO (Newton‑Cotes Open) quadrature rules** on the reference tetrahedron are used (degrees 3 and 5).
- **Functions to implement:**
  - `reference_to_physical_t4(ref_points, tetra)` – maps reference points to a physical tetrahedron.
  - `tetrahedron_volume(tetra)` – volume of a tetrahedron.
  - `integrate_over_tetrahedron(f, tetra, degree)` – integrates a function over a tetrahedron using the chosen quadrature.
  - `partition_bz_into_tetrahedra(n_k)` – subdivides the cubic BZ into tetrahedra.
  - `integrate_bz_3d(f, n_k, degree)` – integrates a function over the whole BZ.
  - `bz_average_energy(H_func, n_k, degree)` – computes the average ground‑state energy over the 3D BZ.

### 5. `nonherm_dynamics.py`
Open‑system time evolution using non‑Hermitian Schrödinger equations and Lindblad master equations.
- **Adaptive Runge‑Kutta‑Fehlberg (RKF45)** is used for stiff/complex ODEs.
- **Functions to implement:**
  - `rkf45_step_complex(f, t, y, h, tol)` – one adaptive step of RKF45 for complex state vectors.
  - `evolve_nonhermitian_schrodinger(H_eff, psi0, t_span, dt0, tol)` – solves i∂_t ψ = H_eff ψ.
  - `lindblad_evolve_2level(H, L_list, r
