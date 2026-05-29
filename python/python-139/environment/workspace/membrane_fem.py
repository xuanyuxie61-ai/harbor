"""
One-dimensional finite-element discretization of the membrane concentration profile.

Adapted from fem1d_sample.m, including node bracketing and linear shape-function
evaluation.  Solves the steady-state and transient diffusion-reaction equations
across the active membrane layer.
"""

import numpy as np
from utils import bracket_interval, linear_interpolate
from sparse_matrix import solve_sparse_system, build_mass_matrix_csc, ge_to_ccs


def build_fem_mesh(L, Nx):
    """
    Generate a uniform 1-D mesh from 0 to L with Nx nodes.
    """
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    return x, dx


def build_stiffness_matrix(Nx, dx, D, reaction_rate):
    """
    Build the FEM stiffness matrix for 1-D diffusion with first-order reaction:
        -D d^2c/dx^2 + k c = 0
    Using linear hat functions and analytical element integration.
    """
    A = np.zeros((Nx, Nx), dtype=float)
    # Diffusion contribution
    diff_coeff = D / dx
    for i in range(1, Nx - 1):
        A[i, i] += 2.0 * diff_coeff
        A[i, i - 1] -= diff_coeff
        A[i, i + 1] -= diff_coeff
    A[0, 0] += diff_coeff
    A[0, 1] -= diff_coeff
    A[-1, -1] += diff_coeff
    A[-1, -2] -= diff_coeff

    # Reaction contribution (mass-matrix-like, lumped)
    react_coeff = reaction_rate * dx / 6.0
    for i in range(1, Nx - 1):
        A[i, i] += 2.0 * react_coeff
        A[i, i - 1] += react_coeff
        A[i, i + 1] += react_coeff
    A[0, 0] += react_coeff
    A[0, 1] += 0.5 * react_coeff
    A[-1, -1] += react_coeff
    A[-1, -2] += 0.5 * react_coeff

    return A


def apply_dirichlet_bc(A, b, idx, value):
    """
    Apply Dirichlet boundary condition c[idx] = value by modifying
    matrix A and RHS b in-place.
    """
    n = A.shape[0]
    for j in range(n):
        if j == idx:
            continue
        b[j] -= A[j, idx] * value
        A[j, idx] = 0.0
        A[idx, j] = 0.0
    A[idx, idx] = 1.0
    b[idx] = value
    return A, b


def apply_robin_bc(A, b, idx, neighbor_idx, h, D, c_inf):
    """
    Apply Robin (convective) boundary condition:
        -D dc/dx = h (c - c_inf)
    Discretized via two-point flux approximation at boundary.
    """
    dx = abs(h)  # dummy, not used directly; we use a film coefficient formulation
    # Simplified: treat as additional flux term
    # -D (c_nb - c_i)/dx = h (c_i - c_inf)
    # => -D c_nb + (D + h dx) c_i = h dx c_inf
    # For stability, we approximate dx as characteristic length
    dx_char = 1e-7
    coeff = D + h * dx_char
    A[idx, idx] += coeff
    A[idx, neighbor_idx] -= D
    b[idx] += h * dx_char * c_inf
    return A, b


def solve_steady_state_diffusion_reaction(L, Nx, D, k, c_feed, c_perm):
    """
    Solve steady-state 1-D diffusion-reaction across membrane thickness L.
    """
    x, dx = build_fem_mesh(L, Nx)
    A = build_stiffness_matrix(Nx, dx, D, k)
    b = np.zeros(Nx, dtype=float)
    # Dirichlet at both ends
    A, b = apply_dirichlet_bc(A, b, 0, c_feed)
    A, b = apply_dirichlet_bc(A, b, Nx - 1, c_perm)
    c = solve_sparse_system(A, b)
    return x, c


def solve_transient_diffusion_reaction(L, Nx, D, k, c_feed, c_perm, Nt, t_final,
                                        mass_matrix, c0=None):
    """
    Solve transient 1-D diffusion-reaction using backward Euler.
    Returns concentration profiles at all time steps.

    Parameters:
        mass_matrix: SparseCCS object representing the FEM consistent mass matrix.
    """
    # TODO Hole 1: 实现瞬态扩散-反应的 Backward Euler 求解
    # 步骤:
    #   1. 构建 FEM 网格: x, dx = build_fem_mesh(L, Nx)
    #   2. 计算时间步长: dt = t_final / Nt
    #   3. 构建稳态刚度矩阵: A_steady = build_stiffness_matrix(Nx, dx, D, k)
    #   4. 从 mass_matrix (SparseCCS) 提取密集矩阵: M = mass_matrix.to_dense()
    #   5. 构建时间步进矩阵: A_time = M + dt * A_steady
    #   6. 初始化浓度分布 c (若 c0 为 None 则用 c_perm 填充)
    #   7. 对 n = 1..Nt 执行 Backward Euler 步进:
    #        b = M.dot(c)
    #        应用 Dirichlet 边界条件 (apply_dirichlet_bc)
    #        c = solve_sparse_system(A_time, b)
    #   8. 返回 x, profiles
    raise NotImplementedError("Hole 1: 请实现瞬态扩散-反应的 Backward Euler 求解")


def evaluate_fem_solution(x_nodes, values, x_query):
    """
    Evaluate the FEM solution at arbitrary query points via linear interpolation.
    Adapted from fem1d_evaluate.
    """
    return linear_interpolate(x_nodes, values, x_query)


def compute_molar_flux(x, c, D):
    """
    Compute local diffusive molar flux J = -D dc/dx using central differences.
    """
    Nx = len(x)
    J = np.zeros(Nx, dtype=float)
    for i in range(1, Nx - 1):
        J[i] = -D * (c[i + 1] - c[i - 1]) / (x[i + 1] - x[i - 1])
    # One-sided at boundaries
    J[0] = -D * (c[1] - c[0]) / (x[1] - x[0])
    J[-1] = -D * (c[-1] - c[-2]) / (x[-1] - x[-2])
    return J


def compute_separation_factor(c_co2_feed, c_co2_perm, c_ch4_feed, c_ch4_perm):
    """
    Ideal separation factor alpha = (c_CO2,perm / c_CH4,perm) /
                                    (c_CO2,feed / c_CH4,feed)
    """
    if c_ch4_perm <= 0 or c_ch4_feed <= 0 or c_co2_feed <= 0:
        return 0.0
    alpha = (c_co2_perm / c_ch4_perm) / (c_co2_feed / c_ch4_feed)
    return alpha
