
import numpy as np
from utils import bracket_interval, linear_interpolate
from sparse_matrix import solve_sparse_system, build_mass_matrix_csc, ge_to_ccs


def build_fem_mesh(L, Nx):
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    return x, dx


def build_stiffness_matrix(Nx, dx, D, reaction_rate):
    A = np.zeros((Nx, Nx), dtype=float)

    diff_coeff = D / dx
    for i in range(1, Nx - 1):
        A[i, i] += 2.0 * diff_coeff
        A[i, i - 1] -= diff_coeff
        A[i, i + 1] -= diff_coeff
    A[0, 0] += diff_coeff
    A[0, 1] -= diff_coeff
    A[-1, -1] += diff_coeff
    A[-1, -2] -= diff_coeff


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
    dx = abs(h)




    dx_char = 1e-7
    coeff = D + h * dx_char
    A[idx, idx] += coeff
    A[idx, neighbor_idx] -= D
    b[idx] += h * dx_char * c_inf
    return A, b


def solve_steady_state_diffusion_reaction(L, Nx, D, k, c_feed, c_perm):
    x, dx = build_fem_mesh(L, Nx)
    A = build_stiffness_matrix(Nx, dx, D, k)
    b = np.zeros(Nx, dtype=float)

    A, b = apply_dirichlet_bc(A, b, 0, c_feed)
    A, b = apply_dirichlet_bc(A, b, Nx - 1, c_perm)
    c = solve_sparse_system(A, b)
    return x, c


def solve_transient_diffusion_reaction(L, Nx, D, k, c_feed, c_perm, Nt, t_final,
                                        mass_matrix, c0=None):













    raise NotImplementedError("Hole 1: 请实现瞬态扩散-反应的 Backward Euler 求解")


def evaluate_fem_solution(x_nodes, values, x_query):
    return linear_interpolate(x_nodes, values, x_query)


def compute_molar_flux(x, c, D):
    Nx = len(x)
    J = np.zeros(Nx, dtype=float)
    for i in range(1, Nx - 1):
        J[i] = -D * (c[i + 1] - c[i - 1]) / (x[i + 1] - x[i - 1])

    J[0] = -D * (c[1] - c[0]) / (x[1] - x[0])
    J[-1] = -D * (c[-1] - c[-2]) / (x[-1] - x[-2])
    return J


def compute_separation_factor(c_co2_feed, c_co2_perm, c_ch4_feed, c_ch4_perm):
    if c_ch4_perm <= 0 or c_ch4_feed <= 0 or c_co2_feed <= 0:
        return 0.0
    alpha = (c_co2_perm / c_ch4_perm) / (c_co2_feed / c_ch4_feed)
    return alpha
