
import numpy as np
import math
from utils import jacobi_solve, matrix_to_st, st_to_dense, safe_divide, clamp
from plasma_drude_model import C_LIGHT


def field_transfer_matrix(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
    E0: float = 1.0,
    theta: float = 0.0,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    eps_profile = np.asarray(eps_profile, dtype=complex)
    if z.size != eps_profile.size:
        raise ValueError("z and eps_profile must have equal length.")
    if z.size < 2:
        raise ValueError("At least two points are required.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing.")

    theta = clamp(float(theta), 0.0, math.pi / 2.0 - 1e-6)
    k0 = omega / C_LIGHT
    N = z.size



    E = np.zeros(N, dtype=complex)
    E[0] = complex(E0)



    for j in range(1, N):
        dz = z[j] - z[j - 1]
        eps_mid = 0.5 * (eps_profile[j] + eps_profile[j - 1])
        kz = k0 * np.sqrt(eps_mid)

        E[j] = E[j - 1] * np.exp(1j * kz * dz)

    return E


def build_fd_matrix(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
    boundary_type: str = "PEC",
) -> tuple:
    z = np.asarray(z, dtype=float)
    eps_profile = np.asarray(eps_profile, dtype=complex)
    N = z.size
    if N < 3:
        raise ValueError("Need at least 3 grid points for FD.")

    k0 = omega / C_LIGHT
    A = np.zeros((N, N), dtype=complex)
    b = np.zeros(N, dtype=complex)



    for j in range(1, N - 1):
        hj = z[j] - z[j - 1]
        hj1 = z[j + 1] - z[j]
        h_sum = hj + hj1

        coeff_jm1 = -2.0 / (hj * h_sum)
        coeff_jp1 = -2.0 / (hj1 * h_sum)
        coeff_j = 2.0 / (hj * hj1) + k0 ** 2 * eps_profile[j]

        A[j, j - 1] = coeff_jm1
        A[j, j] = coeff_j
        A[j, j + 1] = coeff_jp1
        b[j] = 0.0




    A[0, 0] = 1.0
    b[0] = 1.0



    h_end = z[-1] - z[-2]
    k_end = k0 * np.sqrt(eps_profile[-1])
    A[-1, -2] = -1.0 / h_end
    A[-1, -1] = 1.0 / h_end + 1j * k_end
    b[-1] = 0.0

    return A, b


def solve_fd_jacobi(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
    max_iter: int = 10000,
    tol: float = 1e-10,
) -> tuple:
    A, b = build_fd_matrix(z, eps_profile, omega, boundary_type="absorbing")
    N = A.shape[0]


    A_real = np.zeros((2 * N, 2 * N), dtype=float)
    b_real = np.zeros(2 * N, dtype=float)

    A_real[:N, :N] = A.real
    A_real[:N, N:] = -A.imag
    A_real[N:, :N] = A.imag
    A_real[N:, N:] = A.real

    b_real[:N] = b.real
    b_real[N:] = b.imag

    x0 = np.zeros(2 * N, dtype=float)
    x0[0] = 1.0

    x_sol, res, it, conv = jacobi_solve(A_real, b_real, x0=x0, max_iter=max_iter, tol=tol)

    E = x_sol[:N] + 1j * x_sol[N:]
    return E, res, it, conv


def solve_fd_direct(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
) -> np.ndarray:
    A, b = build_fd_matrix(z, eps_profile, omega, boundary_type="absorbing")
    E = np.linalg.solve(A, b)
    return E


def compute_power_density(
    E: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
) -> np.ndarray:
    EPS_0 = 8.854187817e-12
    E = np.asarray(E, dtype=complex)
    eps_profile = np.asarray(eps_profile, dtype=complex)
    if E.shape != eps_profile.shape:
        raise ValueError("E and eps_profile must have the same shape.")

    eps_i = eps_profile.imag
    eps_i = np.where(eps_i < 0, 0.0, eps_i)
    P_abs = 0.5 * omega * EPS_0 * eps_i * np.abs(E) ** 2
    return P_abs


def fd_matrix_to_st(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
) -> tuple:
    A, _ = build_fd_matrix(z, eps_profile, omega, boundary_type="absorbing")
    rows, cols, vals = matrix_to_st(A)
    return rows, cols, vals, A.shape
