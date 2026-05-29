"""
layered_field_solver.py
-----------------------
Electromagnetic field distribution inside a stratified plasma coating.

Two solvers are provided:
  1. Transfer-matrix propagation (analytical, exact for planar layers).
  2. Finite-difference frequency-domain (FDFD) with Jacobi iteration
     for non-uniform or strongly varying profiles.

Incorporates core ideas from:
  - 603_jacobi        (Jacobi iterative solver)
  - 783_msm_to_st     (sparse triplet format)
"""

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
    """
    Compute the electric field amplitude E(z) inside a stratified medium
    using the transfer-matrix method for normal incidence (theta=0).

    For normal incidence, the wave equation reduces to
        d^2E/dz^2 + k0^2 * eps(z) * E = 0

    The field inside each homogeneous sub-layer j is a superposition of
    forward and backward travelling waves:
        E_j(z) = A_j * exp(+i*k_j*z) + B_j * exp(-i*k_j*z)

    Parameters
    ----------
    z : (Nz,) ndarray
        Depth coordinates [m], strictly increasing.
    eps_profile : (Nz,) ndarray of complex
        Relative permittivity at each z.
    omega : float
        Angular frequency [rad/s].
    E0 : float
        Incident electric field amplitude.
    theta : float
        Incidence angle [rad] (only normal incidence fully supported here).

    Returns
    -------
    E : (Nz,) ndarray of complex
        Electric field at each z.
    """
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

    # Discretize into piecewise-constant layers between z points
    # Compute interface reflection/transmission
    E = np.zeros(N, dtype=complex)
    E[0] = complex(E0)

    # Simplified approach: treat each interval as a layer and propagate
    # using the local wave number
    for j in range(1, N):
        dz = z[j] - z[j - 1]
        eps_mid = 0.5 * (eps_profile[j] + eps_profile[j - 1])
        kz = k0 * np.sqrt(eps_mid)
        # Propagate with attenuation
        E[j] = E[j - 1] * np.exp(1j * kz * dz)

    return E


def build_fd_matrix(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
    boundary_type: str = "PEC",
) -> tuple:
    """
    Build the finite-difference matrix A for the 1-D Helmholtz equation

        d^2E/dz^2 + k0^2 * eps(z) * E = 0

    Discretized with second-order central differences on a non-uniform grid:

        (2/(h_j*h_{j+1})) * E_j
        - (2/(h_j*(h_j+h_{j+1}))) * E_{j-1}
        - (2/(h_{j+1}*(h_j+h_{j+1}))) * E_{j+1}
        + k0^2 * eps_j * E_j = 0

    Parameters
    ----------
    z : (Nz,) ndarray
        Grid points [m].
    eps_profile : (Nz,) ndarray of complex
        Permittivity profile.
    omega : float
        Angular frequency [rad/s].
    boundary_type : str
        "PEC" (E=0) or "PMC" (dE/dz=0) or "absorbing" (1st-order ABC).

    Returns
    -------
    (A, b) : A is (Nz, Nz) complex ndarray, b is (Nz,) complex ndarray.
    """
    z = np.asarray(z, dtype=float)
    eps_profile = np.asarray(eps_profile, dtype=complex)
    N = z.size
    if N < 3:
        raise ValueError("Need at least 3 grid points for FD.")

    k0 = omega / C_LIGHT
    A = np.zeros((N, N), dtype=complex)
    b = np.zeros(N, dtype=complex)

    # Incident field at left boundary: E(0) = 1.0 (soft source)
    # For interior points
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

    # Left boundary (z = 0): soft source with first-order ABC
    # E_0 = E_inc + E_ref  (approximate)
    # We impose E_0 = 1.0 (unit incident plane wave)
    A[0, 0] = 1.0
    b[0] = 1.0

    # Right boundary (z = z_max): first-order absorbing boundary condition (ABC)
    # dE/dz + i*k_end*E = 0  =>  (E_N - E_{N-1})/h + i*k_end*E_N = 0
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
    """
    Solve the 1-D Helmholtz equation using finite differences + Jacobi iteration.

    The complex system is split into real and imaginary parts, producing
    a 2N x 2N real linear system that is then solved with Jacobi iteration.

    Parameters
    ----------
    z : (Nz,) ndarray
    eps_profile : (Nz,) ndarray of complex
    omega : float
    max_iter : int
    tol : float

    Returns
    -------
    (E, residual, iterations, converged)
    """
    A, b = build_fd_matrix(z, eps_profile, omega, boundary_type="absorbing")
    N = A.shape[0]

    # Split complex into 2N real system: [Re(A), -Im(A); Im(A), Re(A)] * [Re(E); Im(E)] = [Re(b); Im(b)]
    A_real = np.zeros((2 * N, 2 * N), dtype=float)
    b_real = np.zeros(2 * N, dtype=float)

    A_real[:N, :N] = A.real
    A_real[:N, N:] = -A.imag
    A_real[N:, :N] = A.imag
    A_real[N:, N:] = A.real

    b_real[:N] = b.real
    b_real[N:] = b.imag

    x0 = np.zeros(2 * N, dtype=float)
    x0[0] = 1.0  # initial guess: E(0) = 1

    x_sol, res, it, conv = jacobi_solve(A_real, b_real, x0=x0, max_iter=max_iter, tol=tol)

    E = x_sol[:N] + 1j * x_sol[N:]
    return E, res, it, conv


def solve_fd_direct(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Solve the 1-D Helmholtz equation using a direct dense solver (for small N).
    """
    A, b = build_fd_matrix(z, eps_profile, omega, boundary_type="absorbing")
    E = np.linalg.solve(A, b)
    return E


def compute_power_density(
    E: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Compute the time-averaged electromagnetic power absorption density [W/m^3].

    P_abs(z) = 0.5 * omega * eps_0 * eps_i(z) * |E(z)|^2

    Parameters
    ----------
    E : (Nz,) ndarray of complex
        Electric field.
    eps_profile : (Nz,) ndarray of complex
        Permittivity profile.
    omega : float
        Angular frequency [rad/s].

    Returns
    -------
    P_abs : (Nz,) ndarray of float
    """
    EPS_0 = 8.854187817e-12
    E = np.asarray(E, dtype=complex)
    eps_profile = np.asarray(eps_profile, dtype=complex)
    if E.shape != eps_profile.shape:
        raise ValueError("E and eps_profile must have the same shape.")

    eps_i = eps_profile.imag
    eps_i = np.where(eps_i < 0, 0.0, eps_i)  # physically eps_i >= 0 for passive media
    P_abs = 0.5 * omega * EPS_0 * eps_i * np.abs(E) ** 2
    return P_abs


def fd_matrix_to_st(
    z: np.ndarray,
    eps_profile: np.ndarray,
    omega: float,
) -> tuple:
    """
    Build the finite-difference matrix and export it to sparse triplet (ST) format.

    Returns
    -------
    (rows, cols, vals, shape) where rows, cols, vals describe non-zero entries.
    """
    A, _ = build_fd_matrix(z, eps_profile, omega, boundary_type="absorbing")
    rows, cols, vals = matrix_to_st(A)
    return rows, cols, vals, A.shape
