
import numpy as np
from typing import Tuple


def mode_coupling_matrix(n_modes: int, delta_beta: float,
                         coupling_coeff: float) -> np.ndarray:
    K = np.zeros((n_modes, n_modes), dtype=complex)
    for m in range(n_modes):
        K[m, m] = delta_beta * (m - (n_modes - 1) / 2.0)
        if m + 1 < n_modes:
            K[m, m + 1] = coupling_coeff
            K[m + 1, m] = coupling_coeff
    return K


def xpm_coefficients(n_modes: int, gamma: float,
                      overlap_factors: np.ndarray) -> np.ndarray:
    chi = np.zeros((n_modes, n_modes), dtype=float)
    for m in range(n_modes):
        for n in range(n_modes):
            if m == n:
                chi[m, n] = gamma * overlap_factors[m, n]
            else:
                chi[m, n] = 2.0 * gamma * overlap_factors[m, n]
    return chi


def multimode_propagation_verlet(A0: np.ndarray, z_target: float,
                                  K: np.ndarray, chi: np.ndarray,
                                  dz: float) -> Tuple[np.ndarray, np.ndarray]:
    n_modes = len(A0)
    n_steps = int(np.ceil(z_target / dz)) + 1
    z_array = np.linspace(0.0, z_target, n_steps)
    A_history = np.zeros((n_steps, n_modes), dtype=complex)
    A = A0.copy()
    A_history[0, :] = A

    exp_K_half = np.linalg.matrix_power(np.eye(n_modes, dtype=complex), 1)

    from scipy.linalg import expm
    U_half = expm(-1j * K * dz * 0.5)
    for step in range(1, n_steps):

        A = U_half @ A

        for m in range(n_modes):
            phi = 0.0
            for n in range(n_modes):
                phi += chi[m, n] * abs(A[n]) ** 2
            A[m] *= np.exp(-1j * phi * dz)

        A = U_half @ A
        A_history[step, :] = A
    return z_array, A_history


def mode_power_orbits(A_history: np.ndarray) -> np.ndarray:
    return np.abs(A_history) ** 2


def orbital_angular_momentum_modes(l_values: np.ndarray,
                                    r_grid: np.ndarray,
                                    phi_grid: np.ndarray) -> np.ndarray:
    from scipy.special import genlaguerre
    n_l = len(l_values)
    n_r = len(r_grid)
    n_phi = len(phi_grid)
    modes = np.zeros((n_l, n_r, n_phi), dtype=complex)
    w0 = np.max(r_grid) / 2.0
    for il, l in enumerate(l_values):
        for ir, r in enumerate(r_grid):
            rho = np.sqrt(2.0) * r / w0
            radial = (rho ** abs(l)) * genlaguerre(0, abs(l))(rho ** 2) * np.exp(-rho ** 2 / 2.0)
            for ip, phi in enumerate(phi_grid):
                modes[il, ir, ip] = radial * np.exp(1j * l * phi)
    return modes
