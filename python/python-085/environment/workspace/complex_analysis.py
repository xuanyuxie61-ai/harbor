import numpy as np
from typing import Tuple, List
from utils import c8_norm_l2, c8mat_norm_fro


def build_complex_mass_damping_stiffness(
    M: np.ndarray, C: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return M, C, K


def complex_modal_analysis(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = K.shape[0]

    Minv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def complex_damping_matrix_from_friction(
    K: np.ndarray, contact_nodes: np.ndarray,
    friction_coeff: float, normal_pressure: np.ndarray
) -> np.ndarray:
    n = K.shape[0]
    C = np.zeros((n, n))
    if len(normal_pressure) == 0:
        return C
    p_avg = np.mean(normal_pressure)
    v_ref = 1.0
    eta = friction_coeff * p_avg / v_ref
    for node in contact_nodes:
        gdof_x = 2 * node
        gdof_y = 2 * node + 1
        C[gdof_x, gdof_x] += eta * K[gdof_x, gdof_x]
        C[gdof_y, gdof_y] += eta * K[gdof_y, gdof_y] * 0.1
    return C


def stability_criterion(eigenvalues: np.ndarray) -> dict:
    alpha_max = float(np.max(np.real(eigenvalues)))
    unstable_count = int(np.sum(np.real(eigenvalues) > 0))

    valid = np.abs(eigenvalues) < 1e10
    if np.sum(valid) > 0:
        ev_valid = eigenvalues[valid]
        damping_ratios = -np.real(ev_valid) / (np.abs(ev_valid) + 1e-20)
        flutter_freqs = np.imag(ev_valid) / (2.0 * np.pi)
    else:
        damping_ratios = np.array([0.0])
        flutter_freqs = np.array([0.0])
    return {
        "alpha_max": alpha_max,
        "unstable_count": unstable_count,
        "min_damping_ratio": float(np.min(damping_ratios)),
        "max_flutter_freq_hz": float(np.max(np.abs(flutter_freqs))),
        "critical_modes": unstable_count
    }


def complex_matrix_power_iteration(A_complex: np.ndarray, max_iter: int = 50) -> complex:
    n = A_complex.shape[0]
    z = np.random.randn(n) + 1j * np.random.randn(n)
    z = z / (c8_norm_l2(z) + 1e-20)
    lam = 0.0 + 0.0j
    for _ in range(max_iter):
        w = A_complex @ z
        norm_w = c8_norm_l2(w)
        if norm_w < 1e-20:
            break
        z = w / norm_w
        lam_new = np.vdot(z, A_complex @ z)
        if abs(lam_new - lam) < 1e-12:
            lam = lam_new
            break
        lam = lam_new
    return complex(lam)


def frequency_response_function(K: np.ndarray, M: np.ndarray, C: np.ndarray,
                                 omega_range: np.ndarray,
                                 load_dof: int) -> np.ndarray:
    n = K.shape[0]
    response = np.zeros(len(omega_range))
    f = np.zeros(n)
    f[load_dof] = 1.0
    for idx, w in enumerate(omega_range):
        D = K - w ** 2 * M + 1j * w * C
        try:
            u = np.linalg.solve(D, f)
            response[idx] = np.abs(u[load_dof])
        except np.linalg.LinAlgError:
            response[idx] = 0.0
    return response
