
import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple


def concurrence_from_purity(purity: float) -> float:


    raise NotImplementedError("Hole 2: 请实现 concurrence_from_purity")


def von_neumann_entropy_schmidt(lambdas: np.ndarray) -> float:
    lambdas = np.asarray(lambdas)
    lambdas = lambdas[lambdas > 1e-15]
    S = -np.sum(lambdas * np.log2(lambdas))
    return max(0.0, S)


def hom_visibility(jsa_real: np.ndarray, jsa_imag: np.ndarray,
                   delay_grid: np.ndarray,
                   omega_s: np.ndarray,
                   omega_i: np.ndarray) -> Tuple[np.ndarray, float]:
    jsa = jsa_real + 1j * jsa_imag
    n_s, n_i = jsa.shape


    f2 = np.abs(jsa) ** 2


    Os, Oi = np.meshgrid(omega_s, omega_i, indexing='ij')
    dw = Os - Oi


    R_tau = np.zeros_like(delay_grid, dtype=np.float64)
    for idx, tau in enumerate(delay_grid):
        cos_kernel = np.cos(dw * tau)
        R_tau[idx] = 1.0 - np.sum(f2 * cos_kernel) / np.sum(f2)

    R_max = np.max(R_tau)
    R_min = np.min(R_tau)
    denom = R_max + R_min
    V = (R_max - R_min) / denom if denom > 1e-15 else 0.0
    return R_tau, V


def state_fidelity_target(jsa: np.ndarray,
                          target_type: str = "singlet") -> float:
    jsa = np.asarray(jsa)
    n_s, n_i = jsa.shape
    if n_s != n_i:

        n = min(n_s, n_i)
        jsa = jsa[:n, :n]

    if target_type == "singlet":



        target = np.zeros_like(jsa)
        target[0, 1] = 1.0 / np.sqrt(2.0)
        target[1, 0] = -1.0 / np.sqrt(2.0)
    elif target_type == "triplet":
        target = np.zeros_like(jsa)
        target[0, 0] = 1.0 / np.sqrt(2.0)
        target[1, 1] = 1.0 / np.sqrt(2.0)
    else:
        raise ValueError(f"未知目标类型: {target_type}")

    overlap = np.sum(np.conj(target) * jsa)
    F = np.abs(overlap) ** 2
    return np.clip(F, 0.0, 1.0)


def chsh_parameter(correlation_matrix: np.ndarray) -> float:
    if correlation_matrix.shape != (4, 4):
        raise ValueError("correlation_matrix 必须为 4x4。")
    E = np.zeros((2, 2), dtype=np.float64)
    settings_a = [0, 2]
    settings_b = [0, 2]
    for i, ai in enumerate(settings_a):
        for j, bj in enumerate(settings_b):
            Npp = correlation_matrix[ai, bj]
            Nmm = correlation_matrix[ai + 1, bj + 1]
            Npm = correlation_matrix[ai, bj + 1]
            Nmp = correlation_matrix[ai + 1, bj]
            total = Npp + Nmm + Npm + Nmp
            if total > 1e-15:
                E[i, j] = (Npp + Nmm - Npm - Nmp) / total
            else:
                E[i, j] = 0.0

    S = abs(E[0, 0] - E[0, 1] + E[1, 0] + E[1, 1])
    return S
