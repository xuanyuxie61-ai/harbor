
import numpy as np
from typing import Tuple, Callable


def von_neumann_amplification_ftcs(
    c: float,
    dt: float,
    dx: float,
    k_values: np.ndarray
) -> np.ndarray:
    if dx <= 0 or dt <= 0:
        raise ValueError("dx and dt must be positive")

    cfl = c * dt / dx
    G = 1.0 - 1j * cfl * np.sin(k_values * dx)
    return np.abs(G)


def cfl_condition_hyperbolic(
    wave_speed: float,
    dx: float
) -> float:
    if dx <= 0:
        raise ValueError("dx must be positive")
    if abs(wave_speed) < 1e-15:
        return np.inf
    return dx / abs(wave_speed)


def diffusion_stability_limit(
    D: float,
    dx: float,
    dimension: int = 1
) -> float:
    if D < 0 or dx <= 0:
        raise ValueError("D must be non-negative and dx positive")
    if dimension not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2, or 3")

    factor = {1: 2.0, 2: 4.0, 3: 6.0}[dimension]
    return dx * dx / (factor * D + 1e-15)


def matrix_spectral_radius(A: np.ndarray) -> float:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))


def is_stable_matrix(A: np.ndarray, tol: float = 1.0 + 1e-10) -> bool:
    return matrix_spectral_radius(A) <= tol


def analyze_kernel_matrix_stability(
    K: np.ndarray,
    reg_values: np.ndarray = None
) -> dict:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")

    eigvals = np.linalg.eigvalsh(K)
    pos_eigvals = eigvals[eigvals > 1e-12]

    if len(pos_eigvals) == 0:
        return {
            "condition_number": np.inf,
            "smallest_eigenvalue": 0.0,
            "rank_estimate": 0,
            "recommended_reg": 1e-6,
            "is_well_conditioned": False
        }

    cond = np.max(eigvals) / np.min(pos_eigvals)
    rank = len(pos_eigvals)


    recommended_reg = max(10.0 * np.min(pos_eigvals), 1e-10)


    is_well = cond < 1e12

    return {
        "condition_number": cond,
        "smallest_eigenvalue": np.min(pos_eigvals),
        "rank_estimate": rank,
        "recommended_reg": recommended_reg,
        "is_well_conditioned": is_well
    }


def trotter_error_bound(
    H: np.ndarray,
    dt: float,
    order: int = 1
) -> float:
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")

    norm_H = np.linalg.norm(H, ord=2)

    if order == 1:

        error = dt * dt * norm_H ** 2 / 2.0
    elif order == 2:

        error = dt ** 3 * norm_H ** 3 / 6.0
    else:

        error = dt ** (order + 1) * norm_H ** (order + 1)

    return error


def quantum_kernel_robustness_score(
    K: np.ndarray,
    noise_level: float = 0.01
) -> float:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")

    n = K.shape[0]
    orig_cond = np.linalg.cond(K)


    noise = noise_level * np.random.randn(n, n)
    noise = (noise + noise.T) / 2.0
    K_noisy = K + noise

    noisy_cond = np.linalg.cond(K_noisy)


    ratio = noisy_cond / (orig_cond + 1e-15)


    score = max(0.0, 1.0 - np.log10(ratio) / 10.0)
    return score
