
import numpy as np
from typing import Tuple


def build_coupling_digraph(n_stages: int,
                           coupling_strength: float = 0.1,
                           phase_noise_std: float = 0.05) -> np.ndarray:
    if n_stages < 1:
        raise ValueError("n_stages 必须至少为 1。")
    if coupling_strength < 0.0:
        raise ValueError("coupling_strength 必须非负。")

    N = 2 * n_stages
    A = np.zeros((N, N), dtype=np.complex128)


    for stage in range(n_stages):
        s_idx = 2 * stage
        i_idx = 2 * stage + 1
        A[s_idx, s_idx] = 1.0
        A[i_idx, i_idx] = 1.0

        A[s_idx, i_idx] = coupling_strength * np.exp(1j * np.random.normal(0.0, phase_noise_std))
        A[i_idx, s_idx] = coupling_strength * np.exp(-1j * np.random.normal(0.0, phase_noise_std))


    for stage in range(n_stages - 1):
        s_curr = 2 * stage
        i_curr = 2 * stage + 1
        s_next = 2 * (stage + 1)
        i_next = 2 * (stage + 1) + 1
        t = 0.5 * coupling_strength
        A[s_next, s_curr] = t * np.exp(1j * np.random.normal(0.0, phase_noise_std))
        A[i_next, i_curr] = t * np.exp(1j * np.random.normal(0.0, phase_noise_std))

    return A


def adjacency_to_transition(A: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    col_sums = np.sum(np.abs(A), axis=0)
    T = np.zeros_like(A, dtype=np.complex128)
    for j in range(N):
        if col_sums[j] > 1e-15:
            T[:, j] = A[:, j] / col_sums[j]
        else:
            T[j, j] = 1.0
    return T


def network_photon_number_evolution(n_stages: int,
                                    n_initial: np.ndarray,
                                    source_terms: np.ndarray,
                                    A: np.ndarray) -> np.ndarray:
    N = 2 * n_stages
    if n_initial.shape != (N,):
        raise ValueError("n_initial 维度不匹配。")
    if source_terms.shape[1] != N:
        raise ValueError("source_terms 列数不匹配。")

    T = adjacency_to_transition(A)
    n_history = np.zeros((n_stages + 1, N), dtype=np.float64)
    n_history[0, :] = n_initial

    n_current = n_initial.astype(np.float64)
    for m in range(n_stages):

        n_current = np.abs(T @ n_current) + source_terms[m, :]

        n_current = np.maximum(n_current, 0.0)
        n_history[m + 1, :] = n_current

    return n_history


def transitive_closure_digraph(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    N = A.shape[0]
    C = (np.abs(A) > tol).astype(int)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                C[i, j] = C[i, j] or (C[i, k] and C[k, j])
    return C
