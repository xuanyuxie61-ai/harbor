
import numpy as np


def transfer_matrix_ssh(E, t1, t2, gamma):
    if abs(t2) < 1e-15:
        raise ValueError("t2 must be non-zero.")
    T = np.array([
        [(E - 1j * gamma) / t2, -t1 / t2],
        [1.0, 0.0]
    ], dtype=complex)
    return T


def spectrum_from_transfer_matrix(E_grid, t1, t2, gamma):
    traces = np.zeros(len(E_grid), dtype=complex)
    discriminants = np.zeros(len(E_grid), dtype=complex)
    for i, E in enumerate(E_grid):
        T = transfer_matrix_ssh(E, t1, t2, gamma)
        traces[i] = np.trace(T)
        discriminants[i] = np.trace(T) ** 2 - 4.0 * np.linalg.det(T)
    return traces, discriminants


def lyapunov_exponent_ssh(E, t1, t2, gamma, N=1000, seed=42):
    rng = np.random.default_rng(seed)

    t1_disorder = 0.1

    vec = np.array([1.0, 0.0], dtype=complex)
    norm_log = 0.0

    for _ in range(N):
        t1_n = t1 + t1_disorder * (rng.random() - 0.5)
        T = transfer_matrix_ssh(E, t1_n, t2, gamma)
        vec = T @ vec
        nv = np.linalg.norm(vec)
        if nv < 1e-30:
            return 1e308
        norm_log += np.log(nv)
        vec = vec / nv

    lyap = norm_log / N
    return lyap


def nonhermitian_markov_chain(N, p_forward, p_backward, loss_rate):
    if N <= 1:
        raise ValueError("N must be > 1.")
    L = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            L[i, i - 1] = p_backward
        if i < N - 1:
            L[i, i + 1] = p_forward
        L[i, i] = -(p_forward + p_backward + loss_rate)
    return L


def steady_state_distribution(L):
    N = L.shape[0]

    M = L.T.copy()
    M[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    pi = np.linalg.solve(M, b)

    pi = np.maximum(pi, 0.0)
    pi = pi / np.sum(pi)
    return pi
