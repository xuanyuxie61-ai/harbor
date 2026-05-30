# -*- coding: utf-8 -*-
import numpy as np




H_BAR = 1.0
E_CHARGE = 1.0
MU_B = 0.5




def safe_exp(x, max_val=700.0):
    x = np.asarray(x, dtype=float)
    if np.any(x > max_val):

        result = np.empty_like(x)
        mask = x > max_val
        result[~mask] = np.exp(x[~mask])
        result[mask] = np.exp(max_val) * (1.0 + (x[mask] - max_val) / max_val)
        return result
    return np.exp(x)


def safe_log(x, eps=1e-300):
    x = np.asarray(x, dtype=float)
    x_safe = np.where(x > eps, x, eps)
    return np.log(x_safe)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros_like(v)
    return v / norm


def condition_number(A):
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0 or s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]


def gram_schmidt_qr(V, tol=1e-12):
    V = np.asarray(V, dtype=complex)
    N, k = V.shape
    Q = np.zeros((N, k), dtype=complex)
    R = np.zeros((k, k), dtype=complex)
    for j in range(k):
        v = V[:, j].copy()
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v < tol:

            v = np.random.randn(N) + 1j * np.random.randn(N)
            for i in range(j):
                v = v - np.vdot(Q[:, i], v) * Q[:, i]
            norm_v = np.linalg.norm(v)
        R[j, j] = norm_v
        Q[:, j] = v / norm_v
    return Q, R


def fermi_dirac(E, mu, T, eps=1e-12):
    T = max(T, eps)
    beta = 1.0 / T
    arg = beta * (E - mu)

    arg = np.clip(arg, -700.0, 700.0)
    return 1.0 / (np.exp(arg) + 1.0)


def magnetic_length(B, m_star=1.0):
    if B <= 0:
        raise ValueError("磁场强度 B 必须为正")
    return np.sqrt(H_BAR / (E_CHARGE * B))


def cyclotron_frequency(B, m_star=1.0):
    if B <= 0:
        raise ValueError("磁场强度 B 必须为正")
    return E_CHARGE * B / m_star


def landau_level_energy(n, B, m_star=1.0):
    omega_c = cyclotron_frequency(B, m_star)
    return H_BAR * omega_c * (n + 0.5)


def filling_factor(N_e, B, A, m_star=1.0):
    if B <= 0 or A <= 0 or N_e < 0:
        raise ValueError("参数必须满足 B>0, A>0, N_e>=0")
    flux_quantum = 2.0 * np.pi * H_BAR / E_CHARGE
    N_phi = B * A / flux_quantum
    if N_phi < 1e-15:
        return np.inf
    return N_e / N_phi


def gaussian_2d(x, y, x0, y0, sigma):
    dx = x - x0
    dy = y - y0
    r2 = dx * dx + dy * dy
    return np.exp(-r2 / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma)
