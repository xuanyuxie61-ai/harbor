
import numpy as np


def hankel_spd_cholesky_lower(n, lii, liim1):







    pass


def build_hankel_from_moments(moments):
    m = moments.size
    n = (m + 1) // 2
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            H[i, j] = moments[i + j]
    return H


def compute_spectral_moments_from_hankel(H):

    n = H.shape[0]
    mu = np.zeros(2 * n - 1, dtype=float)
    for k in range(2 * n - 1):
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                mu[k] = H[i, j]
                break

    mu_0 = mu[0] if mu.size > 0 else 1.0
    mu_1 = mu[1] if mu.size > 1 else 0.0
    mu_2 = mu[2] if mu.size > 2 else 0.0

    mean_freq = mu_1 / (mu_0 + 1e-30)
    spectral_width = np.sqrt(max(0.0, mu_2 / (mu_0 + 1e-30) - mean_freq ** 2))

    return {
        "bolometric_flux": mu_0,
        "mean_frequency": mean_freq,
        "spectral_width": spectral_width,
        "moments": mu,
    }


def synthetic_grb_moments(n):
    alpha = 0.5
    beta = 2.5
    nu_b = 1e15

    moments = np.zeros(2 * n - 1, dtype=float)
    for k in range(2 * n - 1):
        term1 = nu_b ** (k + 1 - alpha) / (k + 1 - alpha)
        term2 = nu_b ** (k + 1 - beta) / (beta - k - 1)
        moments[k] = term1 + term2

    return moments
