
import numpy as np
from typing import Tuple


def toep_cholesky_lower(n: int, first_row: np.ndarray) -> np.ndarray:
    if len(first_row) < n:
        raise ValueError("first_row 长度必须 >= n")
    first_row = np.asarray(first_row, dtype=np.float64)

    L = np.zeros((n, n), dtype=np.float64)
    L[0, 0] = np.sqrt(max(first_row[0], 1e-30))
    if n == 1:
        return L

    for j in range(1, n):




        sum_sq = 0.0
        for k in range(j):
            sum_sq += L[j, k] ** 2
        diag = first_row[0] - sum_sq
        L[j, j] = np.sqrt(max(diag, 1e-30))

        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]

            L[i, j] = (first_row[abs(i - j)] - s) / L[j, j]
    return L


def exponential_kernel(tau: np.ndarray, gamma0: float, tau_mem: float) -> np.ndarray:
    return gamma0 * np.exp(-np.abs(tau) / tau_mem)


def generate_correlated_forces(n_steps: int,
                               dt: float,
                               k_B: float = 8.314e-3,
                               T: float = 300.0,
                               gamma0: float = 1.0,
                               tau_mem: float = 0.1) -> np.ndarray:
    sigma = np.sqrt(k_B * T * gamma0)
    rho = np.exp(-dt / max(tau_mem, 1e-12))

    rho = min(rho, 0.999999999)
    z = np.random.randn(n_steps)
    forces = np.empty(n_steps, dtype=np.float64)
    forces[0] = sigma * z[0]
    coeff = sigma * np.sqrt(max(1.0 - rho ** 2, 0.0))
    for i in range(1, n_steps):
        forces[i] = rho * forces[i - 1] + coeff * z[i]
    return forces


def colored_noise_spectrum(forces: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(forces)
    fft_vals = np.fft.rfft(forces)
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.abs(fft_vals) ** 2 / (n * dt)
    return freqs, psd
