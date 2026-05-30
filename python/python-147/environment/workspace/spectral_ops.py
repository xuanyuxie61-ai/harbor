
import numpy as np


def compute_wavenumbers(nx, L_domain):
    if nx % 2 != 0:
        raise ValueError("nx must be even")
    k = np.concatenate([
        np.arange(0, nx // 2),
        np.array([0.0]),
        np.arange(-nx // 2 + 1, 0)
    ]) * (2.0 * np.pi / L_domain)
    return k


def spectral_derivative(u, k, order=1):
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return u.copy()

    factor = (1j * k) ** order
    if u.ndim == 1:
        u_hat = np.fft.fft(u)
        return np.real(np.fft.ifft(factor * u_hat))
    elif u.ndim == 2:
        du = np.zeros_like(u)
        for j in range(u.shape[1]):
            u_hat = np.fft.fft(u[:, j])
            du[:, j] = np.real(np.fft.ifft(factor * u_hat))
        return du
    else:
        raise ValueError("u must be 1D or 2D")


def etdrk4_coefficients(L_op, dt, M=16):
    nx = len(L_op)
    E = np.exp(dt * L_op)
    E2 = np.exp(dt * L_op / 2.0)

    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L_op[:, np.newaxis] + r[np.newaxis, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

    return E, E2, Q, f1, f2, f3


def dealias_2_3_rule(v_hat):
    nx = len(v_hat)
    k_max = nx // 2
    cutoff = int(2.0 / 3.0 * k_max)
    v_filtered = v_hat.copy()
    v_filtered[cutoff:-cutoff] = 0.0
    return v_filtered


def compute_energy_spectrum(u, L_domain):
    nx = len(u)
    k = compute_wavenumbers(nx, L_domain)
    u_hat = np.fft.fft(u)
    E = 0.5 * np.abs(u_hat) ** 2
    return k, E


def kolmogorov_length_scale(u, L_domain):
    k = compute_wavenumbers(len(u), L_domain)
    u_x = spectral_derivative(u, k, order=1)
    u_xx = spectral_derivative(u, k, order=2)
    epsilon = np.mean(u_xx ** 2)
    eta = (1.0 / (epsilon + 1e-12)) ** 0.25
    return eta
