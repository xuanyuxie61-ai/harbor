
import numpy as np
from scipy.special import gamma as gamma_func


def gegenbauer_polynomial_value(m, alpha, x):
    if alpha <= -0.5:
        raise ValueError("alpha 必须大于 -0.5")

    x = np.atleast_1d(x)
    n_points = len(x)
    C = np.zeros((m + 1, n_points))

    if m >= 0:
        C[0, :] = 1.0
    if m >= 1:
        C[1, :] = 2.0 * alpha * x

    for n in range(2, m + 1):
        C[n, :] = (
            (2.0 * n - 2.0 + 2.0 * alpha) * x * C[n - 1, :]
            + (-n + 2.0 - 2.0 * alpha) * C[n - 2, :]
        ) / n

    return C


def gegenbauer_norm_squared(n, alpha):
    from scipy.special import gamma, factorial
    h_n = (np.pi * 2.0**(1.0 - 2.0 * alpha) * gamma(n + 2.0 * alpha)
           / (factorial(n) * (n + alpha) * gamma(alpha)**2))
    return h_n


def sincn(x):
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)
    nz = np.abs(x) > 1e-15
    result[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    return result


def sinc_interpolation_1d(x_samples, f_samples, x_query):
    x_samples = np.asarray(x_samples, dtype=float)
    f_samples = np.asarray(f_samples, dtype=float)
    x_query = np.asarray(x_query, dtype=float)

    dx = np.mean(np.diff(x_samples))
    if dx <= 0:
        raise ValueError("采样点必须严格递增")

    f_query = np.zeros_like(x_query)
    for i, xs in enumerate(x_samples):
        f_query += f_samples[i] * sincn((x_query - xs) / dx)

    return f_query


def analytical_potential_spherical(r, theta, I_source, sigma, R_cochlea,
                                    n_terms=20, alpha=0.5):
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    if len(r) != len(theta):
        raise ValueError("r 和 theta 长度必须相同")

    x = np.cos(theta)
    C = gegenbauer_polynomial_value(n_terms, alpha, x)

    V = np.zeros_like(r)
    prefactor = I_source / (4.0 * np.pi * sigma * R_cochlea * 1e-3)

    for n in range(n_terms + 1):
        ratio = np.where(r < R_cochlea, r / R_cochlea, R_cochlea / r)
        V += prefactor * C[n, :] * ratio**n / (n + 1.0)

    return V


def cylindrical_potential_line_source(rho, z, z_e, I_e, sigma):
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    dist_mm = np.sqrt(rho**2 + (z - z_e)**2)
    dist_m = dist_mm * 1e-3

    dist_m = np.where(dist_m < 1e-6, 1e-6, dist_m)
    V = I_e / (4.0 * np.pi * sigma * dist_m)
    return V


def multi_electrode_superposition(electrode_positions, electrode_currents,
                                   query_points, sigma):
    electrode_positions = np.asarray(electrode_positions, dtype=float)
    electrode_currents = np.asarray(electrode_currents, dtype=float)
    query_points = np.asarray(query_points, dtype=float)

    V = np.zeros(query_points.shape[0])
    for pos, I in zip(electrode_positions, electrode_currents):
        if abs(I) < 1e-15:
            continue
        dists_mm = np.linalg.norm(query_points - pos, axis=1)
        dists_m = dists_mm * 1e-3
        dists_m = np.where(dists_m < 1e-6, 1e-6, dists_m)
        V += I / (4.0 * np.pi * sigma * dists_m)

    return V
