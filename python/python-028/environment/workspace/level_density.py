
import numpy as np
from math import sqrt, pi, exp, log, erf


def bethe_formula(E, a_parameter, spin=None, parity=None):
    E = np.asarray(E, dtype=float)
    rho = np.zeros_like(E)
    mask = E > 1e-6

    if np.any(mask):
        Em = E[mask]
        sqrt_aE = np.sqrt(a_parameter * Em)
        rho_base = (1.0 / (12.0 * np.sqrt(a_parameter))) * np.exp(2.0 * sqrt_aE) / (Em ** (5.0 / 4.0))

        if spin is not None:
            sigma2 = 6.0 * sqrt_aE / (pi ** 2)
            spin_factor = (2.0 * spin + 1.0) / (2.0 * np.sqrt(2.0 * sigma2))
            spin_factor *= np.exp(-(spin + 0.5) ** 2 / (2.0 * sigma2))
            rho[mask] = rho_base * spin_factor
        else:
            rho[mask] = rho_base

    return rho


def bcs_level_density(E, a_parameter, delta, spin=None):
    E_eff = E + delta
    rho_fg = bethe_formula(E_eff, a_parameter, spin)
    sqrt_aE = np.sqrt(a_parameter * np.maximum(E_eff, 1e-10))
    suppression = np.tanh(sqrt_aE * delta / np.maximum(E_eff, 1e-10))
    return rho_fg * suppression


def log_normal_pdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)
    mask = x > 0
    if np.any(mask):
        xm = x[mask]
        pdf[mask] = (1.0 / (xm * sigma * sqrt(2.0 * pi))) * np.exp(
            -((np.log(xm) - mu) ** 2) / (2.0 * sigma ** 2)
        )
    return pdf


def log_normal_cdf(x, mu, sigma):
    if x <= 0:
        return 0.0
    return 0.5 * (1.0 + erf((log(x) - mu) / (sigma * sqrt(2.0))))


def log_normal_sample(mu, sigma, size=1, seed=None):
    rng = np.random.default_rng(seed)
    normal_samples = rng.normal(loc=mu, scale=sigma, size=size)
    return np.exp(normal_samples)


def level_spacing_distribution(s, regime='goe'):
    s = np.asarray(s, dtype=float)
    if regime == 'poisson':
        return np.exp(-s)
    elif regime == 'goe':
        return (pi / 2.0) * s * np.exp(-pi * s * s / 4.0)
    elif regime == 'gue':
        return (32.0 / pi ** 2) * s * s * np.exp(-4.0 * s * s / pi)
    elif regime == 'gse':
        return (2.0 ** 18 / (3.0 ** 6 * pi ** 3)) * s ** 4 * np.exp(-64.0 * s * s / (9.0 * pi))
    else:
        raise ValueError("regime 必须是 'poisson', 'goe', 'gue', 'gse' 之一")


def unfolding_spectrum(energies):
    E_sorted = np.sort(energies)
    n_levels = len(E_sorted)


    N_cum = np.arange(1, n_levels + 1)



    mask = E_sorted > 1e-3
    if np.sum(mask) > 3:
        logE = np.log(E_sorted[mask])
        logN = np.log(N_cum[mask])
        coeffs = np.polyfit(logE, logN, deg=2)
        poly = np.poly1d(coeffs)

        def N_smooth(E):
            Ea = np.asarray(E, dtype=float)
            val = np.exp(poly(np.log(np.maximum(Ea, 1e-10))))
            return val
    else:
        def N_smooth(E):
            return np.asarray(E, dtype=float) / np.mean(np.diff(E_sorted))

    epsilon = N_smooth(E_sorted)
    s = np.diff(epsilon)

    if len(s) > 0 and np.mean(s) > 0:
        s = s / np.mean(s)

    return s, N_smooth


def nuclear_level_density_parameter(A, shell_correction=0.0):
    a_base = A / 8.0

    a_eff = a_base * (1.0 + 0.1 * shell_correction)
    return max(a_eff, 0.1)


def total_level_density_table(A, E_max=20.0, n_points=100):
    a = nuclear_level_density_parameter(A)
    delta = 12.0 / sqrt(A)
    energies = np.linspace(0.5, E_max, n_points)

    rho_total = bethe_formula(energies, a)
    rho_bcs = bcs_level_density(energies, a, delta)


    f_pos = 0.5 + 0.5 * np.exp(-energies / 5.0)
    rho_positive = rho_bcs * f_pos
    rho_negative = rho_bcs * (1.0 - f_pos)

    return energies, rho_total, rho_positive, rho_negative
