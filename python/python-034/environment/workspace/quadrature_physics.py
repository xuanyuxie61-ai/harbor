
import numpy as np


def chebyshev_even_coeffs(n: int, f):
    j = np.arange(n + 1)
    x = np.cos(j * np.pi / n)
    fx = f(x)

    s = n // 2
    a2 = np.zeros(s + 1)
    for r in range(s + 1):
        val = 0.0
        for j_idx in range(n + 1):
            weight = 1.0 if (j_idx == 0 or j_idx == n) else 2.0
            val += weight * fx[j_idx] * np.cos(2 * r * j_idx * np.pi / n)
        a2[r] = val / (2 * n)
    return a2


def gegenbauer_cc(n: int, lambda_param: float, f) -> float:
    if lambda_param <= -0.5:
        raise ValueError("lambda must be > -0.5")
    a2 = chebyshev_even_coeffs(n, f)
    s = n // 2
    sigma = n % 2
    u = 0.5 * (sigma + 1.0) * a2[s]
    for rh in range(s - 1, 0, -1):
        u = (rh - lambda_param) / (rh + lambda_param + 1.0) * u + a2[rh]
    u = -lambda_param * u / (lambda_param + 1.0) + 0.5 * a2[0]

    from math import gamma, sqrt, pi
    value = gamma(lambda_param + 0.5) * sqrt(np.pi) * u / gamma(lambda_param + 1.0)
    return value


def alpert_log_rule(f, n: int = 8) -> float:

    eps = 1e-6

    m = max(n, 4)
    if m % 2 == 1:
        m += 1
    h = (1.0 - eps) / m
    x = np.linspace(eps, 1.0, m + 1)
    y = np.log(x) * f(x)

    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:m:2])
    integral += 2.0 * np.sum(y[2:m-1:2])
    integral *= h / 3.0

    pv = eps * (np.log(eps) - 1.0) * f(0.0)
    return integral + pv


def alpert_power_rule(f, alpha: float, n: int = 8) -> float:
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1")
    eps = 1e-6
    m = max(n, 4)
    if m % 2 == 1:
        m += 1
    h = (1.0 - eps) / m
    x = np.linspace(eps, 1.0, m + 1)
    y = np.power(x, alpha) * f(x)
    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:m:2])
    integral += 2.0 * np.sum(y[2:m-1:2])
    integral *= h / 3.0
    pv = (eps ** (alpha + 1.0)) / (alpha + 1.0) * f(0.0)
    return integral + pv


def decay_constant_integral(meson_mass: float, pion_mass: float,
                            lattice_spacing: float = 1.0) -> float:
    def integrand(k):
        return (1.0 - np.cos(k)) / (pion_mass ** 2 + 2.0 * (1.0 - np.cos(k)) + 1e-10)



    def f_g(x):
        return integrand(0.5 * np.pi * (x + 1.0))

    try:
        val = gegenbauer_cc(32, 0.5, f_g)
    except Exception:

        x = np.linspace(-1, 1, 1000)
        val = np.trapezoid(f_g(x), x)

    f_pi = 0.5 * val / np.pi

    return f_pi * lattice_spacing


def self_energy_integral(mass: float, cutoff: float = np.pi) -> float:
    def radial_integrand(k):

        return k ** 3 / (k ** 2 + mass ** 2 + 1e-10)



    def f_low(x):
        x = np.atleast_1d(x)
        result = np.zeros_like(x, dtype=float)
        mask = x > 1e-15
        result[mask] = radial_integrand(mass * x[mask]) / (mass ** 3 + 1e-15)
        result[~mask] = 1.0
        return result if result.size > 1 else result.item()

    val_low = alpert_log_rule(f_low, n=16)
    val_low *= mass ** 4


    n_seg = 100
    k = np.linspace(mass, cutoff, n_seg)
    y = radial_integrand(k)
    val_high = np.trapezoid(y, k)


    total = (val_low + val_high) / (2.0 * np.pi ** 2)
    return total
