
import numpy as np
from math import sqrt, pi


class MieScatteringError(Exception):
    pass


def legendre_polynomial_value(m, n, x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] != m:
        raise MieScatteringError("legendre_polynomial_value: x 维度错误")
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise MieScatteringError("legendre_polynomial_value: x 超出 [-1,1]")

    v = np.zeros((m, n + 1), dtype=np.float64)
    v[:, 0] = 1.0

    for j in range(1, n + 1):
        if j == 1:
            vjm1 = np.zeros(m, dtype=np.float64)
        else:
            vjm1 = v[:, j - 2]
        v[:, j] = ((2.0 * j - 1.0) * x * v[:, j - 1] - (j - 1.0) * vjm1) / j

    return v


def legendre_coefficients_hg(g, max_l):
    if not (-1.0 < g < 1.0):
        raise MieScatteringError("legendre_coefficients_hg: g 必须在 (-1,1)")

    l = np.arange(0, max_l + 1, dtype=np.float64)
    return g ** l


def phase_function_hg(cos_theta, g):
    if not (-1.0 < g < 1.0):
        raise MieScatteringError("phase_function_hg: g 必须在 (-1,1)")
    denom = (1.0 + g ** 2 - 2.0 * g * cos_theta) ** 1.5
    if np.any(denom < 1e-15):
        raise MieScatteringError("phase_function_hg: 分母过小")
    return (1.0 - g ** 2) / denom


def scattering_asymmetry_parameter(g_eff, num_points=200):
    mu = np.linspace(-1.0, 1.0, num_points)
    p = phase_function_hg(mu, g_eff)
    integrand = p * mu
    return 0.5 * np.trapezoid(integrand, mu)


def expand_phase_function_legendre(cos_theta, coeffs):
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    L = len(coeffs) - 1
    m = cos_theta.shape[0] if cos_theta.ndim > 0 else 1

    if m == 1 and np.isscalar(cos_theta):
        cos_theta = np.array([cos_theta])
        m = 1

    v = legendre_polynomial_value(m, L, cos_theta)
    l_idx = np.arange(0, L + 1)
    prefactor = (2.0 * l_idx + 1.0) / (4.0 * pi)

    phase = np.sum(prefactor * coeffs * v, axis=1)
    return phase


def mie_scattering_cross_section(r, wavelength, m_eff, num_terms=None):
    x = 2.0 * pi * r / wavelength
    if x <= 0:
        raise MieScatteringError("mie_scattering_cross_section: x 必须为正")

    n_r = np.real(m_eff)
    n_i = np.imag(m_eff)

    if x < 0.1:

        ratio = (m_eff ** 2 - 1.0) / (m_eff ** 2 + 2.0)
        q_sca = (8.0 / 3.0) * (x ** 4) * (abs(ratio) ** 2)
        q_abs = 4.0 * x * np.imag(ratio)
        q_ext = q_sca + q_abs
        g = 0.0
    elif x > 50.0:

        q_ext = 2.0
        q_sca = 2.0 * (1.0 + np.exp(-4.0 * x * n_i)) / (1.0 + np.exp(-4.0 * x * n_i))

        g = 0.7
    else:

        if abs(n_r - 1.0) < 1e-6:
            q_ext = 2.0
            q_sca = 2.0
            g = 0.5
        else:
            rho = 2.0 * x * (n_r - 1.0)
            if abs(n_r - 1.0) < 1e-12:
                beta = pi / 2.0
            else:
                beta = np.arctan2(n_i, n_r - 1.0)
            tan_b = np.tan(beta)
            if abs(tan_b) < 1e-12:
                exp_term = np.exp(-rho * 1e12)
            else:
                exp_term = np.exp(-rho * tan_b)
            term1 = 4.0 * exp_term * np.cos(beta) / rho * np.sin(rho - beta)
            term2 = 4.0 * exp_term * (np.cos(beta) / rho) ** 2 * np.cos(rho - 2.0 * beta)
            q_ext = 2.0 - term1 - term2
            q_ext = float(np.real(q_ext))
            if q_ext < 0 or q_ext > 4.0:
                q_ext = 2.0
            q_sca = q_ext * 0.9
            g = 0.6 + 0.2 * (n_r - 1.0)
            g = np.clip(g, -0.9, 0.9)

    area = pi * r ** 2
    c_ext = q_ext * area
    c_sca = q_sca * area
    return float(c_ext), float(c_sca), float(g)
