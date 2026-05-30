
import numpy as np


def doughnut_exact_solution(t, m=3.0, n=5.0, a=1.0, b=1.0, c=3.0):
    t = np.atleast_1d(t)
    delta = 1.0 + a ** 2 + b ** 2 + c ** 2

    denom = delta - 2.0 * c * np.sin(n * t) + (2.0 - delta) * np.cos(n * t)
    denom = np.where(np.abs(denom) < 1.0e-12, 1.0e-12, denom)

    y1 = (2.0 * a * np.cos(m * t) - 2.0 * b * np.sin(m * t)) / denom
    y2 = (2.0 * a * np.sin(m * t) + 2.0 * b * np.cos(m * t)) / denom
    y3 = (2.0 * c * np.cos(n * t) + (2.0 - delta) * np.sin(n * t)) / denom

    return np.column_stack([y1, y2, y3])


def manufactured_solution_temperature(Z, T_ox=300.0, T_ad=2226.0):
    Z = np.clip(Z, 0.0, 1.0)
    T_exact = T_ox + (T_ad - T_ox) * np.sin(np.pi * Z / 2.0)
    d2T_dZ2 = -(np.pi / 2.0) ** 2 * (T_ad - T_ox) * np.sin(np.pi * Z / 2.0)
    return T_exact, d2T_dZ2


def gaussian_flamelet_solution(Z, Z_st, T_ox, T_ad, chi_st, omega_max):
    Z = np.clip(Z, 0.0, 1.0)
    sigma_sq = chi_st / (2.0 * max(omega_max, 1.0e-12))
    sigma = np.sqrt(sigma_sq)

    exponent = -((Z - Z_st) ** 2) / (2.0 * sigma_sq)
    exponent = np.clip(exponent, -700.0, 0.0)

    T_approx = T_ox + (T_ad - T_ox) * np.exp(exponent)
    return T_approx, sigma


def compute_errors(numerical, exact, Z_nodes):
    e = numerical - exact
    dZ = np.diff(Z_nodes)


    e_sq = e ** 2
    L2_sq = np.trapezoid(e_sq, Z_nodes)
    L2_error = np.sqrt(L2_sq)


    Linf_error = np.max(np.abs(e))


    de_dZ = np.zeros_like(e)
    de_dZ[1:-1] = (e[2:] - e[:-2]) / (Z_nodes[2:] - Z_nodes[:-2])
    de_dZ[0] = (e[1] - e[0]) / (Z_nodes[1] - Z_nodes[0])
    de_dZ[-1] = (e[-1] - e[-2]) / (Z_nodes[-1] - Z_nodes[-2])

    H1_semi_sq = np.trapezoid(de_dZ ** 2, Z_nodes)
    H1_semi_error = np.sqrt(H1_semi_sq)

    errors = {
        'L2_error': L2_error,
        'Linf_error': Linf_error,
        'H1_semi_error': H1_semi_error,
        'relative_L2': L2_error / (np.sqrt(np.trapezoid(exact ** 2, Z_nodes)) + 1.0e-12),
    }

    return errors
