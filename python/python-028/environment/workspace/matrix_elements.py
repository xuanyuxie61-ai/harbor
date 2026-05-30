
import numpy as np
from math import sqrt, pi, cos, sin


def gauss_legendre_nodes_weights(n):
    if n <= 0:
        raise ValueError("n 必须为正整数")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def cauchy_principal_value(f, a, b, x0, n=64):
    if n % 2 != 0:
        n += 1

    if not (a < x0 < b):
        nodes, weights = gauss_legendre_nodes_weights(n)
        x_mapped = 0.5 * ((b - a) * nodes + a + b)
        w_mapped = 0.5 * (b - a) * weights
        return np.sum(w_mapped * f(x_mapped) / (x_mapped - x0))

    delta = min(x0 - a, b - x0) * 0.5
    delta = max(delta, 1e-10)

    nodes, weights = gauss_legendre_nodes_weights(n)

    if x0 - delta > a:
        x_left = 0.5 * ((x0 - delta - a) * nodes + x0 - delta + a)
        w_left = 0.5 * (x0 - delta - a) * weights
        I_left = np.sum(w_left * f(x_left) / (x_left - x0))
    else:
        I_left = 0.0

    if b > x0 + delta:
        x_right = 0.5 * ((b - x0 - delta) * nodes + b + x0 + delta)
        w_right = 0.5 * (b - x0 - delta) * weights
        I_right = np.sum(w_right * f(x_right) / (x_right - x0))
    else:
        I_right = 0.0

    I_center = 0.0
    for i in range(n):
        s = nodes[i]
        if abs(s) < 1e-15:
            continue
        x_s = x0 + delta * s
        I_center += weights[i] * f(x_s) / s

    return I_left + I_center + I_right


def self_energy_integral(coupling_squared, energy_levels, E, n_quad=64):
    sigma = 0.0
    gamma_width = 0.5

    for k in range(len(energy_levels)):
        E_k = energy_levels[k]
        V2 = coupling_squared[k]

        def integrand(Ep):
            lorentz = (1.0 / pi) * gamma_width / ((Ep - E_k) ** 2 + gamma_width ** 2)
            return V2 * lorentz

        E_min = E_k - 10.0 * gamma_width
        E_max = E_k + 10.0 * gamma_width

        try:
            contrib = cauchy_principal_value(integrand, E_min, E_max, E, n_quad)
        except Exception:
            nodes, weights = gauss_legendre_nodes_weights(n_quad)
            x_mapped = 0.5 * ((E_max - E_min) * nodes + E_max + E_min)
            w_mapped = 0.5 * (E_max - E_min) * weights
            lorentz = (1.0 / pi) * gamma_width / ((x_mapped - E_k) ** 2 + gamma_width ** 2)
            contrib = np.sum(w_mapped * V2 * lorentz / (E - x_mapped))

        sigma += contrib

    return sigma


def electric_multipole_matrix_element(r_grid, u_i, u_f, lambda_order):
    integrand = u_f * (r_grid ** lambda_order) * u_i
    return np.trapezoid(integrand, r_grid)


def transition_probability(lambda_order, me, E_gamma, mass_number, Ji):
    B_lambda = me ** 2 / (2.0 * Ji + 1.0)

    R = 1.2 * (mass_number ** (1.0 / 3.0))
    B_W = (1.0 / (4.0 * pi)) * (3.0 / (lambda_order + 3.0)) ** 2 * R ** (2 * lambda_order)

    if E_gamma > 1e-6:
        tau_half = 1e-16 / (E_gamma ** (2 * lambda_order + 1) * B_lambda)
    else:
        tau_half = 1e10

    return B_lambda, B_W, tau_half


def overlap_integral(r_grid, u1, u2):
    return np.trapezoid(u1 * u2, r_grid)


def spectroscopic_factor(r_grid, u_orbital, u_residual, A_core, n, l, j):
    overlap = overlap_integral(r_grid, u_orbital, u_residual)
    return overlap ** 2 * (2.0 * j + 1.0)
