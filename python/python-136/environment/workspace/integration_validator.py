
import numpy as np
from scipy.special import roots_legendre


class IntegrationValidatorError(Exception):
    pass


def legendre_2d_monomial_integral(a, b, p):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=int)

    val = 1.0
    for dim in range(2):
        if p[dim] < 0:
            raise IntegrationValidatorError("指数必须非负")
        exp = p[dim]
        if exp == 0:
            val *= (b[dim] - a[dim])
        else:
            val *= (b[dim] ** (exp + 1) - a[dim] ** (exp + 1)) / (exp + 1)
    return val


def validate_2d_quadrature_rule(n_points, degree_max, a=None, b=None):
    if a is None:
        a = np.array([-1.0, -1.0])
    if b is None:
        b = np.array([1.0, 1.0])

    x1d, w1d = roots_legendre(n_points)

    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    x_nodes = scale[0] * x1d + shift[0]
    y_nodes = scale[1] * x1d + shift[1]
    w_x = scale[0] * w1d
    w_y = scale[1] * w1d

    max_error = 0.0
    error_dict = {}

    for total_degree in range(degree_max + 1):
        max_err_td = 0.0
        for py in range(total_degree + 1):
            px = total_degree - py
            exact = legendre_2d_monomial_integral(a, b, [px, py])


            X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
            W = np.outer(w_x, w_y)
            vals = (X ** px) * (Y ** py)
            quad = np.sum(vals * W)

            if abs(exact) < np.finfo(float).eps:
                err = abs(quad - exact)
            else:
                err = abs(quad - exact) / abs(exact)
            max_err_td = max(max_err_td, err)

        error_dict[total_degree] = max_err_td
        max_error = max(max_error, max_err_td)

    return max_error, error_dict


def diffusion_green_function_integral(r, t, D, R):
    if t <= 0 or D <= 0 or R <= 0:
        raise IntegrationValidatorError("t, D, R 必须为正")

    r = np.asarray(r, dtype=float)
    G = (1.0 / (4.0 * np.pi * D * t) ** 1.5) * np.exp(-r ** 2 / (4.0 * D * t))
    integrand = G * 4.0 * np.pi * r ** 2


    integral_value = np.trapezoid(integrand, r)


    from scipy.special import erf
    exact_full = 1.0

    truncation = erf(R / np.sqrt(4.0 * D * t))
    exact_truncated = truncation
    return integral_value, exact_full, exact_truncated


def black_scholes_diffusion_analogy(S, K, T, r, sigma):
    from scipy.stats import norm
    if T <= 0:
        return max(S - K, 0.0), 0.0, 0.0
    if sigma <= 0:
        raise IntegrationValidatorError("sigma 必须为正")

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    call_price = S * N1 - K * np.exp(-r * T) * N2
    return call_price, d1, d2


def validate_reaction_diffusion_conservation(C, R, r_nodes, reaction_rates,
                                              D_eff=1e-6):
    if r_nodes.size != C.size or r_nodes.size != reaction_rates.size:
        raise IntegrationValidatorError("数组长度不一致")


    dr_last = r_nodes[-1] - r_nodes[-2]
    dCdr = (C[-1] - C[-2]) / dr_last
    flux_surface = 4.0 * np.pi * R ** 2 * D_eff * dCdr


    total_reaction = np.trapezoid(reaction_rates * 4.0 * np.pi * r_nodes ** 2,
                                   r_nodes)

    denom = max(abs(flux_surface), abs(total_reaction), np.finfo(float).eps)
    relative_error = abs(flux_surface - total_reaction) / denom
    return flux_surface, total_reaction, relative_error
