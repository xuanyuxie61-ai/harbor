
import numpy as np
from scipy.special import gamma as scipy_gamma
from scipy.special import roots_genlaguerre, roots_legendre, roots_jacobi


class QuadratureError(Exception):
    pass


def gauss_legendre_rule(n, a=-1.0, b=1.0):
    if n < 1:
        raise QuadratureError("n 必须 ≥ 1")
    x, w = roots_legendre(n)

    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    x = scale * x + shift
    w = scale * w
    return x, w


def gauss_genlaguerre_rule(n, alpha, a=0.0, b=1.0):
    if n < 1:
        raise QuadratureError("n 必须 ≥ 1")
    if alpha <= -1.0:
        raise QuadratureError("alpha 必须 > -1")
    if b <= 0.0:
        raise QuadratureError("b 必须 > 0")


    t, wts = roots_genlaguerre(n, alpha)

    x = a + t / b
    w = wts / (b ** (alpha + 1.0))
    return x, w


def radial_quadrature_sphere(n, R):
    if R <= 0:
        raise QuadratureError("R 必须为正")

    t, wt = roots_legendre(n)
    r_nodes = R * (t + 1.0) / 2.0
    jacobian = R / 2.0

    r_weights = 4.0 * np.pi * (r_nodes ** 2) * wt * jacobian
    return r_nodes, r_weights


def gegenbauer_quadrature_exactness(alpha, n_points, degree_max):
    from special_functions import gegenbauer_integral




    x, w = roots_jacobi(n_points, alpha, alpha)






    moment0 = np.sum(w)

    target_moment0 = (2.0 ** (2.0 * alpha + 1.0)) * (scipy_gamma(alpha + 1.0) ** 2) \
                     / scipy_gamma(2.0 * alpha + 2.0)
    scale = target_moment0 / moment0
    w = w * scale

    errors = {}
    for p in range(degree_max + 1):
        exact = gegenbauer_integral(p, alpha)
        quad_val = np.sum(w * (x ** p))
        if abs(exact) < np.finfo(float).eps:
            err = abs(quad_val - exact)
        else:
            err = abs(quad_val - exact) / abs(exact)
        errors[p] = err
    return errors


def integrate_reaction_rate_radial(reaction_rate_func, R, n_quad=16):
    r_nodes, r_weights = radial_quadrature_sphere(n_quad, R)
    rates = np.array([reaction_rate_func(r) for r in r_nodes])
    total_rate = np.sum(rates * r_weights)
    return total_rate


def pore_size_moment_quadrature(pore_diameters, weights, moment_order,
                                alpha_param=0.5):
    pore_diameters = np.asarray(pore_diameters, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if pore_diameters.size == 0:
        raise QuadratureError("孔径分布为空")

    wsum = np.sum(weights)
    if wsum == 0:
        raise QuadratureError("权重之和为零")
    weights = weights / wsum


    d_clip = np.clip(pore_diameters, -1.0 + 1e-12, 1.0 - 1e-12)
    weighted = (d_clip ** moment_order) * ((1.0 - d_clip ** 2) ** alpha_param)
    moment = np.sum(weights * weighted)
    return moment
