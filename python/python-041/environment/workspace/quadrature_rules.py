
import numpy as np
from scipy.special import gamma, factorial, factorial2, hyp2f1


def monomial_integral_legendre(expon):
    if expon < 0:
        return -np.inf
    if expon % 2 == 1:
        return 0.0
    return 2.0 / (expon + 1)


def monomial_integral_jacobi(expon, alpha, beta):
    if alpha <= -1.0 or beta <= -1.0:
        return -np.inf
    s = 1.0 if (expon % 2 == 0) else -1.0
    arg1 = -alpha
    arg2 = 1.0 + expon
    arg3 = 2.0 + beta + expon
    arg4 = -1.0
    value1 = hyp2f1(arg1, arg2, arg3, arg4)
    value2 = hyp2f1(-beta, arg2, 2.0 + alpha + expon, arg4)
    value = (
        gamma(1.0 + expon) * (
            s * gamma(1.0 + beta) * value1 / gamma(2.0 + beta + expon)
            + gamma(1.0 + alpha) * value2 / gamma(2.0 + alpha + expon)
        )
    )
    return value


def monomial_integral_laguerre(expon):
    if expon < 0:
        return -np.inf
    return float(factorial(expon))


def monomial_integral_generalized_laguerre(expon, alpha):
    if alpha <= -1.0:
        return -np.inf
    arg = alpha + expon + 1.0
    return gamma(arg)


def monomial_integral_hermite(expon):
    if expon < 0:
        return -np.inf
    if expon % 2 == 1:
        return 0.0
    return float(factorial2(expon - 1)) * np.sqrt(np.pi) / (2.0 ** (expon / 2))


def monomial_integral_generalized_hermite(expon, alpha):
    if alpha <= -1.0:
        return -np.inf
    if expon % 2 == 1:
        return 0.0
    arg = (alpha + expon) / 2.0
    if arg <= -1.0:
        return -np.inf
    return gamma((alpha + expon + 1.0) / 2.0)


def monomial_integral_mixed(dim_num, rule, alpha, beta, expon):
    value = 1.0
    for dim in range(dim_num):
        r = int(rule[dim])
        if r == 1:
            value *= monomial_integral_legendre(int(expon[dim]))
        elif r == 2:
            value *= monomial_integral_jacobi(int(expon[dim]), alpha[dim], beta[dim])
        elif r == 3:
            value *= monomial_integral_laguerre(int(expon[dim]))
        elif r == 4:
            value *= monomial_integral_generalized_laguerre(int(expon[dim]), alpha[dim])
        elif r == 5:
            value *= monomial_integral_hermite(int(expon[dim]))
        elif r == 6:
            value *= monomial_integral_generalized_hermite(int(expon[dim]), alpha[dim])
        else:
            raise ValueError(f"Unknown rule type: {r}")
    return value


def gauss_lobatto_legendre_points_weights(n):
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 2:
        points = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
        return points, weights

    from numpy.polynomial.legendre import legder, legroots

    coeffs = np.zeros(n)
    coeffs[-1] = 1.0
    dp_coeffs = legder(coeffs)
    inner_pts = np.sort(legroots(dp_coeffs))
    points = np.concatenate([[-1.0], inner_pts, [1.0]])

    from scipy.special import eval_legendre
    weights = np.zeros(n)
    for i in range(n):
        xi = points[i]
        p_n1 = eval_legendre(n - 1, xi)

        if abs(p_n1) < 1e-15:
            weights[i] = 2.0 / (n * (n - 1))
        else:
            weights[i] = 2.0 / (n * (n - 1) * p_n1 ** 2)

    weights = weights / np.sum(weights) * 2.0
    return points, weights


def test_quadrature_exactness(max_degree=6):
    dim_num = 2
    rule = np.array([1, 1])
    alpha = np.zeros(dim_num)
    beta = np.zeros(dim_num)
    errors = []
    for degree in range(max_degree + 1):
        max_err = 0.0

        n_quad = max(2, int(np.ceil((degree + 3) / 2.0)) + 1)
        pts, wts = gauss_lobatto_legendre_points_weights(n_quad)
        for i in range(degree + 1):
            expon = np.array([i, degree - i])
            exact = monomial_integral_mixed(dim_num, rule, alpha, beta, expon)

            approx = 0.0
            for ix in range(n_quad):
                for iy in range(n_quad):
                    fval = (pts[ix] ** expon[0]) * (pts[iy] ** expon[1])
                    approx += wts[ix] * wts[iy] * fval
            err = abs(approx - exact)
            if err > max_err:
                max_err = err
        errors.append(max_err)
    return errors
