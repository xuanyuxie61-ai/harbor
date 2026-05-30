
import numpy as np
from utils import binomial_coeff


def vandermonde_quadrature_weights(n, a, b, x):
    x = np.array(x, dtype=float).flatten()
    if x.size != n:
        raise ValueError("x must have length n")
    v = np.zeros((n, n), dtype=float)
    v[0, :] = 1.0
    for i in range(1, n):
        v[i, :] = v[i - 1, :] * x
    rhs = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        rhs[i - 1] = (b ** i - a ** i) / i

    w = np.linalg.solve(v, rhs)
    return w


def pyramid_monomial_integral(expon):
    e0, e1, e2 = int(expon[0]), int(expon[1]), int(expon[2])
    value = 0.0
    if (e0 % 2 == 0) and (e1 % 2 == 0):
        i_hi = 2 + e0 + e1
        s = 0.0
        for i in range(i_hi + 1):
            s += ((-1) ** i) * binomial_coeff(i_hi, i) / (i + e2 + 1)
        value = s * (2.0 / (e0 + 1)) * (2.0 / (e1 + 1))
    return float(value)


def pyramid_volume():
    return 4.0 / 3.0


def composite_quadrature_2d(func, xl, xr, yb, yt, nx, ny):
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")
    hx = (xr - xl) / nx
    hy = (yt - yb) / ny

    gl_nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gl_weights = np.array([1.0, 1.0])
    total = 0.0
    for i in range(nx):
        x0 = xl + i * hx
        for j in range(ny):
            y0 = yb + j * hy

            for xi, wi in zip(gl_nodes, gl_weights):
                x = x0 + hx * 0.5 * (xi + 1.0)
                for eta, wj in zip(gl_nodes, gl_weights):
                    y = y0 + hy * 0.5 * (eta + 1.0)
                    total += wi * wj * func(x, y) * hx * 0.5 * hy * 0.5
    return float(total)


def estimate_quadrature_error(func, xl, xr, yb, yt, n1, n2):
    i1 = composite_quadrature_2d(func, xl, xr, yb, yt, n1, n1)
    i2 = composite_quadrature_2d(func, xl, xr, yb, yt, n2, n2)
    p = 4.0
    if n2 == n1:
        return 0.0
    factor = 1.0 - (n1 / n2) ** p
    err_est = abs(i2 - i1) / abs(factor)
    return float(err_est)
