
import numpy as np
from utils import check_finite


def composite_trapezoidal(f, a, b, n):
    if n < 2:
        raise ValueError("composite_trapezoidal: n must be >= 2")
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    fx = f(x)
    check_finite(fx, "composite_trapezoidal fx")
    val = 0.5 * fx[0] + np.sum(fx[1:-1]) + 0.5 * fx[-1]
    return val * h


def gauss_legendre_nodes_weights(n):
    if n < 1:
        raise ValueError("gauss_legendre_nodes_weights: n must be >= 1")
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def gauss_legendre_integral(f, a, b, n):
    t, w = gauss_legendre_nodes_weights(n)
    x = 0.5 * (b + a) + 0.5 * (b - a) * t
    fx = f(x)
    check_finite(fx, "gauss_legendre_integral fx")
    return 0.5 * (b - a) * np.sum(w * fx)


def triangle_gauss_rule(order):
    if order == 1:
        xy = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        w = np.array([0.5])
    elif order == 3:
        xy = np.array([
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5]
        ])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 7:
        a = 1.0 / 3.0
        b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
        c = (6.0 - np.sqrt(15.0)) / 21.0
        d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
        e = (6.0 + np.sqrt(15.0)) / 21.0
        u = 0.225
        v = (155.0 - np.sqrt(15.0)) / 1200.0
        ww = (155.0 + np.sqrt(15.0)) / 1200.0
        xy = np.array([
            [a, a],
            [b, c], [c, b], [c, c],
            [d, e], [e, d], [e, e]
        ])
        w = 0.5 * np.array([u, v, v, v, ww, ww, ww])
    else:
        raise ValueError(f"triangle_gauss_rule: unsupported order {order}")
    return xy, w


def integrate_over_triangle(f, p1, p2, p3, order=7):
    from utils import compute_triangle_area
    area = compute_triangle_area(p1, p2, p3)
    if area < 1e-14:
        return 0.0
    xy_ref, w = triangle_gauss_rule(order)

    pts = (p1[None, :] +
           (p2 - p1)[None, :] * xy_ref[:, 0:1] +
           (p3 - p1)[None, :] * xy_ref[:, 1:2])
    vals = np.array([f(pt[0], pt[1]) for pt in pts])
    check_finite(vals, "integrate_over_triangle vals")
    return 2.0 * area * np.sum(w * vals)


def integrate_2d_grid(f, xlim, ylim, nx, ny, method='trapezoidal'):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    dx = (xlim[1] - xlim[0]) / (nx - 1)
    dy = (ylim[1] - ylim[0]) / (ny - 1)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    check_finite(Z, "integrate_2d_grid Z")
    if method == 'trapezoidal':

        W = np.ones((ny, nx))
        W[0, :] = 0.5
        W[-1, :] = 0.5
        W[:, 0] = 0.5
        W[:, -1] = 0.5
        W[0, 0] = 0.25
        W[0, -1] = 0.25
        W[-1, 0] = 0.25
        W[-1, -1] = 0.25
        return dx * dy * np.sum(W * Z)
    else:
        raise ValueError(f"integrate_2d_grid: unknown method {method}")
