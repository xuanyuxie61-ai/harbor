
import numpy as np


class InterpolationError(Exception):
    pass


def bracket_index(sorted_vec, xq):
    n = sorted_vec.size
    if n < 2:
        return -1
    if xq < sorted_vec[0] - 1e-14 or xq > sorted_vec[-1] + 1e-14:
        return -1

    lo, hi = 0, n - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_vec[mid] <= xq <= sorted_vec[mid + 1]:
            return mid
        elif xq < sorted_vec[mid]:
            hi = mid - 1
        else:
            lo = mid + 1

    if abs(xq - sorted_vec[0]) < 1e-12:
        return 0
    if abs(xq - sorted_vec[-1]) < 1e-12:
        return n - 2
    return -1


def pwl_interp_2d_scalar(xd, yd, zd, xi, yi):
    i = bracket_index(xd, xi)
    j = bracket_index(yd, yi)
    if i == -1 or j == -1:
        return np.inf










    dx = xd[i + 1] - xd[i]
    dy = yd[j + 1] - yd[j]
    if dx <= 0 or dy <= 0:
        raise InterpolationError("网格必须严格单调递增")

    y_diag = yd[j] + dy * (xi - xd[i]) / dx

    if yi < y_diag:

        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]
        det = dxa * dyb - dya * dxb
        if abs(det) < np.finfo(float).eps:
            return np.inf
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        zi = alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
    else:

        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]
        dxb = xd[i + 1] - xd[i + 1]
        dyb = yd[j] - yd[j + 1]
        dxi = xi - xd[i + 1]
        dyi = yi - yd[j + 1]
        det = dxa * dyb - dya * dxb
        if abs(det) < np.finfo(float).eps:
            return np.inf
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        zi = alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]

    return zi


def pwl_interp_2d(xd, yd, zd, xi, yi):
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)
    if xi.shape != yi.shape:
        raise InterpolationError("xi 与 yi 形状不一致")

    zi = np.empty_like(xi)
    it = np.nditer([xi, yi, zi], flags=['multi_index'])
    for xv, yv, zv in it:
        zv[...] = pwl_interp_2d_scalar(xd, yd, zd, xv, yv)
    return zi


def radial_to_2d_interpolator(r_nodes, values_r, n_theta=64, n_r=64):
    R = r_nodes[-1]

    r_samples = np.linspace(0.0, R, n_r)
    theta_samples = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)


    values_interp = np.interp(r_samples, r_nodes, values_r,
                              left=values_r[0], right=values_r[-1])


    R_grid, T_grid = np.meshgrid(r_samples, theta_samples)
    X = R_grid * np.cos(T_grid)
    Y = R_grid * np.sin(T_grid)
    Z = np.tile(values_interp.reshape(1, -1), (n_theta, 1))
    return X, Y, Z
