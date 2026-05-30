# -*- coding: utf-8 -*-

import numpy as np


def line_grid_1d(n, a, b, c=1):
    if n < 1:
        raise ValueError("网格点数 n 必须 >= 1。")
    if not (1 <= c <= 5):
        raise ValueError("居中策略 c 必须在 1 到 5 之间。")
    if a > b:
        a, b = b, a

    x = np.zeros(n, dtype=float)

    if c == 1:
        if n == 1:
            x[0] = 0.5 * (a + b)
        else:
            for j in range(n):
                x[j] = ((n - 1 - j) * a + j * b) / (n - 1)
    elif c == 2:
        for j in range(n):
            x[j] = ((n - j) * a + (j + 1) * b) / (n + 1)
    elif c == 3:
        for j in range(n):
            x[j] = ((n - j) * a + j * b) / n
    elif c == 4:
        for j in range(n):
            x[j] = ((n - 1 - j) * a + (j + 1) * b) / n
    elif c == 5:
        for j in range(n):
            x[j] = ((2 * n - 2 * j - 1) * a + (2 * j + 1) * b) / (2 * n)

    return x


def rect_grid_2d(nx, ny, x_bounds, y_bounds, cx=1, cy=1):
    x = line_grid_1d(nx, x_bounds[0], x_bounds[1], cx)
    y = line_grid_1d(ny, y_bounds[0], y_bounds[1], cy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = (x_bounds[1] - x_bounds[0]) / max(nx - 1, 1)
    dy = (y_bounds[1] - y_bounds[0]) / max(ny - 1, 1)
    return X, Y, dx, dy


def cylindrical_grid_2d(nr, nz, r_max, z_bounds):
    if nr < 2 or nz < 2:
        raise ValueError("nr 和 nz 必须 >= 2。")
    if r_max <= 0:
        raise ValueError("r_max 必须为正。")


    t = np.linspace(0.0, 1.0, nr)
    r = r_max * t**2

    z = line_grid_1d(nz, z_bounds[0], z_bounds[1], c=1)
    R, Z = np.meshgrid(r, z, indexing='ij')

    dr_min = r[1] - r[0] if nr > 1 else r_max
    dz = (z_bounds[1] - z_bounds[0]) / (nz - 1)

    return R, Z, dr_min, dz


def grid_spacing_quality(x):
    if len(x) < 2:
        return {'ratio': 1.0, 'max_dx': 0.0, 'min_dx': 0.0, 'mean_dx': 0.0}
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("网格点必须严格单调递增。")
    max_dx = float(np.max(dx))
    min_dx = float(np.min(dx))
    mean_dx = float(np.mean(dx))
    ratio = max_dx / min_dx if min_dx > 0 else np.inf
    return {
        'ratio': ratio,
        'max_dx': max_dx,
        'min_dx': min_dx,
        'mean_dx': mean_dx
    }


def cell_volumes_2d(X, Y):
    dx = np.diff(X[:, 0])
    dy = np.diff(Y[0, :])
    volumes = np.outer(dx, dy)
    return volumes
