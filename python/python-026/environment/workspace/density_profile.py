# -*- coding: utf-8 -*-

import numpy as np


def piecewise_constant_density_2d(xc, yc, nxc, nyc, density_func):
    if len(xc) != nxc + 1 or len(yc) != nyc + 1:
        raise ValueError("断点数组长度必须与单元数匹配。")
    if np.any(np.diff(xc) <= 0) or np.any(np.diff(yc) <= 0):
        raise ValueError("断点坐标必须严格递增。")

    ne_cells = np.zeros((nxc, nyc), dtype=float)
    for i in range(nxc):
        x_mid = 0.5 * (xc[i] + xc[i + 1])
        for j in range(nyc):
            y_mid = 0.5 * (yc[j] + yc[j + 1])
            val = density_func(x_mid, y_mid)
            if val < 0:
                val = 0.0
            ne_cells[i, j] = val

    return ne_cells, xc, yc


def icf_density_profile(x, y, n0, R0, Ls, f_plateau=0.3,
                         perturbation_amplitude=0.0, perturbation_scale=1e-6):
    if n0 < 0 or R0 <= 0 or Ls <= 0:
        raise ValueError("n0 必须非负，R0 和 Ls 必须为正。")

    r = np.sqrt(x**2 + y**2)

    ne = n0 * f_plateau + n0 * (1.0 - f_plateau) / (1.0 + np.exp((r - R0) / Ls))


    ne = np.clip(ne, 0.0, n0 * 1.1)


    if perturbation_amplitude > 0.0:
        if np.isscalar(x):
            rng = np.random.default_rng(seed=int((x + y) * 1e12) % 2**31)
            xi = rng.standard_normal()
        else:
            xi = np.random.default_rng(42).standard_normal(size=np.broadcast(x, y).shape)
        delta = perturbation_amplitude * n0 * np.exp(-r**2 / (2.0 * perturbation_scale**2)) * xi
        ne = ne + delta
        ne = np.clip(ne, 0.0, None)

    return ne


def density_gradient_pwc(ne_cells, xc, yc):
    nxc, nyc = ne_cells.shape
    if nxc < 2 or nyc < 2:
        raise ValueError("密度场至少需要 2x2 个单元以计算梯度。")

    dx = np.diff(xc)
    dy = np.diff(yc)

    grad_x = np.zeros((nxc - 1, nyc), dtype=float)
    for i in range(nxc - 1):
        dx_safe = max(dx[i], 1e-20)
        grad_x[i, :] = (ne_cells[i + 1, :] - ne_cells[i, :]) / dx_safe

    grad_y = np.zeros((nxc, nyc - 1), dtype=float)
    for j in range(nyc - 1):
        dy_safe = max(dy[j], 1e-20)
        grad_y[:, j] = (ne_cells[:, j + 1] - ne_cells[:, j]) / dy_safe

    return grad_x, grad_y


def integrate_density_along_path(ne_cells, xc, yc, path_points):
    if path_points.ndim != 2 or path_points.shape[1] != 2:
        raise ValueError("path_points 必须是形状为 (M, 2) 的数组。")
    M = path_points.shape[0]
    if M < 2:
        return 0.0, 0.0


    n_samples_per_segment = 50
    sampled_points = []
    for k in range(M - 1):
        p0 = path_points[k]
        p1 = path_points[k + 1]
        t = np.linspace(0.0, 1.0, n_samples_per_segment + 1)
        seg = np.outer(1.0 - t, p0) + np.outer(t, p1)
        if k > 0:
            seg = seg[1:]
        sampled_points.append(seg)
    sampled = np.vstack(sampled_points)


    ne_sampled = bilinear_interpolate_density(ne_cells, xc, yc, sampled[:, 0], sampled[:, 1])


    ds = np.sqrt(np.sum(np.diff(sampled, axis=0)**2, axis=1))
    integral = np.sum(0.5 * (ne_sampled[:-1] + ne_sampled[1:]) * ds)
    ds_total = np.sum(ds)

    return integral, ds_total


def bilinear_interpolate_density(ne_cells, xc, yc, xq, yq):
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    nxc, nyc = ne_cells.shape

    x_centers = 0.5 * (xc[:-1] + xc[1:])
    y_centers = 0.5 * (yc[:-1] + yc[1:])


    xq = np.clip(xq, x_centers[0], x_centers[-1])
    yq = np.clip(yq, y_centers[0], y_centers[-1])


    ix = np.searchsorted(x_centers, xq, side='right') - 1
    iy = np.searchsorted(y_centers, yq, side='right') - 1
    ix = np.clip(ix, 0, nxc - 2)
    iy = np.clip(iy, 0, nyc - 2)


    dx = x_centers[ix + 1] - x_centers[ix]
    dy = y_centers[iy + 1] - y_centers[iy]
    dx_safe = np.where(dx > 0, dx, 1.0)
    dy_safe = np.where(dy > 0, dy, 1.0)
    tx = (xq - x_centers[ix]) / dx_safe
    ty = (yq - y_centers[iy]) / dy_safe


    ne_q = (1.0 - tx) * (1.0 - ty) * ne_cells[ix, iy] + \
           tx * (1.0 - ty) * ne_cells[ix + 1, iy] + \
           (1.0 - tx) * ty * ne_cells[ix, iy + 1] + \
           tx * ty * ne_cells[ix + 1, iy + 1]

    return ne_q


def total_plasma_mass(ne_cells, xc, yc, ion_mass=2.5e-26):
    dx = np.diff(xc)
    dy = np.diff(yc)
    volumes = np.outer(dx, dy)
    mass = np.sum(ne_cells * volumes) * ion_mass
    return mass
