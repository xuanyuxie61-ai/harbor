# -*- coding: utf-8 -*-

import numpy as np
from utils import safe_sqrt, safe_divide


def sample_annulus_uniform(n, center, r1, r2):
    if n < 1:
        return np.zeros((0, 2))
    if r1 < 0 or r2 < r1:
        raise ValueError("必须满足 0 <= r1 <= r2。")
    theta = 2.0 * np.pi * np.random.rand(n)
    v = np.random.rand(n)
    r = np.sqrt((1.0 - v) * r1 ** 2 + v * r2 ** 2)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def annulus_distance_stats(n, center, r1, r2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    p = sample_annulus_uniform(n, center, r1, r2)
    q = sample_annulus_uniform(n, center, r1, r2)
    d = np.linalg.norm(p - q, axis=1)
    mean = np.mean(d)
    var = np.var(d, ddof=1)
    return mean, var


def fermi_surface_annulus_stats(kx_grid, ky_grid, epsilon_grid, mu=0.0, delta_e=0.1, n_mc=5000):
    KX, KY = np.meshgrid(kx_grid, ky_grid)
    mask = np.abs(epsilon_grid - mu) < delta_e
    if not np.any(mask):
        return {
            'mean_distance': 0.0,
            'variance': 0.0,
            'centroid': np.array([0.0, 0.0]),
            'r1': 0.0,
            'r2': 0.0,
            'n_points': 0
        }
    pts = np.column_stack([KX[mask].ravel(), KY[mask].ravel()])
    centroid = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    r1 = np.min(dists)
    r2 = np.max(dists)
    mean_d, var_d = annulus_distance_stats(n_mc, centroid, r1, r2)
    return {
        'mean_distance': mean_d,
        'variance': var_d,
        'centroid': centroid,
        'r1': r1,
        'r2': r2,
        'n_points': pts.shape[0]
    }


def trace_fermi_surface_boundary(epsilon_grid, kx_grid, ky_grid, mu=0.0):
    eps = epsilon_grid - mu
    Ny, Nx = eps.shape


    boundary_pts = []
    dx = kx_grid[1] - kx_grid[0] if Nx > 1 else 1.0
    dy = ky_grid[1] - ky_grid[0] if Ny > 1 else 1.0


    for iy in range(Ny):
        for ix in range(Nx - 1):
            e1, e2 = eps[iy, ix], eps[iy, ix + 1]
            if e1 * e2 < 0:
                t = safe_divide(abs(e1), abs(e1) + abs(e2))
                xb = kx_grid[ix] + t * dx
                yb = ky_grid[iy]
                boundary_pts.append((xb, yb))

    for iy in range(Ny - 1):
        for ix in range(Nx):
            e1, e2 = eps[iy, ix], eps[iy + 1, ix]
            if e1 * e2 < 0:
                t = safe_divide(abs(e1), abs(e1) + abs(e2))
                xb = kx_grid[ix]
                yb = ky_grid[iy] + t * dy
                boundary_pts.append((xb, yb))

    if len(boundary_pts) < 3:
        return {
            'boundary_points': np.zeros((0, 2)),
            'area_approx': 0.0,
            'centroid_approx': np.array([0.0, 0.0]),
            'moment_approx': 0.0
        }

    pts = np.array(boundary_pts)

    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    pts = pts[order]


    n = pts.shape[0]
    area = 0.0
    cx_num = 0.0
    cy_num = 0.0
    I0 = 0.0
    for i in range(n):
        x_i, y_i = pts[i]
        x_next, y_next = pts[(i + 1) % n]
        cross = x_i * y_next - x_next * y_i
        area += cross
        cx_num += (x_i + x_next) * cross
        cy_num += (y_i + y_next) * cross
        I0 += (x_i ** 2 + x_i * x_next + x_next ** 2 +
               y_i ** 2 + y_i * y_next + y_next ** 2) * cross
    area = abs(area) * 0.5
    if area < 1e-15:
        centroid = c
        moment = 0.0
    else:
        centroid = np.array([cx_num / (6.0 * area), cy_num / (6.0 * area)])
        I0 /= 12.0
        moment = I0 - area * (centroid[0] ** 2 + centroid[1] ** 2)

    return {
        'boundary_points': pts,
        'area_approx': area,
        'centroid_approx': centroid,
        'moment_approx': moment
    }


def fermi_surface_nesting_vector(kx_grid, ky_grid, epsilon_grid, mu=0.0, delta_e=0.05):
    KX, KY = np.meshgrid(kx_grid, ky_grid)
    mask = np.abs(epsilon_grid - mu) < delta_e
    if not np.any(mask):
        return np.array([0.0, 0.0]), 0.0

    pts = np.column_stack([KX[mask].ravel(), KY[mask].ravel()])
    best_overlap = 0.0
    best_q = np.array([0.0, 0.0])


    for qx in kx_grid:
        for qy in ky_grid:
            shifted = pts + np.array([qx, qy])

            shifted[:, 0] = ((shifted[:, 0] + np.pi) % (2.0 * np.pi)) - np.pi
            shifted[:, 1] = ((shifted[:, 1] + np.pi) % (2.0 * np.pi)) - np.pi


            overlap = 0
            for p in shifted:
                ix = np.argmin(np.abs(kx_grid - p[0]))
                iy = np.argmin(np.abs(ky_grid - p[1]))
                if np.abs(epsilon_grid[iy, ix] - mu) < delta_e:
                    overlap += 1
            overlap = float(overlap) / pts.shape[0]
            if overlap > best_overlap:
                best_overlap = overlap
                best_q = np.array([qx, qy])

    return best_q, best_overlap
