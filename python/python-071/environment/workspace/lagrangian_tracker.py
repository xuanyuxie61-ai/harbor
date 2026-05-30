# -*- coding: utf-8 -*-

import numpy as np
from utils import uniform_on_sphere_phong, brownian_displacement, direction_uniform_nd


def interpolate_velocity_3d(xp, yp, zp, u, v, w, x_grid, y_grid, z_grid):
    nx, ny, nz = u.shape
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]


    ix = np.clip(np.floor((xp - x_grid[0]) / dx).astype(int), 0, nx - 2)
    iy = np.clip(np.floor((yp - y_grid[0]) / dy).astype(int), 0, ny - 2)
    iz = np.clip(np.floor((zp - z_grid[0]) / dz).astype(int), 0, nz - 2)


    xi = (xp - x_grid[ix]) / dx
    eta = (yp - y_grid[iy]) / dy
    zeta = (zp - z_grid[iz]) / dz


    wx0 = 1.0 - xi
    wx1 = xi
    wy0 = 1.0 - eta
    wy1 = eta
    wz0 = 1.0 - zeta
    wz1 = zeta


    up = (
        u[ix, iy, iz] * wx0 * wy0 * wz0
        + u[ix + 1, iy, iz] * wx1 * wy0 * wz0
        + u[ix, iy + 1, iz] * wx0 * wy1 * wz0
        + u[ix + 1, iy + 1, iz] * wx1 * wy1 * wz0
        + u[ix, iy, iz + 1] * wx0 * wy0 * wz1
        + u[ix + 1, iy, iz + 1] * wx1 * wy0 * wz1
        + u[ix, iy + 1, iz + 1] * wx0 * wy1 * wz1
        + u[ix + 1, iy + 1, iz + 1] * wx1 * wy1 * wz1
    )

    vp = (
        v[ix, iy, iz] * wx0 * wy0 * wz0
        + v[ix + 1, iy, iz] * wx1 * wy0 * wz0
        + v[ix, iy + 1, iz] * wx0 * wy1 * wz0
        + v[ix + 1, iy + 1, iz] * wx1 * wy1 * wz0
        + v[ix, iy, iz + 1] * wx0 * wy0 * wz1
        + v[ix + 1, iy, iz + 1] * wx1 * wy0 * wz1
        + v[ix, iy + 1, iz + 1] * wx0 * wy1 * wz1
        + v[ix + 1, iy + 1, iz + 1] * wx1 * wy1 * wz1
    )

    wp = (
        w[ix, iy, iz] * wx0 * wy0 * wz0
        + w[ix + 1, iy, iz] * wx1 * wy0 * wz0
        + w[ix, iy + 1, iz] * wx0 * wy1 * wz0
        + w[ix + 1, iy + 1, iz] * wx1 * wy1 * wz0
        + w[ix, iy, iz + 1] * wx0 * wy0 * wz1
        + w[ix + 1, iy, iz + 1] * wx1 * wy0 * wz1
        + w[ix, iy + 1, iz + 1] * wx0 * wy1 * wz1
        + w[ix + 1, iy + 1, iz + 1] * wx1 * wy1 * wz1
    )

    return up, vp, wp


def lagrangian_particle_tracker(u, v, w, x_grid, y_grid, z_grid, n_particles=100,
                                 n_steps=200, dt=0.01, D_diff=0.01, seed=42):
    rng = np.random.default_rng(seed)


    sphere_points = uniform_on_sphere_phong(n_particles, rng=rng)

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    z_min, z_max = z_grid[0], z_grid[-1]

    x0 = 0.5 * (x_min + x_max) + 0.3 * (x_max - x_min) * sphere_points[:, 0]
    y0 = 0.5 * (y_min + y_max) + 0.3 * (y_max - y_min) * sphere_points[:, 1]
    z0 = 0.5 * (z_min + z_max) + 0.3 * (z_max - z_min) * sphere_points[:, 2]

    trajectories = np.zeros((n_particles, n_steps + 1, 3), dtype=float)
    trajectories[:, 0, 0] = x0
    trajectories[:, 0, 1] = y0
    trajectories[:, 0, 2] = z0

    for step in range(n_steps):
        xp = trajectories[:, step, 0]
        yp = trajectories[:, step, 1]
        zp = trajectories[:, step, 2]


        up = np.zeros(n_particles, dtype=float)
        vp = np.zeros(n_particles, dtype=float)
        wp = np.zeros(n_particles, dtype=float)

        for p in range(n_particles):
            try:
                up[p], vp[p], wp[p] = interpolate_velocity_3d(
                    xp[p], yp[p], zp[p], u, v, w, x_grid, y_grid, z_grid)
            except Exception:
                up[p], vp[p], wp[p] = 0.0, 0.0, 0.0


        dW = brownian_displacement(n_particles, dim=3, dt=dt, D=D_diff, rng=rng)


        trajectories[:, step + 1, 0] = xp + up * dt + dW[:, 0]
        trajectories[:, step + 1, 1] = yp + vp * dt + dW[:, 1]
        trajectories[:, step + 1, 2] = zp + wp * dt + dW[:, 2]


        for p in range(n_particles):
            for d, (g_min, g_max) in enumerate([(x_min, x_max), (y_min, y_max), (z_min, z_max)]):
                if trajectories[p, step + 1, d] < g_min:
                    trajectories[p, step + 1, d] = 2 * g_min - trajectories[p, step + 1, d]
                elif trajectories[p, step + 1, d] > g_max:
                    trajectories[p, step + 1, d] = 2 * g_max - trajectories[p, step + 1, d]


    msd = np.zeros(n_steps + 1, dtype=float)
    for step in range(n_steps + 1):
        dx = trajectories[:, step, 0] - x0
        dy = trajectories[:, step, 1] - y0
        dz = trajectories[:, step, 2] - z0
        msd[step] = np.mean(dx ** 2 + dy ** 2 + dz ** 2)

    return trajectories, msd


def turbulent_diffusion_coefficient(msd, dt):
    n = len(msd)
    t = np.arange(n) * dt
    t[0] = 1e-15

    D = msd / (6.0 * t)
    return D


def pair_separation_statistics(trajectories, dt):
    n_particles, n_steps, _ = trajectories.shape


    r2_mean = np.zeros(n_steps, dtype=float)
    count = 0

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = trajectories[i, :, 0] - trajectories[j, :, 0]
            dy = trajectories[i, :, 1] - trajectories[j, :, 1]
            dz = trajectories[i, :, 2] - trajectories[j, :, 2]
            r2_mean += dx ** 2 + dy ** 2 + dz ** 2
            count += 1

    if count > 0:
        r2_mean /= count

    return r2_mean
