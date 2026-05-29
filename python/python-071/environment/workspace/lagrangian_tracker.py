# -*- coding: utf-8 -*-
"""
lagrangian_tracker.py
拉格朗日粒子追踪与湍流扩散模块

融合来源:
- 1011_random_walk_3d_simulation: 三维随机游走
- 1006_random_data: 球面均匀采样、布朗运动、方向均匀分布

功能:
- 在湍流速度场中追踪拉格朗日粒子轨迹
- 计算粒子位移统计量（均方位移、扩散系数）
- 利用随机游走模型模拟湍流扩散
- 球面上均匀初始化粒子位置

数学背景:
  拉格朗日粒子轨迹方程:
    dX_p/dt = u(X_p, t)
    dY_p/dt = v(X_p, t)
    dZ_p/dt = w(X_p, t)

  对于湍流扩散，粒子均方位移满足:
    <(Delta x)^2> = 2 * D * t   (长时间尺度，爱因斯坦关系)
    <(Delta x)^2> ~ t^2         (短时间尺度，弹道区)
    <(Delta x)^2> ~ t^(2*H)     (分形湍流，H 为 Hurst 指数)

   Richardson 定律（湍流扩散）:
    d<(Delta r)^2>/dt ~ (Delta r)^(4/3)
    即扩散系数随尺度增大而增大。
"""

import numpy as np
from utils import uniform_on_sphere_phong, brownian_displacement, direction_uniform_nd


def interpolate_velocity_3d(xp, yp, zp, u, v, w, x_grid, y_grid, z_grid):
    """
    三线性插值获取粒子位置处的速度。

    数学公式:
      对网格单元 [i, i+1] x [j, j+1] x [k, k+1]，
      三线性插值为 8 个角点值的加权平均:
        u_p = sum_{a,b,c in {0,1}} u_{i+a,j+b,k+c} * w_x(a) * w_y(b) * w_z(c)
      其中 w_x(0) = (x_{i+1} - x_p) / dx, w_x(1) = (x_p - x_i) / dx。
    """
    nx, ny, nz = u.shape
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

    # 找到所在网格单元
    ix = np.clip(np.floor((xp - x_grid[0]) / dx).astype(int), 0, nx - 2)
    iy = np.clip(np.floor((yp - y_grid[0]) / dy).astype(int), 0, ny - 2)
    iz = np.clip(np.floor((zp - z_grid[0]) / dz).astype(int), 0, nz - 2)

    # 局部坐标
    xi = (xp - x_grid[ix]) / dx
    eta = (yp - y_grid[iy]) / dy
    zeta = (zp - z_grid[iz]) / dz

    # 插值权重
    wx0 = 1.0 - xi
    wx1 = xi
    wy0 = 1.0 - eta
    wy1 = eta
    wz0 = 1.0 - zeta
    wz1 = zeta

    # 三线性插值
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
    """
    在三维湍流场中追踪拉格朗日粒子。
    融合自 1011_random_walk_3d_simulation。

    参数:
      u, v, w: 三维速度场
      x_grid, y_grid, z_grid: 网格坐标
      n_particles: 粒子数
      n_steps: 时间步数
      dt: 时间步长
      D_diff: 扩散系数
      seed: 随机种子

    返回:
      trajectories: (n_particles, n_steps+1, 3) 粒子轨迹
      msd: (n_steps+1,) 均方位移
    """
    rng = np.random.default_rng(seed)

    # 在球面上均匀初始化粒子位置
    sphere_points = uniform_on_sphere_phong(n_particles, rng=rng)
    # 映射到计算域
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

        # 获取局部速度
        up = np.zeros(n_particles, dtype=float)
        vp = np.zeros(n_particles, dtype=float)
        wp = np.zeros(n_particles, dtype=float)

        for p in range(n_particles):
            try:
                up[p], vp[p], wp[p] = interpolate_velocity_3d(
                    xp[p], yp[p], zp[p], u, v, w, x_grid, y_grid, z_grid)
            except Exception:
                up[p], vp[p], wp[p] = 0.0, 0.0, 0.0

        # 布朗运动随机位移
        dW = brownian_displacement(n_particles, dim=3, dt=dt, D=D_diff, rng=rng)

        # 更新位置: x_{n+1} = x_n + u_p * dt + dW
        trajectories[:, step + 1, 0] = xp + up * dt + dW[:, 0]
        trajectories[:, step + 1, 1] = yp + vp * dt + dW[:, 1]
        trajectories[:, step + 1, 2] = zp + wp * dt + dW[:, 2]

        # 边界反射处理
        for p in range(n_particles):
            for d, (g_min, g_max) in enumerate([(x_min, x_max), (y_min, y_max), (z_min, z_max)]):
                if trajectories[p, step + 1, d] < g_min:
                    trajectories[p, step + 1, d] = 2 * g_min - trajectories[p, step + 1, d]
                elif trajectories[p, step + 1, d] > g_max:
                    trajectories[p, step + 1, d] = 2 * g_max - trajectories[p, step + 1, d]

    # 计算均方位移
    msd = np.zeros(n_steps + 1, dtype=float)
    for step in range(n_steps + 1):
        dx = trajectories[:, step, 0] - x0
        dy = trajectories[:, step, 1] - y0
        dz = trajectories[:, step, 2] - z0
        msd[step] = np.mean(dx ** 2 + dy ** 2 + dz ** 2)

    return trajectories, msd


def turbulent_diffusion_coefficient(msd, dt):
    """
    从均方位移估计湍流扩散系数。

    数学模型:
      爱因斯坦关系:
        D = MSD / (6 * t)   (三维)
      或
        D = lim_{t->inf} (1/6) * d(MSD)/dt

    参数:
      msd: 均方位移序列
      dt: 时间步长

    返回:
      D: 扩散系数序列
    """
    n = len(msd)
    t = np.arange(n) * dt
    t[0] = 1e-15  # 避免除零

    D = msd / (6.0 * t)
    return D


def pair_separation_statistics(trajectories, dt):
    """
    计算粒子对分离统计量（Richardson 扩散）。

    数学模型:
      粒子对分离距离 r(t) = |x_i(t) - x_j(t)|
      均方分离距离:
        <r^2(t)> = < |x_i(t) - x_j(t)|^2 >
      Richardson 定律:
        d<r^2>/dt ~ <r^2>^(2/3)
    """
    n_particles, n_steps, _ = trajectories.shape

    # 计算所有粒子对的分离
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
