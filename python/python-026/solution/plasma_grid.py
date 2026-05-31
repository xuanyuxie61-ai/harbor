# -*- coding: utf-8 -*-
"""
plasma_grid.py

基于 line_grid 核心算法扩展的空间网格生成模块。

原项目 680_line_grid 提供了 1D 线段上的多种居中网格生成策略。
在激光-等离子体相互作用中，这些策略被扩展用于构建 1D/2D/3D 空间离散网格，
以支持等离子体密度场、电磁场以及射线追踪轨迹的数值表示。

核心算法保留:
    - 均匀网格 (centering=1): x_j = ((n-j)*a + (j-1)*b)/(n-1)
    - 内点网格 (centering=2): x_j = ((n-j+1)*a + j*b)/(n+1)
    - 左闭网格 (centering=3): x_j = ((n-j+1)*a + (j-1)*b)/n
    - 右闭网格 (centering=4): x_j = ((n-j)*a + j*b)/n
    - 中点网格 (centering=5): x_j = ((2n-2j+1)*a + (2j-1)*b)/(2n)
"""

import numpy as np


def line_grid_1d(n, a, b, c=1):
    """
    在一维区间 [a, b] 上生成 n 个网格点。

    Parameters
    ----------
    n : int
        网格点数，必须 >= 1。
    a : float
        区间左端点。
    b : float
        区间右端点。
    c : int, optional
        居中策略，1 <= c <= 5，默认为 1 (均匀)。

    Returns
    -------
    x : ndarray, shape (n,)
        网格点坐标。
    """
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
    """
    在二维矩形区域上生成结构化网格。

    Parameters
    ----------
    nx, ny : int
        x 和 y 方向的网格点数。
    x_bounds : tuple (xmin, xmax)
        x 方向边界。
    y_bounds : tuple (ymin, ymax)
        y 方向边界。
    cx, cy : int, optional
        x 和 y 方向的居中策略。

    Returns
    -------
    X, Y : ndarray
        网格坐标矩阵 (nx, ny)。
    dx, dy : float
        最小网格间距。
    """
    x = line_grid_1d(nx, x_bounds[0], x_bounds[1], cx)
    y = line_grid_1d(ny, y_bounds[0], y_bounds[1], cy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx = (x_bounds[1] - x_bounds[0]) / max(nx - 1, 1)
    dy = (y_bounds[1] - y_bounds[0]) / max(ny - 1, 1)
    return X, Y, dx, dy


def cylindrical_grid_2d(nr, nz, r_max, z_bounds):
    """
    生成柱坐标系下的 2D (r, z) 结构化网格。
    在 r 方向采用非均匀加密（靠近轴线更密），z 方向均匀。

    公式: r_j = r_max * (j / (nr-1))^2, j = 0, ..., nr-1

    Parameters
    ----------
    nr, nz : int
        r 和 z 方向的网格点数。
    r_max : float
        最大半径 [m]。
    z_bounds : tuple (zmin, zmax)
        z 方向边界 [m]。

    Returns
    -------
    R, Z : ndarray
        柱坐标网格矩阵 (nr, nz)。
    dr_min, dz : float
        最小径向和轴向间距。
    """
    if nr < 2 or nz < 2:
        raise ValueError("nr 和 nz 必须 >= 2。")
    if r_max <= 0:
        raise ValueError("r_max 必须为正。")

    # r 方向使用二次映射进行加密
    t = np.linspace(0.0, 1.0, nr)
    r = r_max * t**2

    z = line_grid_1d(nz, z_bounds[0], z_bounds[1], c=1)
    R, Z = np.meshgrid(r, z, indexing='ij')

    dr_min = r[1] - r[0] if nr > 1 else r_max
    dz = (z_bounds[1] - z_bounds[0]) / (nz - 1)

    return R, Z, dr_min, dz


def grid_spacing_quality(x):
    """
    评估一维网格的间距质量。

    计算最大/最小间距比和间距变化率，用于判断网格的数值稳定性。

    Parameters
    ----------
    x : ndarray
        已排序的网格点坐标。

    Returns
    -------
    quality : dict
        包含 'ratio', 'max_dx', 'min_dx', 'mean_dx' 的字典。
    """
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
    """
    计算二维结构化网格上每个控制体的体积（面积）。

    使用梯形法则近似:
        V_{ij} ≈ (x_{i+1} - x_i) * (y_{j+1} - y_j)

    Parameters
    ----------
    X, Y : ndarray
        网格坐标矩阵。

    Returns
    -------
    volumes : ndarray, shape (nx-1, ny-1)
        每个控制体的面积。
    """
    dx = np.diff(X[:, 0])
    dy = np.diff(Y[0, :])
    volumes = np.outer(dx, dy)
    return volumes
