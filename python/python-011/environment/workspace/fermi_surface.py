# -*- coding: utf-8 -*-
"""
fermi_surface.py
----------------
费米面几何分析与关联函数计算模块。

对应种子项目：
  - 007_annulus_distance：环形区域内的随机采样与距离统计
  - 110_boundary_word_square：边界词多连方几何（面积、质心、转动惯量）

物理背景：
  高温超导体（如铜氧化物）在欠掺杂区具有 hole-like 费米面，
  其拓扑在 van Hove 填充附近呈现近嵌套 (nesting) 特征。
  费米面的形状、曲率及电子关联在动量空间中的距离统计
  直接影响磁通量子化和配对对称性。

核心公式：
  - 费米面定义：ε(k) = 0
  - 态密度：N(0) = (1/N) Σ_k δ(ε(k))
  - 环形关联：g(r) = ∫_FS dk dk' δ(|k-k'| - q)
"""

import numpy as np
from utils import safe_sqrt, safe_divide


def sample_annulus_uniform(n, center, r1, r2):
    """
    在二维环形区域 r1 <= |r - center| <= r2 内均匀随机采样 n 个点。

    算法（Shirley, Graphics Gems III）：
        θ ~ U(0, 2π)
        v ~ U(0,1)
        r = sqrt((1-v) r1^2 + v r2^2)   [保证面积均匀]
    """
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
    """
    计算环形区域内两独立随机点之间距离的 Monte Carlo 均值与方差。

    返回 mean, variance。
    """
    if seed is not None:
        np.random.seed(seed)
    p = sample_annulus_uniform(n, center, r1, r2)
    q = sample_annulus_uniform(n, center, r1, r2)
    d = np.linalg.norm(p - q, axis=1)
    mean = np.mean(d)
    var = np.var(d, ddof=1)
    return mean, var


def fermi_surface_annulus_stats(kx_grid, ky_grid, epsilon_grid, mu=0.0, delta_e=0.1, n_mc=5000):
    """
    从离散的色散关系网格中提取费米面附近的环形区域，并计算距离统计。

    步骤：
      1. 找到 |ε(k)-μ| < delta_e 的 k 点构成“费米环”。
      2. 计算该环的质心、等效内外半径。
      3. 用 annulus_distance_stats 计算环上动量关联距离统计。

    Parameters
    ----------
    kx_grid, ky_grid : ndarray, shape (Nk,)
        k 点网格坐标。
    epsilon_grid : ndarray, shape (Nk, Nk)
        色散能量网格。
    mu : float
        化学势（费米能级）。
    delta_e : float
        能量窗口宽度。
    n_mc : int
        Monte Carlo 采样数。

    Returns
    -------
    stats : dict
        包含 'mean_distance', 'variance', 'centroid', 'r1', 'r2'。
    """
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
    """
    用边界追踪算法从离散色散网格中提取费米面的边界多边形。

    对应种子项目 110_boundary_word_square 中的 wall-following 思想：
      从网格中找到 ε=μ 的等值线，将其近似为正交多连方边界，
      然后计算面积、质心、转动惯量。

    Parameters
    ----------
    epsilon_grid : ndarray, shape (Ny, Nx)
        能量网格。
    kx_grid, ky_grid : 1d arrays
        坐标轴。
    mu : float
        费米能级。

    Returns
    -------
    info : dict
        包含 'boundary_points', 'area_approx', 'centroid_approx', 'moment_approx'。
    """
    eps = epsilon_grid - mu
    Ny, Nx = eps.shape
    # 使用简单的 marching squares 提取边界
    # 对每个网格边，若两端符号不同，则记录交点
    boundary_pts = []
    dx = kx_grid[1] - kx_grid[0] if Nx > 1 else 1.0
    dy = ky_grid[1] - ky_grid[0] if Ny > 1 else 1.0

    # 水平边
    for iy in range(Ny):
        for ix in range(Nx - 1):
            e1, e2 = eps[iy, ix], eps[iy, ix + 1]
            if e1 * e2 < 0:
                t = safe_divide(abs(e1), abs(e1) + abs(e2))
                xb = kx_grid[ix] + t * dx
                yb = ky_grid[iy]
                boundary_pts.append((xb, yb))
    # 垂直边
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
    # 按角度排序以构成闭合多边形
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    pts = pts[order]

    # 多边形面积（Shoelace 公式）
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
    """
    计算费米面嵌套向量（nesting vector），即连接费米面上对径点的
    最大匹配动量转移 Q。

    公式：
        χ(Q) = - (1/N) Σ_k [f(ε(k)) - f(ε(k+Q))] / [ε(k) - ε(k+Q)]
    其中 f 为 Fermi-Dirac 分布（T=0 时即为阶跃函数）。
    本函数通过寻找使费米面重叠度最大的 Q 来近似 nesting vector。
    """
    KX, KY = np.meshgrid(kx_grid, ky_grid)
    mask = np.abs(epsilon_grid - mu) < delta_e
    if not np.any(mask):
        return np.array([0.0, 0.0]), 0.0

    pts = np.column_stack([KX[mask].ravel(), KY[mask].ravel()])
    best_overlap = 0.0
    best_q = np.array([0.0, 0.0])

    # 在离散网格上搜索 Q
    for qx in kx_grid:
        for qy in ky_grid:
            shifted = pts + np.array([qx, qy])
            # 将 shifted 映射回第一 BZ
            shifted[:, 0] = ((shifted[:, 0] + np.pi) % (2.0 * np.pi)) - np.pi
            shifted[:, 1] = ((shifted[:, 1] + np.pi) % (2.0 * np.pi)) - np.pi
            # 近似匹配：统计 shifted 点是否在费米面附近
            # 用 KDTree 太慢，改用简单计数
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
