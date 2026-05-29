"""
interpolation.py
================
二维分段线性插值与催化剂场量重构。

基于种子项目 927_pwl_interp_2d 重构：
- 原项目实现二维矩形网格上的分段线性插值；
- 在本系统中用于催化剂截面二维浓度场、温度场的插值重构，
  以及将一维径向解映射到二维截面的后处理。

插值数学：
    对于矩形单元 [x_i, x_{i+1}] × [y_j, y_{j+1}]，
    将单元沿对角线分为两个三角形，
    在三角形内使用重心坐标进行线性插值：
        z(x, y) = α z_1 + β z_2 + γ z_3
    其中 α + β + γ = 1，且 α, β, γ ≥ 0 保证点在三角形内部。
"""

import numpy as np


class InterpolationError(Exception):
    """插值异常。"""
    pass


def bracket_index(sorted_vec, xq):
    """
    在有序向量中找到 xq 所在的区间索引 i，使得
    sorted_vec[i] ≤ xq ≤ sorted_vec[i+1]。

    若 xq 超出范围，返回 -1。
    """
    n = sorted_vec.size
    if n < 2:
        return -1
    if xq < sorted_vec[0] - 1e-14 or xq > sorted_vec[-1] + 1e-14:
        return -1
    # 二分搜索
    lo, hi = 0, n - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_vec[mid] <= xq <= sorted_vec[mid + 1]:
            return mid
        elif xq < sorted_vec[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    # 边界容差
    if abs(xq - sorted_vec[0]) < 1e-12:
        return 0
    if abs(xq - sorted_vec[-1]) < 1e-12:
        return n - 2
    return -1


def pwl_interp_2d_scalar(xd, yd, zd, xi, yi):
    """
    二维分段线性插值（标量点）。

    Parameters
    ----------
    xd : ndarray, shape (nxd,)
        有序 x 坐标。
    yd : ndarray, shape (nyd,)
        有序 y 坐标。
    zd : ndarray, shape (nxd, nyd)
        网格点上的函数值。
    xi, yi : float
        插值点坐标。

    Returns
    -------
    zi : float
        插值结果。若点在范围外返回 np.inf。
    """
    i = bracket_index(xd, xi)
    j = bracket_index(yd, yi)
    if i == -1 or j == -1:
        return np.inf

    # 判断插值点位于哪个三角形
    # 矩形四个角：
    #   (i, j+1) --- (i+1, j+1)
    #      |   \       |
    #      |    \      |
    #   (i, j)   --- (i+1, j)
    #
    # 对角线方程：y = yd[j] + (yd[j+1]-yd[j]) * (x-xd[i]) / (xd[i+1]-xd[i])
    # 若 yi 小于对角线上的 y 值，点在左下三角形
    dx = xd[i + 1] - xd[i]
    dy = yd[j + 1] - yd[j]
    if dx <= 0 or dy <= 0:
        raise InterpolationError("网格必须严格单调递增")

    y_diag = yd[j] + dy * (xi - xd[i]) / dx

    if yi < y_diag:
        # 左下三角形：(i, j), (i+1, j), (i, j+1)
        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]  # 0
        dxb = xd[i] - xd[i]   # 0
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
        # 右上三角形：(i, j+1), (i+1, j), (i+1, j+1)
        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]  # 0
        dxb = xd[i + 1] - xd[i + 1]  # 0
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
    """
    二维分段线性插值（批量点）。

    Parameters
    ----------
    xd : ndarray, shape (nxd,)
    yd : ndarray, shape (nyd,)
    zd : ndarray, shape (nxd, nyd)
    xi, yi : ndarray
        插值点坐标数组，形状一致。

    Returns
    -------
    zi : ndarray
        与 xi, yi 同形状的插值结果。
    """
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
    """
    将一维径向解映射到二维圆盘截面上，使用分段线性插值。

    生成极坐标网格 (r, θ) 并通过插值得到二维场分布。
    返回二维直角坐标网格上的值，可用于后续分析（不绘图）。

    Parameters
    ----------
    r_nodes : ndarray
        径向节点（已排序，包含 0 和 R）。
    values_r : ndarray
        径向节点上的函数值。
    n_theta : int
        角度方向离散数。
    n_r : int
        径向方向离散数。

    Returns
    -------
    X, Y, Z : ndarray
        二维直角坐标网格上的坐标与插值结果。
    """
    R = r_nodes[-1]
    # 生成极坐标采样点
    r_samples = np.linspace(0.0, R, n_r)
    theta_samples = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    # 径向一维插值（线性）
    values_interp = np.interp(r_samples, r_nodes, values_r,
                              left=values_r[0], right=values_r[-1])

    # 映射到二维
    R_grid, T_grid = np.meshgrid(r_samples, theta_samples)
    X = R_grid * np.cos(T_grid)
    Y = R_grid * np.sin(T_grid)
    Z = np.tile(values_interp.reshape(1, -1), (n_theta, 1))
    return X, Y, Z
