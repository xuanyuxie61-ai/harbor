"""
potential_energy.py
势能面计算与插值模块

融合原项目:
  - 927_pwl_interp_2d: 二维分段线性插值 → 粗粒化势能面插值
  - 818_normal_ode: 正态分布ODE精确解 → 高斯型势阱描述

科学背景:
  在DNA-蛋白质相互作用中，结合自由能面可用多维高斯型势阱近似:
    V(r) = -ε · exp( -|r - r₀|² / (2σ²) )
  其中 ε 为势阱深度，σ 为范围参数。

  对于非均匀势能面，需要在离散网格上插值:
    给定网格点 {(x_i, y_j)} 上的势能值 V_{ij}，
    任意点 (x, y) 的势能通过双线性/分段线性插值得到。

  此外，正态分布ODE dy/dt = -t·y 的解 y(t) = exp(-t²/2)/√(2π)
  描述了蛋白质浓度在势阱中的稳态分布。
"""

import numpy as np


def normal_exact(t: np.ndarray) -> np.ndarray:
    """
    正态分布ODE的精确解
        dy/dt = -t·y
    解析解:
        y(t) = exp(-t²/2) / sqrt(2π)

    参数:
        t: 时间/位置变量
    Returns:
        y: 高斯分布值
    """
    t = np.asarray(t, dtype=float)
    return np.exp(-t ** 2 / 2.0) / np.sqrt(2.0 * np.pi)


def normal_deriv(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    正态分布ODE的右端项
        dydt = -t * y
    """
    return -t * y


def pwl_interp_2d_scalar(xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
                         xi: float, yi: float) -> float:
    """
    基于 pwl_interp_2d 的二维分段线性插值 (单点)

    参数:
        xd: shape (nxd,) 排序后的x坐标
        yd: shape (nyd,) 排序后的y坐标
        zd: shape (nxd, nyd) 网格值
        xi, yi: 插值点坐标

    Returns:
        zi: 插值结果
    """
    nxd = len(xd)
    nyd = len(yd)

    if nxd < 2 or nyd < 2:
        raise ValueError("Need at least 2 points in each dimension")

    # 找到包含xi的区间
    i = -1
    for idx in range(nxd - 1):
        if xd[idx] <= xi <= xd[idx + 1]:
            i = idx
            break
    if i == -1:
        return np.inf

    j = -1
    for idx in range(nyd - 1):
        if yd[idx] <= yi <= yd[idx + 1]:
            j = idx
            break
    if j == -1:
        return np.inf

    # 判断落在哪个三角形
    # 矩形单元 (i,j) 到 (i+1,j+1)
    # 对角线从 (i,j+1) 到 (i+1,j)
    diag_y = yd[j + 1] + (yd[j] - yd[j + 1]) * (xi - xd[i]) / (xd[i + 1] - xd[i])

    if yi < diag_y:
        # 下三角形: (i,j), (i+1,j), (i,j+1)
        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return zd[i, j]
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
    else:
        # 上三角形: (i,j+1), (i+1,j), (i+1,j+1)
        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]
        dxb = xd[i + 1] - xd[i + 1]
        dyb = yd[j] - yd[j + 1]
        dxi = xi - xd[i + 1]
        dyi = yi - yd[j + 1]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return zd[i + 1, j + 1]
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]


def pwl_interp_2d_vector(xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
                         xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """
    多点二维分段线性插值
    """
    ni = len(xi)
    zi = np.zeros(ni, dtype=float)
    for k in range(ni):
        zi[k] = pwl_interp_2d_scalar(xd, yd, zd, xi[k], yi[k])
    return zi


def build_lennard_jones_potential_grid(x_range: tuple, y_range: tuple,
                                       nx: int, ny: int,
                                       sigma_nm: float = 0.4,
                                       epsilon_kJ_mol: float = 4.0) -> tuple:
    """
    构建Lennard-Jones型势能面网格

    LJ势能:
        V(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]

    参数:
        x_range, y_range: (min, max) 范围 (nm)
        nx, ny: 网格点数
        sigma: LJ直径 (nm)
        epsilon: 势阱深度 (kJ/mol)

    Returns:
        xd, yd, zd: 网格坐标和势能值
    """
    xd = np.linspace(x_range[0], x_range[1], nx)
    yd = np.linspace(y_range[0], y_range[1], ny)
    zd = np.zeros((nx, ny), dtype=float)

    for i in range(nx):
        for j in range(ny):
            r = np.sqrt(xd[i] ** 2 + yd[j] ** 2)
            if r < 0.1:
                r = 0.1  # 截断避免奇点
            sr6 = (sigma_nm / r) ** 6
            zd[i, j] = 4.0 * epsilon_kJ_mol * (sr6 ** 2 - sr6)

    return xd, yd, zd


def build_morse_potential_1d(r: np.ndarray, D_e: float = 50.0,
                             a: float = 2.0, r_e: float = 0.4) -> np.ndarray:
    """
    Morse势能描述化学键

    公式:
        V(r) = D_e [ 1 - exp(-a(r - r_e)) ]² - D_e

    参数:
        r: 键长数组 (nm)
        D_e: 解离能 (kJ/mol)
        a: 势阱宽度参数 (1/nm)
        r_e: 平衡键长 (nm)

    Returns:
        V: 势能值 (kJ/mol)
    """
    r = np.asarray(r, dtype=float)
    return D_e * (1.0 - np.exp(-a * (r - r_e))) ** 2 - D_e


def gaussian_binding_potential_2d(x: np.ndarray, y: np.ndarray,
                                  x0: float, y0: float,
                                  sigma: float, depth: float) -> np.ndarray:
    """
    二维高斯型结合势阱

    V(x,y) = -depth * exp( -[(x-x0)² + (y-y0)²] / (2σ²) )
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return -depth * np.exp(-r2 / (2.0 * sigma ** 2))


def compute_rad51_dna_binding_energy(distance_nm: float,
                                      well_depth_kJ_mol: float = 35.0,
                                      sigma_nm: float = 0.25) -> float:
    """
    计算RAD51与DNA结合的自由能

    简化模型: 高斯型结合势
        ΔG_bind(r) = -ε exp(-r²/2σ²)
    """
    if distance_nm < 0:
        raise ValueError("distance must be non-negative")
    return -well_depth_kJ_mol * np.exp(-distance_nm ** 2 / (2.0 * sigma_nm ** 2))


def potential_force_from_grid(xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
                              x: float, y: float) -> tuple:
    """
    从势能面网格计算力 (负梯度)

    F = -∇V ≈ (-∂V/∂x, -∂V/∂y)

    使用中心差分:
        ∂V/∂x ≈ [V(x+h,y) - V(x-h,y)] / (2h)
    """
    h_x = (xd[-1] - xd[0]) / (len(xd) - 1)
    h_y = (yd[-1] - yd[0]) / (len(yd) - 1)

    # 边界检查与插值
    if x < xd[0] or x > xd[-1] or y < yd[0] or y > yd[-1]:
        return 0.0, 0.0

    # 用插值计算附近点
    v_px = pwl_interp_2d_scalar(xd, yd, zd, x + h_x, y)
    v_mx = pwl_interp_2d_scalar(xd, yd, zd, x - h_x, y)
    v_py = pwl_interp_2d_scalar(xd, yd, zd, x, y + h_y)
    v_my = pwl_interp_2d_scalar(xd, yd, zd, x, y - h_y)

    # 处理无穷大值
    vals = [v_px, v_mx, v_py, v_my]
    for i in range(len(vals)):
        if np.isinf(vals[i]):
            vals[i] = 0.0

    fx = -(vals[0] - vals[1]) / (2.0 * h_x)
    fy = -(vals[2] - vals[3]) / (2.0 * h_y)
    return fx, fy
