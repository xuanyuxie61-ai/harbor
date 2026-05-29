#!/usr/bin/env python3
"""
proton_potential_solver.py
质子电势场求解模块（源自 poisson_2d_exact + hermite_cubic 项目）

融合二维泊松方程精确解与 Hermite 三次样条插值，求解 PEM 膜内
质子电势分布 φ_m(x,y)。

核心方程（二维泊松方程，稳态）：
    -∇·(σ_m(λ) ∇φ_m) = S_proton(x,y)

其中膜电导率 σ_m 强烈依赖于局部水含量 λ：
    σ_m(λ) = σ_m_ref · exp[ 1268 · (1/303 - 1/T) ] · (0.005139·λ - 0.00326)  [S/m]

本模块同时提供精确解验证函数与 Hermite 插值后处理。
"""

import numpy as np


def proton_potential_exact(x, y, params):
    """
    二维泊松方程的精确解析解（对应原项目 poisson_2d_exact.m）。
    采用 closed-form 解析函数：
        U(x,y) = 2(1+y) / ((3+x)² + (1+y)²)
    及其各阶偏导数，用于验证数值解。

    Parameters
    ----------
    x, y : float or ndarray
        空间坐标（膜平面内，已归一化到 [0,1]×[0,1]）
    params : dict
        物理参数

    Returns
    -------
    U, Ux, Uy, Uxx, Uxy, Uyy : ndarray
        函数值与各阶偏导数
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 精确解
    num = 2.0 * (1.0 + y)
    den = (3.0 + x) ** 2 + (1.0 + y) ** 2
    U = num / den

    # 一阶偏导
    dnum_dx = 0.0
    dnum_dy = 2.0
    dden_dx = 2.0 * (3.0 + x)
    dden_dy = 2.0 * (1.0 + y)

    Ux = (dnum_dx * den - num * dden_dx) / (den ** 2)
    Uy = (dnum_dy * den - num * dden_dy) / (den ** 2)

    # 二阶偏导（解析求导）
    Uxx = (-num * 2.0 * den - (dnum_dx * den - num * dden_dx) * 2.0 * dden_dx) / (den ** 3)
    Uyy = (2.0 * den - num * 2.0 - dnum_dy * 2.0 * (1.0 + y)) / (den ** 2)
    # 简化混导
    Uxy = (-dnum_dy * dden_dx * den - num * 2.0 * (1.0 + y) * dden_dx) / (den ** 3)

    # 处理边界外推
    U = np.where(den > 1e-12, U, 0.0)
    Ux = np.where(den > 1e-12, Ux, 0.0)
    Uy = np.where(den > 1e-12, Uy, 0.0)
    Uxx = np.where(den > 1e-12, Uxx, 0.0)
    Uxy = np.where(den > 1e-12, Uxy, 0.0)
    Uyy = np.where(den > 1e-12, Uyy, 0.0)

    return U, Ux, Uy, Uxx, Uxy, Uyy


def membrane_conductivity(lambda_w, T):
    """
    计算膜电导率 σ_m(λ, T)，基于 Springer 经验公式：
        σ_m = σ_0 · exp[ 1268 · (1/303.15 - 1/T) ] · (c1·λ - c2)
    其中 c1 = 0.005139, c2 = 0.00326。
    """
    sigma_0 = 1.0  # 参考值
    sigma = sigma_0 * np.exp(1268.0 * (1.0 / 303.15 - 1.0 / T))
    sigma *= np.clip(0.005139 * lambda_w - 0.00326, 1e-6, 10.0)
    return sigma


def solve_proton_potential(params, lambda_field=None):
    """
    使用有限差分法求解二维泊松方程得到质子电势场 φ_m。
    采用 5 点差分格式，结合带状矩阵求解。

    方程：
        -∇·(σ(λ) ∇φ) = S_p   在 Ω = [0,1]×[0,1]
        φ = φ_boundary       在 ∂Ω

    这里 S_p 代表电化学反应产生的质子源/汇。
    """
    Nx = params['Nx']
    Ny = max(5, Nx // 2)
    Lx, Ly = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x_grid = np.linspace(0.0, Lx, Nx)
    y_grid = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    # 若未提供水含量场，使用均匀分布
    if lambda_field is None:
        lambda_field = np.full((Nx, Ny), params['lambda_eq'])
    else:
        lambda_field = np.clip(lambda_field, 1.0, 22.0)

    # 计算局部电导率
    sigma = membrane_conductivity(lambda_field, params['T'])

    # 构造源项：模拟阴极侧（y=1）的质子消耗
    S_p = np.zeros((Nx, Ny))
    S_p[:, -1] = -5000.0  # 阴极边界汇
    S_p[:, 0] = 5000.0    # 阳极边界源

    # 构造有限差分离散矩阵 (Nx*Ny, Nx*Ny)
    n_unknowns = Nx * Ny
    A = np.zeros((n_unknowns, n_unknowns), dtype=float)
    b = np.zeros(n_unknowns, dtype=float)

    def idx(i, j):
        return i * Ny + j

    for i in range(Nx):
        for j in range(Ny):
            k = idx(i, j)
            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                # Dirichlet 边界：固定电势
                A[k, k] = 1.0
                # 边界值从精确解获取
                b[k] = proton_potential_exact(X[i, j], Y[i, j], params)[0]
            else:
                # 变系数 5 点差分
                s_c = sigma[i, j]
                s_e = 0.5 * (sigma[i, j] + sigma[i + 1, j])
                s_w = 0.5 * (sigma[i, j] + sigma[i - 1, j])
                s_n = 0.5 * (sigma[i, j] + sigma[i, j + 1])
                s_s = 0.5 * (sigma[i, j] + sigma[i, j - 1])

                A[k, k] = (s_e + s_w) / dx ** 2 + (s_n + s_s) / dy ** 2
                if i + 1 < Nx:
                    A[k, idx(i + 1, j)] = -s_e / dx ** 2
                if i - 1 >= 0:
                    A[k, idx(i - 1, j)] = -s_w / dx ** 2
                if j + 1 < Ny:
                    A[k, idx(i, j + 1)] = -s_n / dy ** 2
                if j - 1 >= 0:
                    A[k, idx(i, j - 1)] = -s_s / dy ** 2
                b[k] = S_p[i, j]

    # 求解线性系统
    phi_vec = np.linalg.solve(A, b)
    phi = phi_vec.reshape((Nx, Ny))

    return phi, x_grid, y_grid


# ---------------------------------------------------------------------------
# Hermite 三次插值（源自 hermite_cubic 项目）
# ---------------------------------------------------------------------------

def hermite_cubic_value(x1, x2, f1, d1, f2, d2, n_eval, x_eval=None):
    """
    在区间 [x1, x2] 上构造 Hermite 三次插值，并求值。
    对应原项目 hermite_cubic_value.m。

    Hermite 基函数表示：
        p(x) = f1·H1(t) + d1·H2(t)·h + f2·H3(t) + d2·H4(t)·h
    其中 t = (x - x1)/h, h = x2 - x1
    """
    h = x2 - x1
    if abs(h) < 1e-14:
        return np.full(n_eval, f1), np.zeros(n_eval), np.zeros(n_eval), np.zeros(n_eval)

    if x_eval is None:
        x_eval = np.linspace(x1, x2, n_eval)
    else:
        x_eval = np.clip(np.asarray(x_eval, dtype=float), x1, x2)

    t = (x_eval - x1) / h
    t2 = t * t
    t3 = t2 * t

    # Hermite 基函数
    H1 = 2.0 * t3 - 3.0 * t2 + 1.0
    H2 = t3 - 2.0 * t2 + t
    H3 = -2.0 * t3 + 3.0 * t2
    H4 = t3 - t2

    p = f1 * H1 + d1 * h * H2 + f2 * H3 + d2 * h * H4

    # 导数（对 x）
    dH1 = 6.0 * t2 - 6.0 * t
    dH2 = 3.0 * t2 - 4.0 * t + 1.0
    dH3 = -6.0 * t2 + 6.0 * t
    dH4 = 3.0 * t2 - 2.0 * t

    dp = (f1 * dH1 + d1 * h * dH2 + f2 * dH3 + d2 * h * dH4) / h
    d2p = (f1 * (12.0 * t - 6.0) + d1 * h * (6.0 * t - 4.0) +
           f2 * (-12.0 * t + 6.0) + d2 * h * (6.0 * t - 2.0)) / (h ** 2)
    d3p = (f1 * 12.0 + d1 * h * 6.0 + f2 * (-12.0) + d2 * h * 6.0) / (h ** 3)

    return p, dp, d2p, d3p


def hermite_cubic_spline(x_nodes, f_nodes, d_nodes, x_query):
    """
    分段 Hermite 三次样条求值，对应原项目 hermite_cubic_spline_value.m。
    """
    x_query = np.asarray(x_query, dtype=float)
    x_nodes = np.asarray(x_nodes, dtype=float)
    f_nodes = np.asarray(f_nodes, dtype=float)
    d_nodes = np.asarray(d_nodes, dtype=float)

    y_out = np.zeros_like(x_query)
    for k in range(x_query.size):
        xq = x_query[k]
        if xq <= x_nodes[0]:
            y_out[k] = f_nodes[0]
            continue
        if xq >= x_nodes[-1]:
            y_out[k] = f_nodes[-1]
            continue
        # 查找所在区间
        idx = np.searchsorted(x_nodes, xq) - 1
        idx = np.clip(idx, 0, len(x_nodes) - 2)
        p, _, _, _ = hermite_cubic_value(
            x_nodes[idx], x_nodes[idx + 1],
            f_nodes[idx], d_nodes[idx],
            f_nodes[idx + 1], d_nodes[idx + 1],
            1, x_eval=[xq]
        )
        y_out[k] = p[0]
    return y_out


def interpolate_proton_potential_hermite(phi, x_grid, y_grid, xq, yq):
    """
    使用 Hermite 三次样条对质子电势场进行二维插值。
    """
    # 先在 x 方向做 Hermite 插值
    Ny = y_grid.size
    phi_at_y = np.zeros(Ny)
    for j in range(Ny):
        # 构造导数（中心差分）
        dphi_dx = np.gradient(phi[:, j], x_grid)
        phi_at_y[j] = hermite_cubic_spline(x_grid, phi[:, j], dphi_dx, [xq])[0]

    # 再在 y 方向插值
    dphi_dy = np.gradient(phi_at_y, y_grid)
    phi_q = hermite_cubic_spline(y_grid, phi_at_y, dphi_dy, [yq])[0]
    return phi_q


if __name__ == '__main__':
    p = {'T': 353.15, 'lambda_eq': 14.0, 'Nx': 41, 'sigma_m_ref': 10.0}
    phi, xg, yg = solve_proton_potential(p)
    print("phi range:", phi.min(), phi.max())
