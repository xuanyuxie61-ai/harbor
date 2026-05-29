"""
property_interpolation.py
=========================
物性插值与数值积分模块。

本模块包含：
1. Shepard 反距离加权插值（1D）（源自项目 1071_shepard_interp_1d）
2. 梯形数值积分（源自项目 945_quad_trapezoid）
3. 物性数据表管理与插值应用

科学背景
--------
在精馏塔设计中，汽液平衡数据往往来自离散的实验测量点。Shepard 插值
提供了一种基于反距离加权的局部平滑插值方法：

    f(x) = Σ_{i=1}^{N} w_i(x) f_i

其中权重：
    w_i(x) = |x - x_i|^{-p} / Σ_j |x - x_j|^{-p}

当 x 恰好等于某个数据点 x_i 时，w_i = 1，其余为 0，保证插值精确性。

梯形法则用于计算沿塔高的传质通量积分：
    Q = ∫_a^b f(z) dz ≈ h/2 [f(z_0) + 2 Σ_{i=1}^{n-1} f(z_i) + f(z_n)]
"""

import numpy as np
from utils import ensure_positive


# ---------------------------------------------------------------------------
# Shepard 1D 插值（源自项目 1071_shepard_interp_1d）
# ---------------------------------------------------------------------------

def shepard_interp_1d(nd, xd, yd, p, ni, xi):
    """
    一维 Shepard 反距离加权插值。

    Parameters
    ----------
    nd : int
        数据点数量。
    xd : ndarray, shape (nd,)
        数据点横坐标。
    yd : ndarray, shape (nd,)
        数据点纵坐标。
    p : float
        距离幂次，p>0 时距离越近权重越大；p=0 时等权平均。
    ni : int
        插值点数量。
    xi : ndarray, shape (ni,)
        插值点横坐标。

    Returns
    -------
    yi : ndarray, shape (ni,)
        插值结果。
    """
    xd = np.asarray(xd, dtype=float).reshape(-1)
    yd = np.asarray(yd, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)

    if xd.size != nd or yd.size != nd:
        nd = min(xd.size, yd.size)
        xd = xd[:nd]
        yd = yd[:nd]
    if xi.size != ni:
        ni = xi.size

    yi = np.zeros(ni, dtype=float)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd, dtype=float) / nd
        else:
            dist = np.abs(xi[i] - xd)
            exact = np.where(dist < 1e-15)[0]
            if exact.size > 0:
                w = np.zeros(nd, dtype=float)
                w[exact[0]] = 1.0
            else:
                w = 1.0 / (dist ** p)
                s = np.sum(w)
                if s > 1e-15:
                    w = w / s
                else:
                    w = np.ones(nd, dtype=float) / nd
        yi[i] = np.dot(w, yd)

    return yi


# ---------------------------------------------------------------------------
# 梯形数值积分（源自项目 945_quad_trapezoid）
# ---------------------------------------------------------------------------

def quad_trapezoid(f_func, a, b, n):
    """
    复合梯形数值积分。

    Parameters
    ----------
    f_func : callable
        被积函数 f(x)。
    a, b : float
        积分区间。
    n : int
        子区间数。

    Returns
    -------
    q : float
        积分估计值。
    """
    if n < 1:
        n = 1
    a = float(a)
    b = float(b)
    x = np.linspace(a, b, n + 1)
    fx = np.asarray([f_func(xi) for xi in x], dtype=float)
    h = (b - a) / n
    q = (h / 2.0) * (fx[0] + 2.0 * np.sum(fx[1:n]) + fx[n])
    return float(q)


# ---------------------------------------------------------------------------
# 物性数据表与插值应用
# ---------------------------------------------------------------------------

def interpolate_vle_data(z_data, T_data, x_data, y_data, z_query, p=2.0):
    """
    沿塔高 z 插值 VLE 数据。

    Parameters
    ----------
    z_data : ndarray
        塔高位置 [m]。
    T_data : ndarray
        温度数据 [K]。
    x_data : ndarray, shape (ndata, nc)
        液相组成。
    y_data : ndarray, shape (ndata, nc)
        汽相组成。
    z_query : ndarray
        查询位置。
    p : float
        Shepard 幂次。

    Returns
    -------
    T_interp : ndarray
        插值温度。
    x_interp : ndarray, shape (nquery, nc)
        插值液相组成。
    y_interp : ndarray, shape (nquery, nc)
        插值汽相组成。
    """
    nd = len(z_data)
    nc = x_data.shape[1] if x_data.ndim > 1 else 1
    nq = len(z_query)

    T_interp = shepard_interp_1d(nd, z_data, T_data, p, nq, z_query)

    x_interp = np.zeros((nq, nc), dtype=float)
    y_interp = np.zeros((nq, nc), dtype=float)

    for j in range(nc):
        x_interp[:, j] = shepard_interp_1d(nd, z_data, x_data[:, j], p, nq, z_query)
        y_interp[:, j] = shepard_interp_1d(nd, z_data, y_data[:, j], p, nq, z_query)

    # 归一化组成
    for i in range(nq):
        xs = np.sum(x_interp[i, :])
        ys = np.sum(y_interp[i, :])
        if xs > 1e-12:
            x_interp[i, :] /= xs
        if ys > 1e-12:
            y_interp[i, :] /= ys

    return T_interp, x_interp, y_interp


def integrate_mass_transfer_flux(z_nodes, N_A_func):
    """
    使用梯形积分计算沿塔高的总传质量 [mol/(m^2 s)]。

    Parameters
    ----------
    z_nodes : ndarray
        离散节点 [m]。
    N_A_func : callable
        传质通量函数 N_A(z)。

    Returns
    -------
    total_mass_transfer : float
        总传质量。
    """
    a = float(z_nodes[0])
    b = float(z_nodes[-1])
    n = len(z_nodes) - 1
    if n < 1:
        n = 1
    return quad_trapezoid(N_A_func, a, b, n)
