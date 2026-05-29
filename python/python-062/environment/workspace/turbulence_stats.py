"""
turbulence_stats.py
================================================================================
湍流统计量计算模块 —— 基于种子项目 805_nintlib（蒙特卡洛积分）

本模块提供边界层湍流的 Reynolds 平均统计量、结构函数与能谱计算。
所有统计量基于体积平均或平面平均。

核心物理公式
--------------------------------------------------------------------------------
Reynolds 分解：φ = ⟨φ⟩ + φ'，其中 ⟨·⟩ 表示系综/空间平均。

湍动能（TKE）：
    k = 1/2 (⟨u'u'⟩ + ⟨v'v'⟩ + ⟨w'w'⟩)

Reynolds 应力张量：
    R_{ij} = ⟨u_i' u_j'⟩

湍流热通量：
    q_i = ⟨u_i' θ'⟩  [W/m²]

二阶结构函数（纵向）：
    D_LL(r) = ⟨(u_L(x+r) - u_L(x))²⟩

Kolmogorov 的 2/3 定律（惯性子区）：
    D_LL(r) = C_2 (ε r)^{2/3}

其中 C_2 ≈ 2.0 为普适常数。

Taylor 微尺度：
    λ² = ⟨u'²⟩ / ⟨(∂u'/∂x)²⟩

积分尺度：
    L = (3π / 4) E(0) / ⟨(u')²⟩
"""

import numpy as np


def plane_average(field):
    """
    计算水平平面平均（沿 x, y 平均，保留 z 方向剖面）。

    参数
    ----------
    field : np.ndarray, shape (nx, ny, nz)

    返回
    -------
    profile : np.ndarray, shape (nz,)
    """
    return np.mean(np.mean(field, axis=0), axis=0)


def volume_average(field):
    """
    计算体积平均。

    参数
    ----------
    field : np.ndarray

    返回
    -------
    mean : float
    """
    return np.mean(field)


def compute_tke(u, v, w):
    """
    计算湍动能 TKE = 0.5 (u'² + v'² + w'²)。

    参数
    ----------
    u, v, w : np.ndarray

    返回
    -------
    tke : np.ndarray
        每个网格点的 TKE
    tke_mean : float
        体积平均 TKE
    """
    # 去均值
    up = u - volume_average(u)
    vp = v - volume_average(v)
    wp = w - volume_average(w)

    tke = 0.5 * (up**2 + vp**2 + wp**2)
    tke_mean = volume_average(tke)
    return tke, tke_mean


def compute_reynolds_stresses(u, v, w):
    """
    计算 Reynolds 应力张量分量。

    参数
    ----------
    u, v, w : np.ndarray

    返回
    -------
    R : dict
        {'uu', 'vv', 'ww', 'uv', 'uw', 'vw'} 的体积平均值
    """
    up = u - volume_average(u)
    vp = v - volume_average(v)
    wp = w - volume_average(w)

    R = {
        'uu': volume_average(up * up),
        'vv': volume_average(vp * vp),
        'ww': volume_average(wp * wp),
        'uv': volume_average(up * vp),
        'uw': volume_average(up * wp),
        'vw': volume_average(vp * wp),
    }
    return R


def compute_heat_flux(u, v, w, theta):
    """
    计算湍流热通量 ⟨u_i' θ'⟩。

    参数
    ----------
    u, v, w, theta : np.ndarray

    返回
    -------
    qx, qy, qz : float
    """
    up = u - volume_average(u)
    vp = v - volume_average(v)
    wp = w - volume_average(w)
    tp = theta - volume_average(theta)

    qx = volume_average(up * tp)
    qy = volume_average(vp * tp)
    qz = volume_average(wp * tp)
    return qx, qy, qz


def longitudinal_structure_function(u, axis=0, max_lag=None):
    """
    计算纵向二阶结构函数 D_LL(r)。

    参数
    ----------
    u : np.ndarray, shape (nx, ny, nz)
        速度分量
    axis : int
        分离方向
    max_lag : int, optional
        最大分离距离

    返回
    -------
    r : np.ndarray
        分离距离（网格单位）
    D_ll : np.ndarray
        结构函数值
    """
    if max_lag is None:
        max_lag = u.shape[axis] // 2

    D_ll = np.zeros(max_lag, dtype=np.float64)
    counts = np.zeros(max_lag, dtype=np.int64)

    # 沿指定轴计算差分
    for lag in range(1, max_lag + 1):
        slc1 = [slice(None)] * 3
        slc2 = [slice(None)] * 3
        slc1[axis] = slice(lag, None)
        slc2[axis] = slice(None, -lag)

        diff = u[tuple(slc1)] - u[tuple(slc2)]
        D_ll[lag - 1] = np.sum(diff**2)
        counts[lag - 1] = diff.size

    r = np.arange(1, max_lag + 1)
    D_ll = D_ll / np.maximum(counts, 1)

    return r, D_ll


def monte_carlo_turbulence_stat(samples, dim=3, eval_num=10000):
    """
    使用蒙特卡洛采样估计高维湍流统计量积分。

    参数
    ----------
    samples : callable
        采样函数 samples(dim, x)
    dim : int
        空间维度
    eval_num : int
        采样点数

    返回
    -------
    result : float
        积分估计值
    """
    # 简化实现：在单位超立方体上均匀采样
    total = 0.0
    for _ in range(eval_num):
        x = np.random.rand(dim)
        total += samples(dim, x)

    volume = 1.0  # 单位超立方体
    result = total * volume / eval_num
    return result


def compute_kolmogorov_scales(epsilon, nu):
    """
    计算 Kolmogorov 微观尺度。

    参数
    ----------
    epsilon : float
        耗散率（m²/s³）
    nu : float
        运动粘性（m²/s）

    返回
    -------
    eta : float
        Kolmogorov 长度尺度（m）
    tau_eta : float
        Kolmogorov 时间尺度（s）
    u_eta : float
        Kolmogorov 速度尺度（m/s）
    Re_lambda : float
        Taylor 尺度 Reynolds 数
    """
    eta = (nu**3 / epsilon) ** 0.25
    tau_eta = (nu / epsilon) ** 0.5
    u_eta = (nu * epsilon) ** 0.25

    # Taylor 尺度 Reynolds 数（近似）
    Re_lambda = (15.0 * epsilon * (nu / epsilon) ** 0.5 / nu) ** 0.5

    return eta, tau_eta, u_eta, Re_lambda
