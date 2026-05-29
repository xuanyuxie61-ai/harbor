"""
property_interpolator.py
材料物性插值模块
基于重心拉格朗日插值（Barycentric Lagrange Interpolation）在 Chebyshev 节点上
对温度相关的材料热物性（导热系数、比热容、密度等）进行高精度插值。
原项目映射:
  - 072_barycentric_interp_1d (一维重心 Chebyshev 插值)
"""

import numpy as np


def chebyshev1_nodes(n, a=-1.0, b=1.0):
    """
    生成 Chebyshev Type I 节点（第一类 Chebyshev 点）。
    
    x_i = cos((2i + 1) * π / (2n)), i = 0, ..., n-1
    
    映射到区间 [a, b]:
        x_mapped = (b - a)/2 * x_i + (b + a)/2
    """
    i = np.arange(n, dtype=np.float64)
    x = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    return x_mapped


def barycentric_weights_cheby1(n):
    """
    计算第一类 Chebyshev 节点的重心权重。
    
    w_j = (-1)^j * sin((2j+1) * π / (2n))
    """
    j = np.arange(n, dtype=np.float64)
    w = ((-1.0) ** j) * np.sin((2.0 * j + 1.0) * np.pi / (2.0 * n))
    return w


def barycentric_interp_1d(xd, yd, xi):
    """
    一维重心拉格朗日插值。
    映射自 lagcheby1_interp_1d.m。
    
    插值公式:
        p(x) = Σ (w_j / (x - x_j)) * y_j / Σ (w_j / (x - x_j))
    
    参数:
        xd: 数据点 (n,)
        yd: 数据值 (n,)
        xi: 插值点 (m,)
    返回:
        yi: 插值值 (m,)
    """
    xd = np.asarray(xd, dtype=np.float64).ravel()
    yd = np.asarray(yd, dtype=np.float64).ravel()
    xi = np.asarray(xi, dtype=np.float64).ravel()
    n = len(xd)

    # 计算重心权重（假设 xd 为 Chebyshev 节点）
    wd = barycentric_weights_cheby1(n)

    numer = np.zeros(len(xi), dtype=np.float64)
    denom = np.zeros(len(xi), dtype=np.float64)
    exact = np.zeros(len(xi), dtype=np.int64) - 1

    for j in range(n):
        diff = xi - xd[j]
        # 精确匹配的点
        mask = np.abs(diff) < 1e-14
        exact[mask] = j
        # 避免除以零
        diff_safe = np.where(np.abs(diff) < 1e-14, 1.0, diff)
        t = wd[j] / diff_safe
        numer += t * yd[j]
        denom += t

    yi = numer / denom
    # 对精确匹配的点直接赋值为数据值
    valid_exact = exact >= 0
    yi[valid_exact] = yd[exact[valid_exact]]
    return yi


def thermal_conductivity_interpolator(T_points, kappa_points):
    """
    构造温度相关导热系数 κ(T) 的插值器。
    
    典型生物质导热系数随温度变化（经验数据）:
        T [K]    κ [W/(m·K)]
        300      0.12
        400      0.14
        500      0.16
        600      0.18
        700      0.20
        800      0.15
        900      0.10
    """
    xd = np.asarray(T_points, dtype=np.float64)
    yd = np.asarray(kappa_points, dtype=np.float64)

    def interpolator(T):
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.array([T_arr])
        result = barycentric_interp_1d(xd, yd, T_arr)
        # 物理约束: 导热系数非负
        return np.maximum(result, 0.01)

    return interpolator


def specific_heat_interpolator(T_points, Cp_points):
    """
    构造温度相关比热容 Cp(T) 的插值器。
    """
    xd = np.asarray(T_points, dtype=np.float64)
    yd = np.asarray(Cp_points, dtype=np.float64)

    def interpolator(T):
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.array([T_arr])
        result = barycentric_interp_1d(xd, yd, T_arr)
        return np.maximum(result, 100.0)

    return interpolator


def density_interpolator(T_points, rho_points):
    """
    构造温度相关密度 ρ(T) 的插值器。
    """
    xd = np.asarray(T_points, dtype=np.float64)
    yd = np.asarray(rho_points, dtype=np.float64)

    def interpolator(T):
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.array([T_arr])
        result = barycentric_interp_1d(xd, yd, T_arr)
        return np.maximum(result, 10.0)

    return interpolator


def default_biomass_properties():
    """
    返回生物质典型热物性数据及插值器。
    """
    T_data = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0], dtype=np.float64)
    kappa_data = np.array([0.12, 0.14, 0.16, 0.18, 0.20, 0.15, 0.10], dtype=np.float64)
    Cp_data = np.array([1200.0, 1350.0, 1500.0, 1700.0, 1850.0, 1600.0, 1400.0], dtype=np.float64)
    rho_data = np.array([550.0, 480.0, 420.0, 350.0, 280.0, 220.0, 180.0], dtype=np.float64)

    kappa_interp = thermal_conductivity_interpolator(T_data, kappa_data)
    Cp_interp = specific_heat_interpolator(T_data, Cp_data)
    rho_interp = density_interpolator(T_data, rho_data)

    return {
        'T_data': T_data,
        'kappa_interp': kappa_interp,
        'Cp_interp': Cp_interp,
        'rho_interp': rho_interp
    }
