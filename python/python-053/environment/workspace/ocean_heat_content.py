"""
ocean_heat_content.py
=====================
基于 prism_witherden_rule (917_prism_witherden_rule) 的棱柱体高斯求积算法，
用于计算三维海洋热含量 (Ocean Heat Content, OHC) 及温跃层相关积分量。

科学背景
--------
海洋热含量是 ENSO 动力学的核心变量。根据 recharge-discharge oscillator 理论，
暖水体积 (WWV) 的充放电过程决定了 ENSO 事件的相位转换。温跃层深度变化
（由风应力驱动的 Rossby/Kelvin 波调制）通过热力学反馈影响 SST，
形成 Bjerknes 正反馈。

核心公式
--------
1. 海洋热含量定义：
   
   OHC = ρ_0 * c_p * ∭_{V} T(x, y, z) dV

   其中 ρ_0 = 1025 kg/m³ 为参考密度，c_p = 3993 J/(kg·K) 为比热容。

2. 单位三角棱柱上的高斯求积：
   对于定义在单位棱柱 P = {(x,y,z) | x≥0, y≥0, x+y≤1, 0≤z≤1} 上的函数 f，
   
   ∫_P f(x,y,z) dV ≈ Σ_{k=1}^{N} w_k * f(x_k, y_k, z_k)

   其中 (x_k, y_k, z_k, w_k) 为 Witherden-Vincent (2015) 棱柱求积规则。
   本实现采用三角形 7 点 Dunavant 规则（精度 5）与 z 方向 3 点 Gauss-Legendre
   规则的张量积构造，共 21 个求积点。

3. 棱柱上的单项式精确积分：
   对于单项式 x^α y^β z^γ：
   
   ∫_P x^α y^β z^γ dV = α! * β! / ((γ+1) * (α+β+2)!)

   单位棱柱体积 = 1/2。

4. 温跃层深度 (Thermocline Depth) 的热力学定义：
   
   D_{20}(x, y) = depth where T(x, y, z) = 20°C

5. 暖水体积 (Warm Water Volume, WWV)：
   
   WWV = ∬_{equatorial} max(0, D_{20}(x, y) - D_{clim}) dx dy
"""

import numpy as np
from typing import Tuple


def _triangle_rule_dunavant_7():
    """
    返回 Dunavant 7 点三角形求积规则（精度 5）。

    参考：Dunavant (1985), Int. J. Num. Meth. Eng.
    单位三角形：x≥0, y≥0, x+y≤1。
    """
    sqrt15 = np.sqrt(15.0)
    a1 = (6.0 + sqrt15) / 21.0
    b1 = (9.0 - 2.0 * sqrt15) / 21.0
    a2 = (6.0 - sqrt15) / 21.0
    b2 = (9.0 + 2.0 * sqrt15) / 21.0

    w1 = 9.0 / 80.0
    w2 = (155.0 + sqrt15) / 2400.0
    w3 = (155.0 - sqrt15) / 2400.0

    # 重心坐标 (L1, L2, L3) -> 直角坐标 (x=L1, y=L2)
    bary = np.array([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [a1, a1, b1],
        [a1, b1, a1],
        [b1, a1, a1],
        [a2, a2, b2],
        [a2, b2, a2],
        [b2, a2, a2],
    ])
    weights = np.array([w1, w2, w2, w2, w3, w3, w3])

    x_tri = bary[:, 0]
    y_tri = bary[:, 1]
    return x_tri, y_tri, weights


def _gauss_legendre_1d(n: int):
    """返回 [0,1] 上的 n 点 Gauss-Legendre 节点和权重。"""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def prism_unit_monomial_integral(expon: Tuple[int, int, int]) -> float:
    """
    计算单位棱柱上的单项式积分。

    公式：
    ∫_P x^α y^β z^γ dV = α! * β! / ((γ+1) * (α+β+2)!)

    参数
    ----
    expon : Tuple[int, int, int]
        指数 (α, β, γ)。

    返回
    ----
    value : float
        积分值。
    """
    alpha, beta, gamma = expon
    if alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("Exponents must be non-negative")

    import math
    num = math.factorial(alpha) * math.factorial(beta)
    den = (gamma + 1) * math.factorial(alpha + beta + 2)
    return num / den


def prism_witherden_rule(p: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回棱柱求积规则。

    本实现采用 Dunavant 三角形规则（精度 5，7 点）与 z 方向 Gauss-Legendre
    规则（精度 5，3 点）的张量积构造，共 21 个求积点。

    参数
    ----
    p : int
        精度阶数。支持 p ≤ 5。

    返回
    ----
    x, y, z : np.ndarray
        求积点坐标（单位棱柱）。
    w : np.ndarray
        求积权重。
    """
    if p < 0 or p > 10:
        raise ValueError("Precision p must be in [0, 10]")

    if p <= 5:
        x_tri, y_tri, w_tri = _triangle_rule_dunavant_7()
        n_tri = x_tri.shape[0]
        # z 方向：p=5 需要 3 点 GL
        z_1d, w_z = _gauss_legendre_1d(3)
        n_z = z_1d.shape[0]

        # 张量积
        x = np.zeros(n_tri * n_z)
        y = np.zeros(n_tri * n_z)
        z = np.zeros(n_tri * n_z)
        w = np.zeros(n_tri * n_z)

        idx = 0
        for i in range(n_tri):
            for j in range(n_z):
                x[idx] = x_tri[i]
                y[idx] = y_tri[i]
                z[idx] = z_1d[j]
                w[idx] = w_tri[i] * w_z[j]
                idx += 1

        return x, y, z, w

    # 更高精度：使用复合规则
    n_sub = 2
    x_base, y_base, z_base, w_base = prism_witherden_rule(p=5)
    x_all, y_all, z_all, w_all = [], [], [], []

    for i in range(n_sub):
        for j in range(n_sub):
            for k in range(n_sub):
                sx, sy, sz = i / n_sub, j / n_sub, k / n_sub
                ds = 1.0 / n_sub

                x_all.extend(sx + ds * x_base)
                y_all.extend(sy + ds * y_base)
                z_all.extend(sz + ds * z_base)
                w_all.extend((ds ** 3) * w_base)

    return np.array(x_all), np.array(y_all), np.array(z_all), np.array(w_all)


def map_prism_to_physical(x_ref: np.ndarray, y_ref: np.ndarray, z_ref: np.ndarray,
                          vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    将单位棱柱映射到物理棱柱，并计算 Jacobian 行列式。

    物理棱柱由 6 个顶点定义（底面三角形 + 顶面三角形）：
    vertices[0:3, :] = 底面 (z=0)
    vertices[3:6, :] = 顶面 (z=1)

    参数
    ----
    x_ref, y_ref, z_ref : np.ndarray
        参考坐标。
    vertices : np.ndarray, shape (6, 3)
        物理顶点。

    返回
    ----
    x, y, z : np.ndarray
        物理坐标。
    detJ : float
        Jacobian 行列式（体积缩放因子）。
    """
    if vertices.shape != (6, 3):
        raise ValueError("vertices must have shape (6, 3)")

    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    v3, v4, v5 = vertices[3], vertices[4], vertices[5]

    # 底面和顶面的形心
    z_bottom = (v0[2] + v1[2] + v2[2]) / 3.0
    z_top = (v3[2] + v4[2] + v5[2]) / 3.0
    height = z_top - z_bottom

    # 使用底面三角形进行 (x,y) 映射（假设棱柱侧面垂直）
    x_phys = v0[0] + (v1[0] - v0[0]) * x_ref + (v2[0] - v0[0]) * y_ref
    y_phys = v0[1] + (v1[1] - v0[1]) * x_ref + (v2[1] - v0[1]) * y_ref
    z_phys = z_bottom + height * z_ref

    # Jacobian: |∂(x,y,z)/∂(ξ,η,ζ)|
    # 对于直棱柱，底面面积决定 (x,y) 部分的 Jacobian
    area_base = 0.5 * abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )
    # 单位三角形面积为 1/2，所以物理三角形面积 = area_base
    # 映射 (ξ,η) -> (x,y) 的 Jacobian 行列式 = 2 * area_base
    detJ_xy = 2.0 * area_base
    detJ = detJ_xy * height

    return x_phys, y_phys, z_phys, detJ


def integrate_ohc_over_prism(temperature_func, vertices: np.ndarray,
                             rho0: float = 1025.0, cp: float = 3993.0) -> float:
    """
    计算单个物理棱柱内的海洋热含量。

    公式：
    OHC_prism = ρ_0 * c_p * ∫∫∫_{prism} T(x,y,z) dV

    参数
    ----
    temperature_func : callable
        温度场函数 T(x, y, z) -> float。
    vertices : np.ndarray, shape (6, 3)
        棱柱顶点。
    rho0, cp : float
        参考密度和比热容。

    返回
    ----
    ohc : float
        该棱柱的热含量 (J)。
    """
    x_ref, y_ref, z_ref, w = prism_witherden_rule(p=5)
    x_phys, y_phys, z_phys, detJ = map_prism_to_physical(x_ref, y_ref, z_ref, vertices)

    # 求积：∫ T dV = Σ w_k * T(x_k, y_k, z_k) * |detJ|
    # 注意：w_k 已经包含单位三角形的面积权重（1/2），
    # detJ 包含从单位棱柱到物理棱柱的全部 Jacobian
    integral = 0.0
    for i in range(x_ref.shape[0]):
        t_val = temperature_func(x_phys[i], y_phys[i], z_phys[i])
        if not np.isfinite(t_val):
            t_val = 0.0  # 数值鲁棒性
        integral += w[i] * t_val

    ohc = rho0 * cp * integral * abs(detJ)
    return float(ohc)


def thermocline_depth_from_profile(z: np.ndarray, t: np.ndarray,
                                   t_crit: float = 20.0) -> float:
    """
    从温度垂向廓线计算温跃层深度（等温线深度）。

    参数
    ----
    z : np.ndarray
        深度坐标（米，正值向下或负值向上）。
    t : np.ndarray
        温度（℃）。
    t_crit : float
        临界温度，默认 20℃。

    返回
    ----
    d20 : float
        温跃层深度（米，正值）。
    """
    if z.shape != t.shape or z.ndim != 1:
        raise ValueError("z and t must be 1D arrays of same shape")
    if z.shape[0] < 2:
        return float(np.nan)

    # 确保深度向下递增（z 变得更负或更正取决于约定）
    # 本函数约定：z 为负值表示深度（例如 z=0 海面, z=-200 为 200m 深度）
    # 或 z 为正值表示深度。我们统一处理为绝对深度。
    z_abs = np.abs(z)

    # 按深度排序
    sort_idx = np.argsort(z_abs)
    z_sorted = z_abs[sort_idx]
    t_sorted = t[sort_idx]

    # 线性插值找到 T = t_crit 的深度
    for i in range(z_sorted.shape[0] - 1):
        if (t_sorted[i] - t_crit) * (t_sorted[i + 1] - t_crit) <= 0.0:
            if abs(t_sorted[i + 1] - t_sorted[i]) < 1e-10:
                return float(z_sorted[i])
            frac = (t_crit - t_sorted[i]) / (t_sorted[i + 1] - t_sorted[i])
            return float(z_sorted[i] + frac * (z_sorted[i + 1] - z_sorted[i]))

    # 未找到：取边界
    if np.all(t_sorted > t_crit):
        return float(z_sorted[-1])
    return float(z_sorted[0])


def warm_water_volume(thermocline_depth: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                      clim_depth: np.ndarray, dx: float, dy: float) -> float:
    """
    计算赤道暖水体积 (WWV)。

    公式：
    WWV = Σ_{i,j} max(0, D_{20}(i,j) - D_{clim}(i,j)) * dx * dy * cos(lat_j)

    参数
    ----
    thermocline_depth : np.ndarray, shape (nx, ny)
        温跃层深度场（米，正值）。
    lon, lat : np.ndarray
        经纬度网格。
    clim_depth : np.ndarray, shape (nx, ny)
        气候态温跃层深度。
    dx, dy : float
        网格间距（米）。

    返回
    ----
    wwv : float
        暖水体积 (m³)。
    """
    if thermocline_depth.shape != clim_depth.shape:
        raise ValueError("Shape mismatch")

    anomaly = thermocline_depth - clim_depth
    anomaly = np.where(anomaly > 0, anomaly, 0.0)

    cos_lat = np.cos(np.radians(lat))[None, :]
    area_element = dx * dy * cos_lat

    wwv = np.sum(anomaly * area_element)
    return float(wwv)
