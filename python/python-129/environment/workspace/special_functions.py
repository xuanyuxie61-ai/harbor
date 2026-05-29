"""
special_functions.py
基于 079_besselj 的 Bessel 函数数值表与计算思想，
扩展为血凝 clot 径向分布建模所需的特殊函数库。

在血栓形成过程中，纤维蛋白聚合体的空间分布可用柱坐标系下的
修正 Bessel 函数 I_ν(r) 与 K_ν(r) 描述，表征 clot 核心到边缘的
浓度衰减模式。
"""

import numpy as np
from scipy.special import jv, iv, kv, gamma


def bessel_jx_fractional(nu, x):
    """
    计算非整数阶 Bessel 函数 J_nu(x)。
    使用级数展开：
        J_nu(x) = Σ_{m=0}^{∞} [(-1)^m / (m! Γ(m+nu+1))] (x/2)^{2m+nu}

    参数:
        nu : float, Bessel 阶数（非负）
        x  : float or ndarray, 自变量

    返回:
        J_nu(x) 的值
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("x 必须非负")
    if nu < 0:
        raise ValueError("nu 必须非负")

    # 对于小参数使用级数展开，大参数调用 scipy
    result = np.empty_like(x, dtype=float)
    small_mask = x < 20.0
    if np.any(small_mask):
        xs = x[small_mask]
        s = np.zeros_like(xs)
        for m in range(0, 30):
            term = ((-1.0) ** m) / (gamma(m + 1) * gamma(m + nu + 1)) * (xs / 2.0) ** (2 * m + nu)
            s += term
        result[small_mask] = s
    if np.any(~small_mask):
        result[~small_mask] = jv(nu, x[~small_mask])
    return result


def modified_bessel_clot_profile(r, r0, D_fibrin, k_poly, nu=0.5):
    """
    使用修正 Bessel 函数描述血栓内部纤维蛋白浓度径向分布。

    物理模型：在柱坐标下，稳态反应-扩散方程
        D_fibrin * (1/r) d/dr(r dC/dr) - k_poly * C = 0
    的通解为
        C(r) = A I_ν(α r) + B K_ν(α r),   α = sqrt(k_poly / D_fibrin)

    在 clot 中心 (r→0) 处要求有界，故 B=0；
    在 clot 边缘 (r=r0) 处浓度为 C_edge。

    参数:
        r         : ndarray, 径向坐标 (μm)
        r0        : float, clot 外半径 (μm)
        D_fibrin  : float, 纤维蛋白有效扩散系数 (μm²/s)
        k_poly    : float, 聚合反应速率 (1/s)
        nu        : float, 柱对称修正阶数

    返回:
        C(r) / C_edge 的归一化浓度分布
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("径向坐标 r 必须非负")
    if r0 <= 0 or D_fibrin <= 0 or k_poly <= 0:
        raise ValueError("r0, D_fibrin, k_poly 必须为正")

    alpha = np.sqrt(k_poly / D_fibrin)
    # 为避免在 r=0 处 K_ν 发散，只保留 I_ν 项
    # 归一化条件：C(r0) = 1
    denom = iv(nu, alpha * r0)
    if abs(denom) < 1e-300:
        return np.ones_like(r)
    profile = iv(nu, alpha * r) / denom
    # 边界处理：r > r0 时指数衰减
    profile = np.where(r > r0, np.exp(-alpha * (r - r0)), profile)
    return profile


def bessel_jx_values_table():
    """
    基于 079_besselj 的数值表，提供非整数阶 Bessel 函数测试数据。
    用于验证特殊函数计算的数值精度。
    """
    fx_vec = np.array([
        0.3544507442114011, 0.6713967071418031, 0.5130161365618278,
        0.3020049060623657, 0.06500818287737578, -0.3421679847981618,
        -0.1372637357550505, 0.1628807638550299, 0.2402978391234270,
        0.4912937786871623, -0.1696513061447408, 0.1979824927558931,
        -0.1094768729883180, 0.04949681022847794, 0.2239245314689158,
        0.2403772011113174, 0.1966584835818184, 0.02303721950962553,
        0.3314145508558904, 0.5461734240402840, -0.2616584152094124,
        0.1296035513791289, -0.1117432171933552, 0.03142623570527935,
        0.1717922192746527, 0.3126634069544786, 0.1340289119304364,
        0.06235967135106445
    ], dtype=float)
    nu_vec = np.array([
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        2.5, 2.5, 2.5, 2.5, 2.5,
        1.25, 1.25, 1.25, 1.25, 1.25,
        2.75, 2.75, 2.75, 2.75, 2.75
    ], dtype=float)
    x_vec = np.array([
        0.2, 1.0, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0,
        1.0, 2.0, 5.0, 10.0, 50.0,
        1.0, 2.0, 5.0, 10.0, 50.0,
        1.0, 2.0, 5.0, 10.0, 50.0,
        1.0, 2.0, 5.0, 10.0, 50.0
    ], dtype=float)
    return nu_vec, x_vec, fx_vec


def verify_bessel_accuracy():
    """
    验证 Bessel 函数计算的精度。
    返回最大绝对误差。
    """
    nu_vec, x_vec, fx_vec = bessel_jx_values_table()
    max_err = 0.0
    for nu, x, expected in zip(nu_vec, x_vec, fx_vec):
        computed = bessel_jx_fractional(nu, x)
        err = abs(computed - expected)
        max_err = max(max_err, err)
    return max_err


if __name__ == "__main__":
    err = verify_bessel_accuracy()
    print(f"Bessel 函数最大验证误差: {err:.3e}")

    r = np.linspace(0, 50, 200)
    prof = modified_bessel_clot_profile(r, r0=30.0, D_fibrin=2.5e-2, k_poly=1.2e-3)
    print(f"Clot profile at r=0: {prof[0]:.6f}, at r=r0: {prof[-1]:.6f}")
