"""
special_math.py
特殊数学函数模块

原项目映射: 040_asa121 (trigamma 函数)

在微地震与压裂裂缝网络研究中，特殊函数用于描述裂缝尺寸分布、
应力强度因子的贝叶斯先验分布以及破裂概率的统计模型。

核心公式:
1. Trigamma 函数（二阶对数伽马导数）:
   ψ'(x) = d²/dx² [ln Γ(x)] = Σ_{k=0}^{∞} 1/(x+k)²
   用于裂缝尺寸的先验分布参数估计。

2. 广义超几何应力强度修正因子:
   K_I = σ√(πa) · F(a/W, φ)
   其中 F 为几何修正函数，常借助特殊函数级数展开计算。

3. 裂缝尺寸累积分布（幂律/帕累托型）:
   P(a > A) = (A₀/A)^{D_f},  D_f 为分形维数
   对应的概率密度导数涉及 trigamma 函数在贝叶斯推断中的共轭先验。
"""

import numpy as np
from typing import Tuple


def trigamma(x: float) -> Tuple[float, int]:
    """
    计算 trigamma(x) = ψ'(x) = d² ln Γ(x) / dx²。

    算法基于 Schneider (1978) AS 121，采用递推升角与渐近展开相结合的策略。

    参数:
        x: 自变量，要求 x > 0。

    返回:
        (value, ifault)
        value: trigamma(x) 的近似值。
        ifault: 错误码，0 表示无错误，1 表示 x <= 0。

    公式:
        当 x <= A (A=1e-4) 时，使用小值近似:
            ψ'(x) ≈ 1/x²

        当 x < B (B=5) 时，利用递推关系:
            ψ'(x) = ψ'(x+1) + 1/x²
            逐步将自变量增加至 x >= B。

        当 x >= B 时，使用渐近展开:
            ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - 1/(30x⁹)
                  = y/2 + (1 + y*(B2 + y*(B4 + y*(B6 + y*B8)))) / z
            其中 y = 1/z², z = x, B2=1/6, B4=-1/30, B6=1/42, B8=-1/30。
    """
    a = 1.0e-4
    b = 5.0
    b2 = 0.1666666667
    b4 = -0.03333333333
    b6 = 0.02380952381
    b8 = -0.03333333333

    if x <= 0.0:
        return 0.0, 1

    ifault = 0
    z = x

    # 小值近似
    if x <= a:
        return 1.0 / (x * x), ifault

    # 递推升角
    value = 0.0
    while z < b:
        value += 1.0 / (z * z)
        z += 1.0

    # 渐近展开
    y = 1.0 / (z * z)
    value += 0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8)))) / z

    return value, ifault


def trigamma_array(x_arr: np.ndarray) -> np.ndarray:
    """
    对数组逐元素计算 trigamma，返回 NaN 表示输入不合法。
    """
    out = np.empty_like(x_arr, dtype=float)
    for i in range(x_arr.size):
        val, flt = trigamma(float(x_arr.flat[i]))
        out.flat[i] = val if flt == 0 else np.nan
    return out


def fracture_size_pdf(a: np.ndarray, a_min: float, D_f: float) -> np.ndarray:
    """
    裂缝尺寸的概率密度函数（截断幂律分布）。

    公式:
        p(a) = (D_f / a_min) * (a / a_min)^{-(D_f + 1)},  a >= a_min

    参数:
        a: 裂缝尺寸数组 (m)。
        a_min: 最小可解析裂缝尺寸 (m)。
        D_f: 分形维数 (1 < D_f < 3)。

    返回:
        概率密度数组。
    """
    a = np.asarray(a, dtype=float)
    if a_min <= 0:
        raise ValueError("a_min 必须为正数")
    if not (1.0 < D_f < 3.0):
        raise ValueError("分形维数 D_f 应在 (1, 3) 区间")

    pdf = np.zeros_like(a)
    mask = a >= a_min
    pdf[mask] = (D_f / a_min) * (a[mask] / a_min) ** (-(D_f + 1.0))
    return pdf


def stress_intensity_factor(sigma: float, a: float, geometry_factor: float = 1.0) -> float:
    """
    计算 I 型应力强度因子 K_I。

    公式（Griffith-Irwin）:
        K_I = σ * sqrt(π a) * F_geom

    参数:
        sigma: 远场拉应力 (Pa)。
        a: 裂缝半长度 (m)。
        geometry_factor: 几何修正因子，默认为 1.0（无限大板中心裂纹）。

    返回:
        K_I (Pa·m^{1/2})。
    """
    if sigma < 0 or a < 0:
        return 0.0
    return sigma * np.sqrt(np.pi * a) * geometry_factor


def log_likelihood_fracture_size(a_obs: np.ndarray, a_min: float, D_f: float) -> float:
    """
    给定观测裂缝尺寸，计算幂律分布的对数似然。

    公式:
        ln L = N ln(D_f / a_min) - (D_f + 1) Σ ln(a_i / a_min)

    参数:
        a_obs: 观测裂缝尺寸数组。
        a_min: 截断下限。
        D_f: 分形维数。

    返回:
        对数似然值。
    """
    a_obs = np.asarray(a_obs, dtype=float)
    if np.any(a_obs < a_min):
        return -np.inf
    N = a_obs.size
    return N * np.log(D_f / a_min) - (D_f + 1.0) * np.sum(np.log(a_obs / a_min))
