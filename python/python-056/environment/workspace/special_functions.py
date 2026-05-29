"""
special_functions.py
================================================================================
特殊函数计算模块 (来源于 036_asa103 项目)
================================================================================
本模块提供潮汐能分析所需的特殊数学函数，核心为 Digamma 函数。
在潮汐势的频谱分析中，Digamma 函数用于处理潮汐分潮的振幅
调制与对数伽马分布相关项。

核心公式:
    Digamma(x) = d/dx [ln Γ(x)] = Γ'(x) / Γ(x)

    小参数近似 (x ≤ 1e-6):
        ψ(x) ≈ -γ - 1/x + (π²/6)·x
        其中 γ = 0.5772156649... 为 Euler-Mascheroni 常数

    大参数近似 (x ≥ 8.5):
        ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - ...
"""

import numpy as np
from typing import Tuple


def digamma(x: float) -> Tuple[float, int]:
    """
    计算 Digamma 函数 ψ(x)。

    参数:
        x: 输入参数，必须 x > 0

    返回:
        (value, ifault)
        value: ψ(x) 的值
        ifault: 错误标志，0=正常，1=x≤0
    """
    if x <= 0.0:
        return 0.0, 1

    euler_mascheroni = 0.57721566490153286060
    value = 0.0

    # 小参数近似
    if x <= 1.0e-6:
        value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
        return value, 0

    # 降阶到 x ≥ 8.5
    while x < 8.5:
        value = value - 1.0 / x
        x = x + 1.0

    # Stirling / de Moivre 展开
    r = 1.0 / x
    value = value + np.log(x) - 0.5 * r
    r2 = r * r
    value -= r2 * (1.0 / 12.0
                   - r2 * (1.0 / 120.0
                           - r2 * (1.0 / 252.0
                                   - r2 * (1.0 / 240.0
                                           - r2 * (1.0 / 132.0)))))
    return value, 0


def digamma_vector(x_arr: np.ndarray) -> np.ndarray:
    """
    向量化 Digamma 计算。

    参数:
        x_arr: 正数数组

    返回:
        ψ(x) 数组
    """
    x_arr = np.asarray(x_arr, dtype=float)
    if np.any(x_arr <= 0):
        raise ValueError("digamma_vector: 所有输入必须大于 0")
    return np.array([digamma(x)[0] for x in x_arr.flat]).reshape(x_arr.shape)


def tidal_digamma_modulation(freq_ratio: float, n_harmonics: int = 6) -> float:
    """
    利用 Digamma 函数计算潮汐分潮的振幅调制因子。

    物理背景:
        在引潮势的频谱展开中，高阶分潮的振幅衰减可用
        Digamma 函数的差分来建模:
            A_n ∝ ψ(n + α) - ψ(n)
        其中 α 为与天体轨道偏心率相关的参数。

    公式:
        M(ν) = Σ_{k=1}^{N} [ψ(k + ν) - ψ(k)] / k²

    参数:
        freq_ratio: 频率比 ν，典型值 0.5~2.0
        n_harmonics: 谐波数量

    返回:
        调制因子 M(ν)
    """
    if freq_ratio <= 0.0:
        raise ValueError("tidal_digamma_modulation: freq_ratio 必须大于 0")
    total = 0.0
    for k in range(1, n_harmonics + 1):
        psi_knu, _ = digamma(k + freq_ratio)
        psi_k, _ = digamma(float(k))
        total += (psi_knu - psi_k) / (k * k)
    return total


def polygamma2(x: float) -> float:
    """
    计算 Trigamma 函数 ψ'(x) = d²/dx² ln Γ(x)。

    公式 (大参数近似):
        ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + ...

    参数:
        x: 输入参数，x > 0

    返回:
        ψ'(x) 的值
    """
    if x <= 0.0:
        raise ValueError("polygamma2: x 必须大于 0")
    if x < 1.0e-3:
        return 1.0 / (x * x) + np.pi * np.pi / 6.0
    r = 1.0 / x
    r2 = r * r
    return r + 0.5 * r2 + r2 * r / 6.0 - r2 * r2 * r / 30.0
