"""
exact_benchmark.py
==================
基于精确解析解的数值求解器验证模块。

核心算法源自 doughnut_exact (Project 316)，并改造用于
验证火焰面方程数值求解器的精度。

原始 Doughnut ODE 的精确解：
    y₁(t) = [2a cos(mt) - 2b sin(mt)] / Δ(t)
    y₂(t) = [2a sin(mt) + 2b cos(mt)] / Δ(t)
    y₃(t) = [2c cos(nt) + (2-δ) sin(nt)] / Δ(t)

其中 Δ(t) = δ - 2c sin(nt) + (2-δ) cos(nt)，
δ = 1 + a² + b² + c²。

在燃烧科学中，我们构造一个具有解析解的"测试火焰面方程"：

    d²T/dZ² = f(Z),  Z ∈ [0, 1]

选取一个已知解析解 T_exact(Z) = T_0 + ΔT * sin(π Z)，
则源项为：
    f(Z) = -π² ΔT sin(π Z)

对于更复杂的非线性火焰面方程，我们使用渐近解作为基准：

    T_asymptotic(Z) = T_ox + (T_ad - T_ox) * exp[ - (Z - Z_st)² / (2 σ²) ]

其中 σ² = χ_st / (2 ω̇_max) 为火焰面厚度参数。

误差度量：
    L² 误差:  ||e||₂ = sqrt( ∫ e² dZ )
    L∞ 误差:  ||e||_∞ = max |e|
    H¹ 半范数: |e|₁ = sqrt( ∫ (e')² dZ )
"""

import numpy as np


def doughnut_exact_solution(t, m=3.0, n=5.0, a=1.0, b=1.0, c=3.0):
    """
    Doughnut ODE 的精确解（用于验证 ODE 积分器）。

    Parameters
    ----------
    t : float or ndarray
        时间变量。
    m, n : float
        频率参数。
    a, b, c : float
        初始条件参数。

    Returns
    -------
    y : ndarray
        精确解 [y1, y2, y3]。
    """
    t = np.atleast_1d(t)
    delta = 1.0 + a ** 2 + b ** 2 + c ** 2

    denom = delta - 2.0 * c * np.sin(n * t) + (2.0 - delta) * np.cos(n * t)
    denom = np.where(np.abs(denom) < 1.0e-12, 1.0e-12, denom)

    y1 = (2.0 * a * np.cos(m * t) - 2.0 * b * np.sin(m * t)) / denom
    y2 = (2.0 * a * np.sin(m * t) + 2.0 * b * np.cos(m * t)) / denom
    y3 = (2.0 * c * np.cos(n * t) + (2.0 - delta) * np.sin(n * t)) / denom

    return np.column_stack([y1, y2, y3])


def manufactured_solution_temperature(Z, T_ox=300.0, T_ad=2226.0):
    """
    构造用于验证的精确温度分布（Manufactured Solution）。

    公式：
        T_exact(Z) = T_ox + (T_ad - T_ox) * sin(π Z / 2)

    Parameters
    ----------
    Z : ndarray
        混合分数坐标，[0, 1]。
    T_ox, T_ad : float
        边界温度和绝热温度。

    Returns
    -------
    T_exact : ndarray
        精确温度分布。
    d2T_dZ2 : ndarray
        二阶导数（作为源项）。
    """
    Z = np.clip(Z, 0.0, 1.0)
    T_exact = T_ox + (T_ad - T_ox) * np.sin(np.pi * Z / 2.0)
    d2T_dZ2 = -(np.pi / 2.0) ** 2 * (T_ad - T_ox) * np.sin(np.pi * Z / 2.0)
    return T_exact, d2T_dZ2


def gaussian_flamelet_solution(Z, Z_st, T_ox, T_ad, chi_st, omega_max):
    """
    高斯型火焰面渐近解析解。

    公式：
        T(Z) = T_ox + (T_ad - T_ox) * exp[ -(Z - Z_st)² / (2 σ²) ]

    其中 σ² = χ_st / (2 ω_max) 为火焰厚度参数。

    Parameters
    ----------
    Z : ndarray
        混合分数坐标。
    Z_st : float
        化学计量混合分数。
    T_ox, T_ad : float
        氧化剂温度和绝热温度。
    chi_st : float
        化学计量标量耗散率。
    omega_max : float
        最大反应速率。

    Returns
    -------
    T_approx : ndarray
        近似温度分布。
    flame_thickness : float
        火焰厚度 σ。
    """
    Z = np.clip(Z, 0.0, 1.0)
    sigma_sq = chi_st / (2.0 * max(omega_max, 1.0e-12))
    sigma = np.sqrt(sigma_sq)

    exponent = -((Z - Z_st) ** 2) / (2.0 * sigma_sq)
    exponent = np.clip(exponent, -700.0, 0.0)

    T_approx = T_ox + (T_ad - T_ox) * np.exp(exponent)
    return T_approx, sigma


def compute_errors(numerical, exact, Z_nodes):
    """
    计算数值解与精确解之间的误差。

    Parameters
    ----------
    numerical : ndarray
        数值解。
    exact : ndarray
        精确解。
    Z_nodes : ndarray
        空间节点。

    Returns
    -------
    errors : dict
        包含 L², L∞, H¹ 半范数误差的字典。
    """
    e = numerical - exact
    dZ = np.diff(Z_nodes)

    # L² 误差（梯形法则）
    e_sq = e ** 2
    L2_sq = np.trapezoid(e_sq, Z_nodes)
    L2_error = np.sqrt(L2_sq)

    # L∞ 误差
    Linf_error = np.max(np.abs(e))

    # H¹ 半范数误差
    de_dZ = np.zeros_like(e)
    de_dZ[1:-1] = (e[2:] - e[:-2]) / (Z_nodes[2:] - Z_nodes[:-2])
    de_dZ[0] = (e[1] - e[0]) / (Z_nodes[1] - Z_nodes[0])
    de_dZ[-1] = (e[-1] - e[-2]) / (Z_nodes[-1] - Z_nodes[-2])

    H1_semi_sq = np.trapezoid(de_dZ ** 2, Z_nodes)
    H1_semi_error = np.sqrt(H1_semi_sq)

    errors = {
        'L2_error': L2_error,
        'Linf_error': Linf_error,
        'H1_semi_error': H1_semi_error,
        'relative_L2': L2_error / (np.sqrt(np.trapezoid(exact ** 2, Z_nodes)) + 1.0e-12),
    }

    return errors
