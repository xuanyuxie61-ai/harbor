"""
mie_scattering.py
Mie 散射与相函数展开模块

整合原项目:
  - 661_legendre_polynomial: 勒让德多项式展开

功能:
  1. 勒让德多项式计算 (递推关系)
  2. 散射相函数的勒让德展开
  3. 不对称因子 g 的计算
  4. Henyey-Greenstein 相函数的勒让德系数

核心公式:
  - 勒让德递推:
      P_0(x) = 1
      P_1(x) = x
      P_n(x) = ((2n-1)x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n

  - 相函数展开:
      P(cos Θ) = Σ_{l=0}^{L} (2l+1)/(4π) * a_l * P_l(cos Θ)
    其中 a_l 为展开系数，Θ 为散射角。

  - Henyey-Greenstein 相函数:
      P_HG(Θ) = (1 - g^2) / (4π (1 + g^2 - 2g cos Θ)^{3/2})
    其勒让德系数为 a_l = g^l。
"""

import numpy as np
from math import sqrt, pi


class MieScatteringError(Exception):
    pass


def legendre_polynomial_value(m, n, x):
    """
    计算勒让德多项式 P_0(x) 到 P_n(x) 在 m 个点上的值。

    参数:
      m: 评估点数量
      n: 最高阶数
      x: 形状为 (m,) 的数组，评估点，需在 [-1, 1] 内

    返回:
      v: 形状为 (m, n+1) 的数组，v[:, j] = P_j(x)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] != m:
        raise MieScatteringError("legendre_polynomial_value: x 维度错误")
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise MieScatteringError("legendre_polynomial_value: x 超出 [-1,1]")

    v = np.zeros((m, n + 1), dtype=np.float64)
    v[:, 0] = 1.0

    for j in range(1, n + 1):
        if j == 1:
            vjm1 = np.zeros(m, dtype=np.float64)
        else:
            vjm1 = v[:, j - 2]
        v[:, j] = ((2.0 * j - 1.0) * x * v[:, j - 1] - (j - 1.0) * vjm1) / j

    return v


def legendre_coefficients_hg(g, max_l):
    """
    Henyey-Greenstein 相函数的勒让德展开系数。

    公式:
      a_l = g^l

    参数:
      g: 不对称因子 (-1 < g < 1)
      max_l: 最大展开阶数

    返回:
      长度为 max_l+1 的系数数组
    """
    if not (-1.0 < g < 1.0):
        raise MieScatteringError("legendre_coefficients_hg: g 必须在 (-1,1)")

    l = np.arange(0, max_l + 1, dtype=np.float64)
    return g ** l


def phase_function_hg(cos_theta, g):
    """
    Henyey-Greenstein 相函数。

    公式:
      P(cos Θ) = (1 - g^2) / ( (1 + g^2 - 2g cos Θ)^{3/2} )
    归一化: 0.5 ∫_{-1}^{1} P(μ) dμ = 1
    """
    if not (-1.0 < g < 1.0):
        raise MieScatteringError("phase_function_hg: g 必须在 (-1,1)")
    denom = (1.0 + g ** 2 - 2.0 * g * cos_theta) ** 1.5
    if np.any(denom < 1e-15):
        raise MieScatteringError("phase_function_hg: 分母过小")
    return (1.0 - g ** 2) / denom


def scattering_asymmetry_parameter(g_eff, num_points=200):
    """
    通过数值积分计算 HG 相函数的不对称因子:
      g = 0.5 ∫_{-1}^{1} P(μ) μ dμ

    对于 HG 相函数，解析解就是 g_eff，这里用于验证数值积分。
    """
    mu = np.linspace(-1.0, 1.0, num_points)
    p = phase_function_hg(mu, g_eff)
    integrand = p * mu
    return 0.5 * np.trapezoid(integrand, mu)


def expand_phase_function_legendre(cos_theta, coeffs):
    """
    使用勒让德展开重构相函数。

    公式:
      P(Θ) = Σ_{l=0}^{L} (2l+1) / (4π) * a_l * P_l(cos Θ)

    参数:
      cos_theta: 散射角余弦值，数组
      coeffs: 勒让德系数 a_l

    返回:
      相函数值
    """
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    L = len(coeffs) - 1
    m = cos_theta.shape[0] if cos_theta.ndim > 0 else 1

    if m == 1 and np.isscalar(cos_theta):
        cos_theta = np.array([cos_theta])
        m = 1

    v = legendre_polynomial_value(m, L, cos_theta)
    l_idx = np.arange(0, L + 1)
    prefactor = (2.0 * l_idx + 1.0) / (4.0 * pi)

    phase = np.sum(prefactor * coeffs * v, axis=1)
    return phase


def mie_scattering_cross_section(r, wavelength, m_eff, num_terms=None):
    """
    Mie 散射理论近似计算消光截面。

    使用小参数展开与几何光学近似的混合公式:
      x = 2πr/λ
      若 x << 1: 使用 Rayleigh 近似
      若 x >> 1: 使用几何光学 Q_ext ≈ 2
      若 0.1 < x < 50: 使用 van de Hulst 近似
        Q_ext = 2 - 4 exp(-ρ tan β) cos β / ρ * sin(ρ - β)
                - 4 exp(-ρ tan β) (cos β / ρ)^2 cos(ρ - 2β)
        其中 ρ = 2x (n_r - 1), tan β = n_i / (n_r - 1)

    参数:
      r: 粒径 (μm)
      wavelength: 波长 (μm)
      m_eff: 复折射率 n = n_r + i n_i
      num_terms: Mie 级数项数 (未在简化版中使用)

    返回:
      C_ext: 消光截面 (μm²)
      C_sca: 散射截面 (μm²)
      g: 不对称因子
    """
    x = 2.0 * pi * r / wavelength
    if x <= 0:
        raise MieScatteringError("mie_scattering_cross_section: x 必须为正")

    n_r = np.real(m_eff)
    n_i = np.imag(m_eff)

    if x < 0.1:
        # Rayleigh 区
        ratio = (m_eff ** 2 - 1.0) / (m_eff ** 2 + 2.0)
        q_sca = (8.0 / 3.0) * (x ** 4) * (abs(ratio) ** 2)
        q_abs = 4.0 * x * np.imag(ratio)
        q_ext = q_sca + q_abs
        g = 0.0
    elif x > 50.0:
        # 几何光学区
        q_ext = 2.0
        q_sca = 2.0 * (1.0 + np.exp(-4.0 * x * n_i)) / (1.0 + np.exp(-4.0 * x * n_i))
        # 简化为对称散射
        g = 0.7
    else:
        # van de Hulst 近似
        if abs(n_r - 1.0) < 1e-6:
            q_ext = 2.0
            q_sca = 2.0
            g = 0.5
        else:
            rho = 2.0 * x * (n_r - 1.0)
            if abs(n_r - 1.0) < 1e-12:
                beta = pi / 2.0
            else:
                beta = np.arctan2(n_i, n_r - 1.0)
            tan_b = np.tan(beta)
            if abs(tan_b) < 1e-12:
                exp_term = np.exp(-rho * 1e12)
            else:
                exp_term = np.exp(-rho * tan_b)
            term1 = 4.0 * exp_term * np.cos(beta) / rho * np.sin(rho - beta)
            term2 = 4.0 * exp_term * (np.cos(beta) / rho) ** 2 * np.cos(rho - 2.0 * beta)
            q_ext = 2.0 - term1 - term2
            q_ext = float(np.real(q_ext))
            if q_ext < 0 or q_ext > 4.0:
                q_ext = 2.0
            q_sca = q_ext * 0.9  # 简化假设
            g = 0.6 + 0.2 * (n_r - 1.0)
            g = np.clip(g, -0.9, 0.9)

    area = pi * r ** 2
    c_ext = q_ext * area
    c_sca = q_sca * area
    return float(c_ext), float(c_sca), float(g)
