"""
spherical_harmonics.py
球谐函数展开与高维球面分析
融合原项目: 1132_spherical_harmonic

核心科学思想:
在高维流形降维中，数据投影到单位球面 S^{d-1} 上后，
利用球谐函数对球面分布进行谱展开，提取角度特征。

数学模型:
球谐函数 Y_l^m(θ, φ) 是球坐标系下Laplace方程的角向解，
满足正交归一性:
    ∫_S^2 Y_l^m(θ,φ) Y_{l'}^{m'*}(θ,φ) dΩ = δ_{ll'} δ_{mm'}

展开式:
    f(θ, φ) = Σ_{l=0}^{∞} Σ_{m=-l}^{l} c_l^m Y_l^m(θ, φ)

其中系数:
    c_l^m = ∫_S^2 f(θ,φ) Y_l^{m*}(θ,φ) dΩ

归一化关联Legendre多项式:
    P_l^m(x) = (-1)^m sqrt((2l+1)(l-m)! / (2(l+m)!)) (1-x²)^{m/2} d^m/dx^m P_l(x)
"""

import numpy as np
from typing import Tuple, List
from math import factorial, sqrt


def legendre_polynomial(l: int, x: np.ndarray) -> np.ndarray:
    """
    计算Legendre多项式 P_l(x) 使用递推关系:
        (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)
    """
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x.copy()
    p_prev2 = np.ones_like(x)
    p_prev1 = x.copy()
    for n in range(1, l):
        p_curr = ((2.0 * n + 1.0) * x * p_prev1 - n * p_prev2) / (n + 1.0)
        p_prev2 = p_prev1
        p_prev1 = p_curr
    return p_prev1


def associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    计算归一化关联Legendre多项式 P_l^m(x)
    使用递推关系保证数值稳定性
    """
    m_abs = abs(m)
    if m_abs > l:
        return np.zeros_like(x)
    # 计算非归一化的关联Legendre多项式
    # 首先计算 P_m^m(x)
    p = np.ones_like(x)
    if m_abs > 0:
        p = (-1.0) ** m_abs * (1.0 - x ** 2) ** (m_abs / 2.0)
        # 乘积因子
        for k in range(1, m_abs + 1):
            p *= (2.0 * k - 1.0)
    # 递推到 P_l^m
    if l == m_abs:
        p_lm = p
    else:
        p_lm_prev = p
        # P_{m+1}^m
        p_lm = x * (2.0 * m_abs + 1.0) * p_lm_prev
        for n in range(m_abs + 1, l):
            p_lm_next = ((2.0 * n + 1.0) * x * p_lm - (n + m_abs) * p_lm_prev) / (n - m_abs + 1.0)
            p_lm_prev = p_lm
            p_lm = p_lm_next
    # 归一化因子
    norm = sqrt((2.0 * l + 1.0) * factorial(l - m_abs) / (4.0 * np.pi * factorial(l + m_abs)))
    return norm * p_lm


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算实球谐函数
    Y_l^m(θ, φ) = P_l^m(cos θ) * Φ_m(φ)
    其中 Φ_m(φ) = cos(mφ) (m>0), sin(|m|φ) (m<0), 1 (m=0)
    返回 (Y, phase_factor)
    """
    x = np.cos(theta)
    plm = associated_legendre(l, abs(m), x)
    if m > 0:
        y = plm * np.cos(m * phi)
    elif m < 0:
        y = plm * np.sin(abs(m) * phi)
    else:
        y = plm
    return y, plm


def spherical_harmonics_expansion(values: np.ndarray, theta: np.ndarray,
                                   phi: np.ndarray, l_max: int = 4) -> dict:
    """
    球谐函数展开，计算展开系数 c_l^m
    使用离散点上的蒙特卡洛近似积分
    在球面上: dΩ = sin(θ) dθ dφ
    """
    coefficients = {}
    # 球面总面积 = 4π
    n_points = len(values)
    d_omega = 4.0 * np.pi / n_points  # 平均面积元
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y_lm, _ = spherical_harmonic(l, m, theta, phi)
            # 离散积分: Σ f(θ_i, φ_i) Y_l^m(θ_i, φ_i) sin(θ_i) ΔΩ
            integrand = values * y_lm * np.sin(theta)
            c_lm = np.sum(integrand) * d_omega
            coefficients[(l, m)] = c_lm
    return coefficients


def project_to_sphere(data: np.ndarray) -> np.ndarray:
    """
    将数据投影到单位球面 S^{D-1}
    x -> x / ||x||
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return data / norms


def spherical_coordinates(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将D维数据转换为球坐标 (仅适用于D=3)
    返回 (theta, phi)
    """
    D = data.shape[1]
    if D != 3:
        # 对于高维，先PCA到3维再映射到球面
        from linear_algebra_core import jacobi_eigenvalue
        cov = np.cov(data.T)
        eigvals, eigvecs = jacobi_eigenvalue(cov)
        data_3d = data @ eigvecs[:, :3]
        data = data_3d
    r = np.linalg.norm(data, axis=1)
    r = np.where(r < 1e-15, 1.0, r)
    theta = np.arccos(np.clip(data[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(data[:, 1], data[:, 0])
    return theta, phi


def high_dim_spherical_harmonics_spectrum(data: np.ndarray, l_max: int = 4) -> np.ndarray:
    """
    高维数据的球谐谱分析
    1. 数据中心化并投影到球面
    2. 使用PCA降到3维
    3. 计算球谐展开系数
    4. 返回谱能量分布
    """
    data_centered = data - np.mean(data, axis=0)
    data_sphere = project_to_sphere(data_centered)
    theta, phi = spherical_coordinates(data_sphere)
    # 使用数据密度作为展开函数
    values = np.ones(len(data))
    coeffs = spherical_harmonics_expansion(values, theta, phi, l_max)
    # 计算每个l的能量
    spectrum = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        energy = 0.0
        for m in range(-l, l + 1):
            energy += coeffs[(l, m)] ** 2
        spectrum[l] = energy
    return spectrum


def reconstruct_from_harmonics(coefficients: dict, theta: np.ndarray,
                                phi: np.ndarray, l_max: int = 4) -> np.ndarray:
    """
    从球谐系数重构函数
    f(θ, φ) ≈ Σ_{l=0}^{l_max} Σ_{m=-l}^{l} c_l^m Y_l^m(θ, φ)
    """
    result = np.zeros_like(theta)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y_lm, _ = spherical_harmonic(l, m, theta, phi)
            result += coefficients[(l, m)] * y_lm
    return result
