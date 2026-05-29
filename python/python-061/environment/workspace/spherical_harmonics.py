"""
球谐函数谱展开模块
==================
基于种子项目 990_r8poly 的 Legendre 和 Chebyshev 多项式思想。

核心科学问题：
    气象场在球面上的展开需要正交基函数。球谐函数 Y_l^m(θ, φ) 是
    球面 Laplacian 的本征函数，构成 L²(S²) 的完备正交基：
    
    ∇² Y_l^m = -l(l+1)/R² Y_l^m
    
    其中 l = 0, 1, 2, ... 为角阶数，m = -l, ..., l 为方位阶数。

数学定义：
    Y_l^m(θ, φ) = N_l^m * P_l^m(cos θ) * e^{imφ}
    
    其中 P_l^m(x) 为连带 Legendre 函数，归一化系数：
    
    N_l^m = √((2l+1)/(4π) * (l-m)!/(l+m)!)

气压场展开：
    h(θ, φ) = Σ_{l=0}^{L} Σ_{m=-l}^{l} h_l^m Y_l^m(θ, φ)

谱系数：
    h_l^m = ∫∫ h(θ, φ) * conj(Y_l^m(θ, φ)) * sin(θ) dθ dφ

Truncation: 三角截断 T_L，保留 l ≤ L 的所有模态。
"""

import numpy as np
from scipy.special import lpmv, factorial

# 地球半径常数
EARTH_RADIUS = 6.371e6


def associated_legendre(l, m, x):
    """
    计算连带 Legendre 函数 P_l^m(x)。
    
    基于种子项目 990_r8poly 的多项式递推思想。
    
    递推关系（Ferrers 定义）：
        P_0^0(x) = 1
        P_l^l(x) = -(2l-1) * √(1-x²) * P_{l-1}^{l-1}(x)
        P_l^{l-1}(x) = x * (2l-1) * P_{l-1}^{l-1}(x)
        P_l^m(x) = [(2l-1)*x*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)] / (l-m)
    
    参数:
        l: 角阶数（非负整数）
        m: 方位阶数（|m| ≤ l）
        x: 自变量，范围 [-1, 1]
    
    返回:
        P_l^m(x) 的值
    """
    x = np.atleast_1d(x)
    # 使用 scipy 的 lpmv，注意 lpmv 的输入是 m, l, x
    return lpmv(m, l, x)


def spherical_harmonic_y(l, m, theta, phi):
    """
    计算球谐函数 Y_l^m(θ, φ)。
    
    使用 Condon-Shortley 相位约定的复数球谐函数：
        Y_l^m(θ, φ) = N_l^m * P_l^m(cos θ) * e^{imφ}
    
    参数:
        l: 角阶数
        m: 方位阶数
        theta: 极距 [0, π]
        phi: 经度 [0, 2π]
    
    返回:
        复数 Y_l^m 值
    """
    x = np.cos(theta)
    
    # 归一化系数
    if m >= 0:
        norm_sq = (2.0 * l + 1.0) / (4.0 * np.pi) * factorial(l - m) / factorial(l + m)
    else:
        # 对负m使用关系 Y_l^{-m} = (-1)^m * conj(Y_l^m)
        m_pos = -m
        norm_sq = (2.0 * l + 1.0) / (4.0 * np.pi) * factorial(l - m_pos) / factorial(l + m_pos)
    
    norm = np.sqrt(norm_sq)
    
    # 连带 Legendre 函数
    p_lm = associated_legendre(l, abs(m), x)
    
    # 相位因子
    if m >= 0:
        y_val = norm * p_lm * np.exp(1j * m * phi)
    else:
        y_val = (-1)**abs(m) * norm * p_lm * np.exp(1j * m * phi)
    
    return y_val


def compute_spectral_coefficients_1d(theta, values, L_max=20):
    """
    计算一维（仅依赖 θ）场的球谐展开系数。
    
    对于仅依赖 θ 的场，只有 m=0 的模态非零：
        f(θ) = Σ_{l=0}^{L} f_l * Y_l^0(θ)
    
    由于 Y_l^0(θ) = √((2l+1)/(4π)) * P_l(cos θ)，
    系数可通过高斯求积计算：
        f_l = ∫_0^π f(θ) * Y_l^0(θ) * sin(θ) dθ
    
    参数:
        theta: 极距网格
        values: 场值
        L_max: 最大截断阶数
    
    返回:
        coeffs: 谱系数数组，长度 L_max+1
    """
    n = len(theta)
    coeffs = np.zeros(L_max + 1, dtype=complex)
    
    # 梯形法则积分
    dtheta = np.diff(theta)
    
    for l in range(L_max + 1):
        y_l0 = spherical_harmonic_y(l, 0, theta, 0.0)
        integrand = values * np.conj(y_l0) * np.sin(theta)
        
        # 复合梯形积分
        integral = 0.0
        for i in range(n - 1):
            integral += 0.5 * (integrand[i] + integrand[i + 1]) * dtheta[i]
        
        coeffs[l] = integral
    
    return coeffs


def reconstruct_from_spectral_1d(theta, coeffs):
    """
    从谱系数重构一维场。
    
    参数:
        theta: 极距网格
        coeffs: 谱系数
    
    返回:
        values: 重构场值
    """
    L_max = len(coeffs) - 1
    values = np.zeros(len(theta), dtype=complex)
    
    for l in range(L_max + 1):
        y_l0 = spherical_harmonic_y(l, 0, theta, 0.0)
        values += coeffs[l] * y_l0
    
    return np.real(values)


def spectral_laplacian_1d(coeffs, R=EARTH_RADIUS):
    """
    计算谱系数的 Laplacian。
    
    在谱空间中，Laplacian 是对角的：
        ∇² f_l = -l(l+1)/R² * f_l
    
    参数:
        coeffs: 输入谱系数
        R: 球半径
    
    返回:
        lap_coeffs: Laplacian 的谱系数
    """
    L_max = len(coeffs) - 1
    lap_coeffs = np.zeros_like(coeffs)
    
    for l in range(L_max + 1):
        lap_coeffs[l] = -l * (l + 1.0) / (R**2) * coeffs[l]
    
    return lap_coeffs


def chebyshev_spectral_filter(coeffs, order=4):
    """
    Chebyshev 型谱滤波器（基于 990_r8poly 的 Chebyshev 多项式思想）。
    
    用于抑制高波数噪声的滤波器：
        f_l^{filtered} = f_l / (1 + α * (l/L_max)^{2p})
    
    其中 p 为滤波阶数，α 为滤波强度。
    
    参数:
        coeffs: 谱系数
        order: 滤波阶数
    
    返回:
        filtered: 滤波后的系数
    """
    L_max = len(coeffs) - 1
    if L_max <= 0:
        return coeffs.copy()
    
    alpha = 1.0
    filtered = np.zeros_like(coeffs)
    
    for l in range(L_max + 1):
        damping = 1.0 / (1.0 + alpha * (l / L_max)**(2 * order))
        filtered[l] = coeffs[l] * damping
    
    return filtered



def spectral_variance_spectrum(coeffs):
    """
    计算谱方差分布（能量谱）。
    
    能量谱定义为：
        E(l) = (l(l+1)/R²) * |f_l|²
    
    参数:
        coeffs: 谱系数
    
    返回:
        energy: 每个波数的能量
        wavenumbers: 波数 l
    """
    L_max = len(coeffs) - 1
    energy = np.zeros(L_max + 1)
    wavenumbers = np.arange(L_max + 1)
    
    for l in range(L_max + 1):
        energy[l] = l * (l + 1.0) * np.abs(coeffs[l])**2
    
    return energy, wavenumbers
