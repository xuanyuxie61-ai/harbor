"""
spherical_harmonics.py
================================================================================
球谐函数模块 —— 基于种子项目 1132_spherical_harmonic

在行星边界层大涡模拟（PBL-LES）中，水平方向通常采用谱方法或球谐展开。
本模块提供归一化连带 Legendre 函数与球谐函数 Y_l^m(θ,φ) 的计算，
用于水平速度场的谱展开与涡度-散度分解。

核心物理公式
--------------------------------------------------------------------------------
球谐函数定义为：
    Y_l^m(θ,φ) = N_l^m  P_l^{|m|}(cos θ)  e^{i m φ}

其中归一化因子：
    N_l^m = sqrt( (2l+1)/(4π) * (l-|m|)! / (l+|m|)! )

对于大气科学中的地转流函数 ψ 与速度势 χ，水平速度可分解为：
    u = - (1/a) ∂ψ/∂φ  -  (1/a sinθ) ∂(χ sinθ)/∂θ
    v =  (1/a sinθ) ∂ψ/∂θ  -  (1/a) ∂χ/∂φ

其中 a 为地球半径。该分解在谱方法中是无散与无旋分量的精确分离。
"""

import numpy as np
from scipy.special import lpmv, factorial


def associated_legendre_normalized(l_max, m_order, x):
    """
    计算 Schmidt 半归一化连带 Legendre 函数 P_l^m(x)。

    参数
    ----------
    l_max : int
        最高阶数，l >= 0
    m_order : int
        阶数 m >= 0
    x : float 或 np.ndarray
        自变量，范围 [-1, 1]

    返回
    -------
    plm : np.ndarray
        形状为 (l_max+1, ...) 的数组，plm[l] = P_l^m(x)
    """
    x = np.atleast_1d(x)
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise ValueError("associated_legendre_normalized: |x| 必须 <= 1")

    plm = np.zeros((l_max + 1,) + x.shape, dtype=np.float64)
    m = abs(m_order)

    # 使用 scipy.special.lpmv 计算非归一化的 P_l^m
    for l in range(m, l_max + 1):
        # lpmv(m, l, x) 计算的是 P_l^m(x)
        val = lpmv(m, l, x)
        # Schmidt 半归一化因子
        if m == 0:
            norm = 1.0
        else:
            # 归一化确保在球面积分中正交
            norm = np.sqrt(2.0 * factorial(l - m) / factorial(l + m))
        plm[l] = norm * val

    return plm


def spherical_harmonic_basis(l, m, theta, phi):
    """
    计算实球谐函数基 Y_l^m(θ,φ) 的实部与虚部。

    参数
    ----------
    l : int
        阶数，0 <= l
    m : int
        次，-l <= m <= l
    theta : float 或 np.ndarray
        极角 [0, π]
    phi : float 或 np.ndarray
        方位角 [0, 2π]

    返回
    -------
    c, s : np.ndarray
        实部 c = Re[Y_l^m]，虚部 s = Im[Y_l^m]
    """
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    if np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("spherical_harmonic_basis: theta 必须在 [0, π] 内")
    if np.any((phi < 0) | (phi > 2 * np.pi)):
        raise ValueError("spherical_harmonic_basis: phi 必须在 [0, 2π] 内")

    m_abs = abs(m)
    x = np.cos(theta)

    # 计算连带 Legendre 函数
    plm = associated_legendre_normalized(l, m_abs, x)

    # 归一化因子 N_l^m = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)
    norm = np.sqrt((2 * l + 1) / (4.0 * np.pi) *
                   factorial(l - m_abs) / factorial(l + m_abs))

    # 实部与虚部
    c = norm * plm[l] * np.cos(m * phi)
    s = norm * plm[l] * np.sin(m * phi)

    if m < 0:
        c = -c
        s = -s

    return c, s


def velocity_spectral_decomposition(psi_coeffs, chi_coeffs, l_max, theta_grid, phi_grid, earth_radius=6.371e6):
    """
    由流函数 ψ 与速度势 χ 的谱系数重构水平速度场 (u, v)。

    公式：
        u = -(1/a sinθ) ∂ψ/∂φ  -  (1/a) ∂χ/∂θ
        v =  (1/a sinθ) ∂χ/∂φ  -  (1/a) ∂ψ/∂θ

    参数
    ----------
    psi_coeffs : dict
        {(l,m): complex} 流函数谱系数
    chi_coeffs : dict
        {(l,m): complex} 速度势谱系数
    l_max : int
        截断波数
    theta_grid : np.ndarray
        极角网格
    phi_grid : np.ndarray
        方位角网格
    earth_radius : float
        地球半径（米）

    返回
    -------
    u, v : np.ndarray
        水平风速分量（m/s）
    """
    theta = np.atleast_1d(theta_grid)
    phi = np.atleast_1d(phi_grid)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    u = np.zeros_like(THETA, dtype=np.float64)
    v = np.zeros_like(THETA, dtype=np.float64)
    a = earth_radius

    sin_theta = np.sin(THETA)
    sin_theta = np.where(np.abs(sin_theta) < 1e-12, 1e-12, sin_theta)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            key = (l, m)
            psi_lm = psi_coeffs.get(key, 0.0)
            chi_lm = chi_coeffs.get(key, 0.0)

            if abs(psi_lm) < 1e-15 and abs(chi_lm) < 1e-15:
                continue

            # 计算 Y_l^m 及其导数（有限差分近似）
            dtheta = 1e-6
            c_p, s_p = spherical_harmonic_basis(l, m, THETA + dtheta, PHI)
            c_m, s_m = spherical_harmonic_basis(l, m, THETA - dtheta, PHI)
            dY_dtheta = (complex(c_p, s_p) - complex(c_m, s_m)) / (2 * dtheta)

            dphi = 1e-6
            c_p, s_p = spherical_harmonic_basis(l, m, THETA, PHI + dphi)
            c_m, s_m = spherical_harmonic_basis(l, m, THETA, PHI - dphi)
            dY_dphi = (complex(c_p, s_p) - complex(c_m, s_m)) / (2 * dphi)

            Y = complex(*spherical_harmonic_basis(l, m, THETA, PHI))

            psi_val = psi_lm * Y
            dpsi_dtheta = psi_lm * dY_dtheta
            dpsi_dphi = psi_lm * dY_dphi

            chi_val = chi_lm * Y
            dchi_dtheta = chi_lm * dY_dtheta
            dchi_dphi = chi_lm * dY_dphi

            # 速度分量
            u += (-1.0 / (a * sin_theta) * dpsi_dphi.imag -
                  1.0 / a * dchi_dtheta.real)
            v += (1.0 / (a * sin_theta) * dchi_dphi.imag -
                  1.0 / a * dpsi_dtheta.real)

    return u, v
