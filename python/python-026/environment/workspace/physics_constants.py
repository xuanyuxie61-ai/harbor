# -*- coding: utf-8 -*-
"""
physics_constants.py

等离子体物理与激光相互作用核心常数及公式库。

本模块封装了惯性约束聚变（ICF）与高功率激光-等离子体相互作用研究中
所需的全套物理常数、特征尺度计算公式、以及关键的无量纲参数。

核心公式体系：
1. 等离子体频率 (Plasma frequency):
   ω_p = sqrt(n_e * e^2 / (ε_0 * m_e))

2. 德拜长度 (Debye length):
   λ_D = sqrt(ε_0 * k_B * T_e / (n_e * e^2))

3. 等离子体标长 (Plasma skin depth):
   c / ω_p

4. 折射率 (Refractive index in cold plasma):
   η = sqrt(1 - ω_p^2 / ω_0^2)

5. 有质动力势 (Ponderomotive potential):
   U_p = e^2 * E_0^2 / (4 * m_e * ω_0^2)

6. 有质动力 (Ponderomotive force):
   F_p = -∇U_p = - (e^2 / (4 * m_e * ω_0^2)) * ∇|E|^2

7. 电子抖动速度 (Quiver velocity):
   v_osc = e * E_0 / (m_e * ω_0)

8. 激光临界密度 (Critical density):
   n_c = ε_0 * m_e * ω_0^2 / e^2

9. 激光在等离子体中的波数:
   k = (ω_0 / c) * η

10. SRS 增长率 (Rosenbluth-Liu):
    γ_0 = (v_osc / (2c)) * sqrt(ω_p * ω_0)

11. Landau 阻尼率:
    γ_L = sqrt(π/8) * (ω_p / (k^3 * λ_D^3)) * exp(-1/(2*k^2*λ_D^2) - 3/2)

12. 碰撞频率 (Spitzer):
    ν_ei = (Z * n_e * e^4 * lnΛ) / (3 * (2π)^(3/2) * ε_0^2 * m_e^(1/2) * (k_B*T_e)^(3/2))

13. 逆轫致吸收系数:
    κ_ib = (ν_ei / c) * (ω_p^2 / ω_0^2) * (1 / sqrt(1 - ω_p^2/ω_0^2))
"""

import numpy as np

# 基本物理常数 (SI单位制)
E_CHARGE = 1.602176634e-19       # C, 元电荷
E_MASS = 9.1093837015e-31        # kg, 电子静止质量
EPSILON_0 = 8.8541878128e-12     # F/m, 真空介电常数
MU_0 = 4.0 * np.pi * 1.0e-7      # N/A^2, 真空磁导率
C_LIGHT = 299792458.0            # m/s, 真空中光速
K_BOLTZMANN = 1.380649e-23       # J/K, 玻尔兹曼常数
PLANCK_H = 6.62607015e-34        # J·s, 普朗克常数


def plasma_frequency(ne):
    """
    计算等离子体频率 ω_p。

    公式:
        ω_p = sqrt(ne * e^2 / (ε_0 * m_e))

    Parameters
    ----------
    ne : float or ndarray
        电子数密度 [m^{-3}]

    Returns
    -------
    omega_p : float or ndarray
        等离子体角频率 [rad/s]
    """
    raise NotImplementedError("Hole 1: 请实现 plasma_frequency 函数体")


def debye_length(ne, Te):
    """
    计算电子德拜长度 λ_D。

    公式:
        λ_D = sqrt(ε_0 * k_B * T_e / (n_e * e^2))

    Parameters
    ----------
    ne : float or ndarray
        电子数密度 [m^{-3}]
    Te : float or ndarray
        电子温度 [K]

    Returns
    -------
    lambda_d : float or ndarray
        德拜长度 [m]
    """
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    if np.any(ne <= 0):
        raise ValueError("电子密度 ne 必须为正数。")
    if np.any(Te <= 0):
        raise ValueError("电子温度 Te 必须为正数。")
    lambda_d = np.sqrt(EPSILON_0 * K_BOLTZMANN * Te / (ne * E_CHARGE**2))
    return lambda_d


def critical_density(omega0):
    """
    计算激光临界密度 n_c。

    公式:
        n_c = ε_0 * m_e * ω_0^2 / e^2

    Parameters
    ----------
    omega0 : float
        激光角频率 [rad/s]

    Returns
    -------
    nc : float
        临界密度 [m^{-3}]
    """
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    nc = EPSILON_0 * E_MASS * omega0**2 / E_CHARGE**2
    return nc


def refractive_index(ne, omega0):
    """
    计算冷等离子体中的激光折射率 η。

    公式:
        η = sqrt(max(0, 1 - ω_p^2 / ω_0^2))

    边界处理: 当 ne >= n_c 时，η = 0（截止）。

    Parameters
    ----------
    ne : float or ndarray
        电子数密度 [m^{-3}]
    omega0 : float
        激光角频率 [rad/s]

    Returns
    -------
    eta : float or ndarray
        折射率 [无量纲]
    """
    ne = np.asarray(ne, dtype=float)
    omega_p = plasma_frequency(ne)
    ratio = (omega_p / omega0) ** 2
    ratio = np.clip(ratio, 0.0, 1.0)
    eta = np.sqrt(1.0 - ratio)
    return eta


def quiver_velocity(E0, omega0):
    """
    计算电子在激光场中的抖动速度 v_osc。

    公式:
        v_osc = e * E_0 / (m_e * ω_0)

    Parameters
    ----------
    E0 : float
        激光电场振幅 [V/m]
    omega0 : float
        激光角频率 [rad/s]

    Returns
    -------
    v_osc : float
        抖动速度 [m/s]
    """
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    v_osc = E_CHARGE * E0 / (E_MASS * omega0)
    return v_osc


def ponderomotive_potential(E0, omega0):
    """
    计算有质动力势 U_p。

    公式:
        U_p = e^2 * E_0^2 / (4 * m_e * ω_0^2)

    Parameters
    ----------
    E0 : float
        激光电场振幅 [V/m]
    omega0 : float
        激光角频率 [rad/s]

    Returns
    -------
    Up : float
        有质动力势 [J]
    """
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    Up = E_CHARGE**2 * E0**2 / (4.0 * E_MASS * omega0**2)
    return Up


def ponderomotive_force_gradient(E0, omega0, grad_E2):
    """
    计算有质动力 F_p = -∇U_p。

    公式:
        F_p = - (e^2 / (4 * m_e * ω_0^2)) * ∇|E|^2

    Parameters
    ----------
    E0 : float
        参考电场振幅 [V/m]
    omega0 : float
        激光角频率 [rad/s]
    grad_E2 : ndarray
        |E|^2 的空间梯度 [V^2/m^3]

    Returns
    -------
    Fp : ndarray
        有质动力 [N]
    """
    if omega0 <= 0:
        raise ValueError("激光角频率必须为正。")
    coeff = -E_CHARGE**2 / (4.0 * E_MASS * omega0**2)
    Fp = coeff * np.asarray(grad_E2, dtype=float)
    return Fp


def srs_growth_rate(ne, E0, omega0):
    """
    计算受激拉曼散射（SRS）的线性增长率 γ_0。

    Rosenbluth-Liu 公式:
        γ_0 = (v_osc / (2c)) * sqrt(ω_p * ω_0)

    Parameters
    ----------
    ne : float
        电子数密度 [m^{-3}]
    E0 : float
        激光电场振幅 [V/m]
    omega0 : float
        激光角频率 [rad/s]

    Returns
    -------
    gamma0 : float
        SRS 增长率 [rad/s]
    """
    omega_p = plasma_frequency(ne)
    v_osc = quiver_velocity(E0, omega0)
    gamma0 = (v_osc / (2.0 * C_LIGHT)) * np.sqrt(omega_p * omega0)
    return gamma0


def landau_damping_rate(ne, Te, k):
    """
    计算朗道阻尼率 γ_L。

    公式:
        γ_L = sqrt(π/8) * (ω_p / (k^3 * λ_D^3)) * exp(-1/(2*k^2*λ_D^2) - 3/2)

    Parameters
    ----------
    ne : float
        电子数密度 [m^{-3}]
    Te : float
        电子温度 [K]
    k : float
        等离子体波波数 [rad/m]

    Returns
    -------
    gamma_L : float
        朗道阻尼率 [rad/s]
    """
    omega_p = plasma_frequency(ne)
    lambda_d = debye_length(ne, Te)
    if k <= 0 or lambda_d <= 0:
        raise ValueError("k 和 λ_D 必须为正。")
    k_lambda = k * lambda_d
    gamma_L = np.sqrt(np.pi / 8.0) * (omega_p / (k**3 * lambda_d**3)) * \
              np.exp(-1.0 / (2.0 * k_lambda**2) - 1.5)
    return gamma_L


def coulomb_logarithm(ne, Te, Z=1):
    """
    计算库仑对数 ln Λ (Spitzer 近似)。

    公式:
        ln Λ = 23.5 - ln(sqrt(ne) * Z / Te_eV^(3/2))

    Parameters
    ----------
    ne : float
        电子数密度 [m^{-3}]
    Te : float
        电子温度 [K]
    Z : int, optional
        离子电荷数, 默认为 1。

    Returns
    -------
    ln_lambda : float
        库仑对数 [无量纲]
    """
    Te_eV = Te / E_CHARGE  # 转换为 eV
    if ne <= 0 or Te_eV <= 0:
        raise ValueError("ne 和 Te 必须为正。")
    ln_lambda = 23.5 - np.log(np.sqrt(ne) * Z / Te_eV**1.5)
    if ln_lambda < 1.0:
        ln_lambda = 1.0  # 边界保护
    return ln_lambda


def electron_ion_collision_frequency(ne, Te, Z=1):
    """
    计算电子-离子碰撞频率 ν_ei (Spitzer 公式)。

    公式:
        ν_ei = (Z * n_e * e^4 * lnΛ) / (3 * (2π)^(3/2) * ε_0^2 * m_e^(1/2) * (k_B*T_e)^(3/2))

    Parameters
    ----------
    ne : float
        电子数密度 [m^{-3}]
    Te : float
        电子温度 [K]
    Z : int, optional
        离子电荷数, 默认为 1。

    Returns
    -------
    nu_ei : float
        碰撞频率 [rad/s]
    """
    ln_lambda = coulomb_logarithm(ne, Te, Z)
    nu_ei = (Z * ne * E_CHARGE**4 * ln_lambda) / \
            (3.0 * (2.0 * np.pi)**1.5 * EPSILON_0**2 * np.sqrt(E_MASS) * (K_BOLTZMANN * Te)**1.5)
    return nu_ei


def inverse_bremsstrahlung_absorption(ne, Te, omega0, Z=1):
    """
    计算逆轫致吸收系数 κ_ib。

    公式:
        κ_ib = (ν_ei / c) * (ω_p^2 / ω_0^2) * (1 / η)

    边界处理: η -> 0 时，κ_ib 被截断以避免发散。

    Parameters
    ----------
    ne : float
        电子数密度 [m^{-3}]
    Te : float
        电子温度 [K]
    omega0 : float
        激光角频率 [rad/s]
    Z : int, optional
        离子电荷数, 默认为 1。

    Returns
    -------
    kappa_ib : float
        吸收系数 [m^{-1}]
    """
    nu_ei = electron_ion_collision_frequency(ne, Te, Z)
    omega_p = plasma_frequency(ne)
    eta = refractive_index(ne, omega0)
    eta_safe = max(eta, 1e-6)
    kappa_ib = (nu_ei / C_LIGHT) * (omega_p / omega0)**2 * (1.0 / eta_safe)
    if np.isinf(kappa_ib) or np.isnan(kappa_ib):
        kappa_ib = 0.0
    return kappa_ib


def laser_wavenumber(ne, omega0):
    """
    计算激光在等离子体中的局域波数 k。

    公式:
        k = (ω_0 / c) * η

    Parameters
    ----------
    ne : float or ndarray
        电子数密度 [m^{-3}]
    omega0 : float
        激光角频率 [rad/s]

    Returns
    -------
    k : float or ndarray
        波数 [rad/m]
    """
    eta = refractive_index(ne, omega0)
    k = (omega0 / C_LIGHT) * eta
    return k


def laser_intensity_from_E0(E0):
    """
    由电场振幅计算激光强度 I。

    公式:
        I = (1/2) * c * ε_0 * |E_0|^2

    Parameters
    ----------
    E0 : float
        电场振幅 [V/m]

    Returns
    -------
    I : float
        激光强度 [W/m^2]
    """
    I = 0.5 * C_LIGHT * EPSILON_0 * E0**2
    return I


def laser_E0_from_intensity(I):
    """
    由激光强度计算电场振幅 E_0。

    公式:
        E_0 = sqrt(2 * I / (c * ε_0))

    Parameters
    ----------
    I : float
        激光强度 [W/m^2]

    Returns
    -------
    E0 : float
        电场振幅 [V/m]
    """
    if I < 0:
        raise ValueError("激光强度必须为非负数。")
    E0 = np.sqrt(2.0 * I / (C_LIGHT * EPSILON_0))
    return E0
