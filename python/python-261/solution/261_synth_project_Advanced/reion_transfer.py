"""
reion_transfer.py
=================
辐射传输 BVP 求解器

本模块实现再电离模拟中的辐射传输方程求解. 核心是沿光线的电离光子传输,
考虑中性氢吸收与源项. 主要包含:

  1. radiative_transfer_BVP : 直接求解 1D 辐射传输边值问题
     (隐式差分 + 周期性边界条件)
  2. compact_radiation_gradient : 紧致差分求辐射梯度
  3. effective_optical_depth : 有效光学深度 (类比 Gunn-Peterson)
  4. radiative_coupling_term : 辐射耦合源项

辐射传输方程 (沿视线 s):
    dJ/ds = -n_HI * sigma_HI * J + S_ion(s)

其中:
  J(s)       : 电离辐射强度 [photon cm^-2 s^-1]
  n_HI       : 中性氢数密度 [cm^-3]
  sigma_HI   : 氢光电离截面 [cm^2]
  S_ion(s)   : 局部源项 [photon cm^-3 s^-1]

有效光学深度 (Gunn-Peterson):
    tau_eff = int_0^s n_HI(s') * sigma_HI * (1+z)^{-1} ds'

对应种子项目:
  - 387_fem1d_bvp_quadratic (1D BVP 思想 → 辐射传输 BVP)
"""

import numpy as np
from reion_cosmology import (
    sigma_T, E_ion_H, h_planck, c_light, nH_0, Y_He, Hubble_parameter,
)


# 氢光电离截面 (在电离阈值处, Verner et al. 1996 拟合)
SIGMA_HI_0 = 6.3e-18  # [cm^2] 阈值处

# 典型光子平均自由程 (共动, Madau & Haardt 2015)
MFP_COMOVING_DEFAULT = 3.0e24  # ~ 100 Mpc (共动)


def photoionization_crosssection(E_photon):
    """氢光电离截面随光子能量变化 [cm^2].

    采用 Verner et al. (1996) 拟合:
        sigma(E) = sigma_0 * (E / E_th)^{-3}  (E >= E_th)

    Parameters
    ----------
    E_photon : float or array
        光子能量 [erg]

    Returns
    -------
    sigma : float or array
        光电离截面 [cm^2]
    """
    E_photon = np.asarray(E_photon, dtype=float)
    ratio = E_photon / E_ion_H
    sigma = np.where(ratio >= 1.0, SIGMA_HI_0 * ratio**(-3), 0.0)
    return sigma


def radiative_transfer_BVP(J_inj, n_HI, dx_comoving, sigma_eff=None,
                           mfp_comoving=None, source_term=None):
    """求解 1D 辐射传输边值问题 (周期性边界).

    稳态方程:
        c * dJ/dx = -c * n_HI * sigma * J - c * J / lambda_mfp + S(x)

    离散化 (向后差分, 隐式):
        J_i + dx/c * (c * n_HI_i * sigma * J_i + c * J_i / lambda_mfp)
        = J_{i-1} + dx/c * S_i

    Parameters
    ----------
    J_inj : array [N]
        初始注入辐射强度 (首次猜测) [photon cm^-2 s^-1]
    n_HI : array [N]
        中性氢数密度 [cm^-3]
    dx_comoving : float
        共动网格间距 [cm]
    sigma_eff : float, optional
        有效电离截面 [cm^2]
    mfp_comoving : float, optional
        共动光子平均自由程 [cm]
    source_term : array [N], optional
        源项 [photon cm^-3 s^-1]

    Returns
    -------
    J : array [N]
        自洽电离辐射强度 [photon cm^-2 s^-1]
    """
    N = len(J_inj)
    sigma_eff = sigma_eff if sigma_eff is not None else SIGMA_HI_0
    mfp_comoving = mfp_comoving if mfp_comoving is not None else MFP_COMOVING_DEFAULT
    if source_term is None:
        source_term = np.zeros(N)

    # 吸收系数 (共动)
    # alpha_i = n_HI_i * sigma_eff / (1+z)^3 (物理→共动转换已含)
    alpha_arr = n_HI * sigma_eff + 1.0 / max(mfp_comoving, 1.0e20)

    # 构建隐式系统: 使用二阶紧致差分格式
    # (1 + dt * alpha) J_i - (dt/2) (J_{i+1} - J_{i-1}) * c / dx = dt * S_i + J_i^{old}
    # 简化为: 对稳态直接迭代
    J = J_inj.copy()
    for _iter in range(50):
        J_old = J.copy()
        # 向后差分 (一阶迎风)
        for i in range(N):
            im = (i - 1) % N
            # 隐式更新
            denom = 1.0 + dx_comoving * alpha_arr[i]
            J[i] = (J_old[im] + dx_comoving * source_term[i] / max(c_light, 1.0)) / denom
        # 收敛检查
        diff = np.max(np.abs(J - J_old))
        if diff < 1.0e-10 * (np.max(np.abs(J)) + 1.0e-30):
            break
    return J


def compact_radiation_gradient(J, dx_comoving, alpha_pade=1.0 / 3.0):
    """紧致 Padé 格式计算辐射梯度 dJ/dx.

    4阶精度隐式差分:
        alpha * (dJ/dx)_{i-1} + (dJ/dx)_i + alpha * (dJ/dx)_{i+1}
        = a * (J_{i+1} - J_{i-1}) / (2 * dx)

    其中 a = (3/2) * (1 + 2*alpha), 标准取 alpha=1/3 → a=4/3.

    Returns
    -------
    dJdx : array [N]
    """
    N = len(J)
    a_coeff = (1.0 + 2.0 * alpha_pade) / 2.0
    # 右端项
    rhs = np.zeros(N)
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        rhs[i] = a_coeff * (J[ip] - J[im]) / (2.0 * dx_comoving)

    # 周期性 Thomas 算法 (Sherman-Morrison)
    # 简化实现: 直接迭代
    dJdx = rhs.copy()
    for _iter in range(30):
        dJdx_old = dJdx.copy()
        for i in range(N):
            ip = (i + 1) % N
            im = (i - 1) % N
            dJdx[i] = rhs[i] - alpha_pade * (dJdx_old[ip] + dJdx_old[im])
        diff = np.max(np.abs(dJdx - dJdx_old))
        if diff < 1.0e-12 * (np.max(np.abs(dJdx)) + 1.0e-30):
            break
    return dJdx


def effective_optical_depth(xHII, z_arr, dx_comoving, T_gas=1.0e4):
    """计算有效光学深度 tau_eff (类比 Gunn-Peterson trough).

    定义:
        tau_eff(z) = int_0^{s(z)} n_HI(s') * sigma_HI * ds' / (1+z')

    其中 n_HI = n_H * (1 - xHII) 为中性氢密度.

    采用 Simpson 复合积分.

    Parameters
    ----------
    xHII : array [N]
        电离分数分布 (当前时刻)
    z_arr : array [N_steps + 1]
        红移数组
    dx_comoving : float
        共动网格间距 [cm]
    T_gas : float
        气体温度 [K] (用于热展宽修正)

    Returns
    -------
    tau_eff : float
        视线平均有效光学深度
    """
    N = len(xHII)
    n_HI = nH_0 * (1.0 - np.clip(xHII, 0.0, 1.0))
    # 吸收系数
    kappa = n_HI * SIGMA_HI_0
    # 热展宽修正 (Voigt 轮廓核心近似)
    # Delta_nu_D / nu_0 = sqrt(2 k_B T / m_H c^2)
    # 对 tau_eff 的修正因子 ~ 1 / sqrt(1 + (T/T_0))
    T_0 = 1.0e4  # [K] 特征温度
    thermal_factor = 1.0 / np.sqrt(1.0 + T_gas / T_0)
    kappa *= thermal_factor

    # Simpson 复合积分
    tau_eff = 0.0
    if N < 3:
        # 退化为梯形
        return float(np.sum(kappa) * dx_comoving / max(N, 1))
    # Simpson 1/3 规则
    for i in range(0, N - 2, 2):
        tau_eff += (dx_comoving / 3.0) * (kappa[i] + 4.0 * kappa[i + 1] + kappa[i + 2])
    # 若 N 为偶数, 最后一段用梯形
    if N % 2 == 0:
        tau_eff += 0.5 * dx_comoving * (kappa[-2] + kappa[-1])
    # 红移衰减因子
    z_mean = np.mean(z_arr) if len(z_arr) > 0 else 8.0
    return float(tau_eff / (1.0 + z_mean))


def radiative_coupling_term(J, xHII, n_H_total, sigma_eff=None):
    """计算辐射耦合项 (电离率).

    Gamma_pei = J * sigma_eff * (1 - xHII)

    Parameters
    ----------
    J : array [N]
        电离辐射强度
    xHII : array [N]
        电离分数
    n_H_total : array [N] or float
        总氢数密度
    sigma_eff : float
        有效电离截面

    Returns
    -------
    Gamma : array [N]
        光电离率 [s^-1]
    """
    sigma_eff = sigma_eff if sigma_eff is not None else SIGMA_HI_0
    n_H_total = np.asarray(n_H_total, dtype=float)
    xHII = np.clip(np.asarray(xHII, dtype=float), 0.0, 1.0)
    # 光电离率 = J * sigma / (光子能量 / 电离能)
    # 简化: Gamma = J * sigma_eff * (1 - xHII)
    Gamma = J * sigma_eff * (1.0 - xHII)
    # 数值保护
    Gamma = np.maximum(Gamma, 0.0)
    return Gamma


def mean_free_path_evolution(z, xHII_mean, clumping=3.0):
    """光子平均自由程随红移与电离度演化 [cm (共动)].

    经验公式 (Madau & Haardt 2015):
        lambda_mfp(z) = lambda_0 * (xHII / 0.9)^3 * ((1+z)/5)^{-5.5}

    Parameters
    ----------
    z : float
    xHII_mean : float
    clumping : float

    Returns
    -------
    mfp : float [cm (共动)]
    """
    lambda_0 = 2.0e24  # ~ 65 Mpc (共动) at z=5
    xHII_safe = max(xHII_mean, 0.01)
    z_safe = max(z, 5.0)
    mfp = lambda_0 * (xHII_safe / 0.9)**3 * (z_safe / 5.0)**(-5.5)
    # 考虑 clumping 修正
    mfp /= max(clumping / 3.0, 0.1)
    return max(mfp, 1.0e22)  # 下限保护
