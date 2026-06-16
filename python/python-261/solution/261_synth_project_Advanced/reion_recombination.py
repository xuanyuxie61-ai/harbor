"""
reion_recombination.py
======================
复合动力学与延迟反馈

本模块实现再电离模拟中的复合物理, 重点包括:

  1. 温度依赖的 Case-B 复合系数 (氢、氦)
  2. 非均匀密度下的 clumping factor 演化
  3. Mackey-Glass 型延迟复合反馈
  4. 多物种 (H, HeI, HeII) 动力学

Mackey-Glass 延迟方程在再电离中的物理对应:
    dxHII / dt = Gamma_pei * (1 - xHII)
               - alpha_B(T) * n_H * C * xHII * n_e(t - tau_delay)

其中 tau_delay 为小密度区复合时标 (~ 10^8 yr), 体现复合过程对过去
电离状态的依赖. 这与 Mackey-Glass 方程在形式上完全一致:
    dy/dt = beta * y(t-tau) / (1 + y(t-tau)^n) - gamma * y(t)

对应种子项目:
  - 707_mackey_glass_dde (延迟微分方程 → 延迟复合)
  - 1138_santiago-schnell (Ehrlich occupancy time ODE → 复合物种演化)
  - 1025_USTBifrt_Pyrolysis-of-Coal (物种计数 → 多物种电离追踪)
"""

import numpy as np
from reion_cosmology import (
    k_B_cgs, m_H_cgs, m_e_cgs, e_charge, hbar, E_ion_H, E_ion_He,
    alpha_B_H, alpha_B_He, Y_He, nH_0, Hubble_parameter,
)


def alpha_B_hydrogen(T):
    """氢 Case-B 复合系数 [cm^3/s] (温度依赖).

    Hummer (1994) 拟合公式:
        alpha_B(T) = 10^{-13} * a4 * (T/10^4 K)^(-eta)
                   * (1 + (T/10^4 K)^beta)^{-delta}

    简化拟合 (Hui & Gnedin 1997):
        alpha_B(T) = 2.753e-14 * (T/10^4)^{-0.816}
                   + 1.269e-14 * (T/10^4)^{-0.616} * exp(-T_4)

    Parameters
    ----------
    T : float or array
        气体温度 [K]

    Returns
    -------
    alpha : float or array
        Case-B 复合系数 [cm^3/s]
    """
    T = np.maximum(np.asarray(T, dtype=float), 10.0)
    T_4 = T / 1.0e4
    # Hui & Gnedin (1997) 拟合
    alpha = (2.753e-14 * T_4**(-0.816)
             + 1.269e-14 * T_4**(-0.616) * np.exp(-3.0 / T_4))
    return np.maximum(alpha, 1.0e-20)


def alpha_B_helium(T):
    """氦 Case-B 复合系数 [cm^3/s].

    类似氢, 但系数不同 (Verner & Ferland 1996).
    """
    T = np.maximum(np.asarray(T, dtype=float), 10.0)
    T_4 = T / 1.0e4
    # 氦拟合 (简化)
    alpha = 1.53e-12 * T_4**(-0.699) * (1.0 + T_4**0.407)**(-2.242)
    return np.maximum(alpha, 1.0e-20)


def clumping_factor(z, model="constant", C0=3.0, z_ref=6.0, beta_cf=-1.5):
    """Clumping factor 模型 <n_H^2> / <n_H>^2.

    描述小尺度密度涨落对复合率的增强.

    Parameters
    ----------
    z : float
        红移
    model : str
        "constant" : 常数 C0
        "power_law": C0 * ((1+z)/(1+z_ref))^beta
        "redshift_dependent": 基于模拟结果的经验拟合
    C0 : float
        参考 clumping 值
    z_ref : float
        参考红移
    beta_cf : float
        红移幂律指数

    Returns
    -------
    C : float
    """
    if model == "constant":
        return C0
    elif model == "power_law":
        ratio = (1.0 + z) / (1.0 + z_ref)
        return C0 * ratio**beta_cf
    elif model == "redshift_dependent":
        # Pawlik et al. (2009) 拟合
        C_base = 26.29 * ((1.0 + z) / 10.0)**(-1.57)
        return max(C_base, 1.0)
    return C0


def recombination_rate_density(xHII, n_H, T, clumping=3.0, species="H"):
    """复合率密度 [cm^-3 s^-1].

    R_rec = alpha_B(T) * n_H * C * xHII * n_e

    对纯氢: n_e = xHII * n_H
    对氢+氦: n_e = (xHII + y_He * xHeII) * n_H, 其中 y_He = Y_He / (4*(1-Y_He))

    Parameters
    ----------
    xHII : float or array
    n_H : float or array
        氢数密度 [cm^-3]
    T : float
        气体温度 [K]
    clumping : float
    species : str
        "H" 或 "H_He"

    Returns
    -------
    R : float or array
    """
    xHII = np.clip(np.asarray(xHII, dtype=float), 0.0, 1.0)
    n_H = np.asarray(n_H, dtype=float)
    if species == "H":
        n_e = xHII * n_H
        alpha = alpha_B_hydrogen(T)
    else:
        y_He = Y_He / (4.0 * (1.0 - Y_He))
        # 简化: 假设氦已双重电离 (再电离后期合理)
        n_e = (xHII + y_He) * n_H
        alpha = alpha_B_hydrogen(T) + y_He * alpha_B_helium(T)
    return alpha * clumping * xHII * n_e


def mackey_glass_recombination(xHII, xHII_delayed, n_H, T, tau_delay,
                               gamma_MG=0.1, beta_MG=0.2, n_MG=2.0,
                               clumping=3.0):
    """Mackey-Glass 型延迟复合率 [s^-1].

    将经典复合率改造为延迟反馈形式:
        dxHII/dt = Gamma * (1 - xHII)
                 - beta * n_H * xHII_delayed / (1 + xHII_delayed^n)
                 - gamma * xHII

    物理意义: 小密度区域的复合时标很长 (~10^8 yr), 导致当前复合率
    依赖于过去的电离状态 (延迟反馈).

    Parameters
    ----------
    xHII : float or array
        当前电离分数
    xHII_delayed : float or array
        延迟时刻的电离分数
    n_H : float
        氢数密度 [cm^-3]
    T : float
        温度 [K]
    tau_delay : float
        延迟时间 [s]
    gamma_MG : float
        Mackey-Glass 线性衰减率
    beta_MG : float
        非线性复合系数
    n_MG : float
        非线性指数
    clumping : float

    Returns
    -------
    rate : float or array
        净电离率 (正=净电离, 负=净复合) [s^-1]
    """
    xHII = np.clip(np.asarray(xHII, dtype=float), 0.0, 1.0)
    xHII_delayed = np.clip(np.asarray(xHII_delayed, dtype=float), 0.0, 1.0)
    # 电离项 (正比于中性分数)
    ionization = beta_MG * (1.0 - xHII)
    # 延迟复合项 (Mackey-Glass 非线性)
    denom = 1.0 + np.power(np.maximum(xHII_delayed, 1.0e-30), n_MG)
    delayed_recomb = beta_MG * n_H * clumping * xHII_delayed / denom
    # 线性衰减
    linear_loss = gamma_MG * xHII
    # 温度修正
    T_4 = max(T / 1.0e4, 0.01)
    rate = ionization - delayed_recomb * T_4**(-0.7) - linear_loss
    return rate


def delayed_field(field_history, t_arr, t_current, tau_delay):
    """从历史场中提取延迟时刻的值.

    Parameters
    ----------
    field_history : array [N_steps+1, N]
        历史场数据
    t_arr : array [N_steps+1]
        时间数组
    t_current : float
        当前时间
    tau_delay : float
        延迟时间 [s]

    Returns
    -------
    field_delayed : array [N]
    """
    t_target = t_current - tau_delay
    if t_target < t_arr[0]:
        return field_history[0].copy()
    # 线性插值
    idx = np.searchsorted(t_arr, t_target) - 1
    idx = max(0, min(idx, len(t_arr) - 2))
    frac = ((t_target - t_arr[idx])
            / max(t_arr[idx + 1] - t_arr[idx], 1.0e-30))
    frac = np.clip(frac, 0.0, 1.0)
    return (1.0 - frac) * field_history[idx] + frac * field_history[idx + 1]


def multi_species_kinetics(xHII, xHeII, n_H, T, Gamma_HI, Gamma_HeI):
    """多物种电离动力学 (H, HeI, HeII).

    速率方程:
        dxHII/dt = Gamma_HI * (1 - xHII) - alpha_B_H * n_e * xHII
        dxHeII/dt = Gamma_HeI * (1 - xHeII) - alpha_B_He * n_e * xHeII

    其中 n_e = xHII * n_H + xHeII * n_He (电子密度)

    Parameters
    ----------
    xHII, xHeII : float
        氢、氦电离分数
    n_H : float
        氢数密度 [cm^-3]
    T : float
        温度 [K]
    Gamma_HI, Gamma_HeI : float
        光电离率 [s^-1]

    Returns
    -------
    dxHII_dt, dxHeII_dt : float
    """
    y_He = Y_He / (4.0 * (1.0 - Y_He))  # 氦/氢数密度比
    n_He = y_He * n_H
    n_e = xHII * n_H + xHeII * n_He
    n_e = max(n_e, 1.0e-30)

    alpha_H = alpha_B_hydrogen(T)
    alpha_He = alpha_B_helium(T)

    # 氢动力学
    dxHII_dt = Gamma_HI * (1.0 - np.clip(xHII, 0.0, 1.0)) - alpha_H * n_e * np.clip(xHII, 0.0, 1.0)
    # 氦动力学
    dxHeII_dt = Gamma_HeI * (1.0 - np.clip(xHeII, 0.0, 1.0)) - alpha_He * n_e * np.clip(xHeII, 0.0, 1.0)
    return dxHII_dt, dxHeII_dt


def photoheating_rate(Gamma_HI, J_mean, z, xHII):
    """光电加热率 [erg s^-1 cm^-3].

    epsilon_pei = n_HI * int_{nu_0}^{infty} (h nu - h nu_0) * J_nu / (h nu) dnu

    简化: epsilon = Gamma_HI * n_HI * epsilon_avg
    其中 epsilon_avg ~ 2 eV (典型超量能量)

    Parameters
    ----------
    Gamma_HI : float
        光电离率 [s^-1]
    J_mean : float
        平均辐射强度
    z : float
    xHII : float

    Returns
    -------
    epsilon : float [erg s^-1 cm^-3]
    """
    epsilon_avg = 2.0 * 1.602e-12  # 2 eV → erg
    n_HI = nH_0 * (1.0 + z)**3 * (1.0 - np.clip(xHII, 0.0, 1.0))
    return Gamma_HI * n_HI * epsilon_avg


def gas_temperature_evolution(T_gas, xHII, n_H, z, Gamma_HI, Hubble_z):
    """气体温度演化方程 dT/dt.

    dT/dt = (2/3) * epsilon_pei / (n_total * k_B)
          - 2 * H(z) * T  (Hubble 冷却)
          + (2/3) * (alpha_rec * T * xHII * n_e) / k_B  (复合加热)

    简化: 仅考虑光电加热与 Hubble 冷却.

    Parameters
    ----------
    T_gas : float
    xHII : float
    n_H : float
    z : float
    Gamma_HI : float
    Hubble_z : float

    Returns
    -------
    dTdt : float [K/s]
    """
    # Hubble 冷却
    cooling = -2.0 * Hubble_z * T_gas
    # 光电加热
    epsilon = photoheating_rate(Gamma_HI, Gamma_HI, z, xHII)
    n_total = n_H * (1.0 + Y_He / (4.0 * (1.0 - Y_He))) + n_H * xHII
    n_total = max(n_total, 1.0e-30)
    heating = (2.0 / 3.0) * epsilon / (n_total * k_B_cgs)
    return heating + cooling
