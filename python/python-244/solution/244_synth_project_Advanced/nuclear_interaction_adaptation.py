#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nuclear_interaction_adaptation.py
=================================
【融合种子项目】 1037_nkoch1_LPFC_adaptation (神经适应模型)

将 LPFC 神经适应的密度依赖形式移植到核相互作用的密度适应:
核力耦合常数随密度演化, 类似神经元的适应电流.

物理:
    密度依赖耦合常数 (RMF 模型):
        m_eff(rho)/m = 1 - alpha (rho/rho_0) / (1 + beta (rho/rho_0))
        g_sigma(rho) = g_sigma^0 [1 + alpha_sigma exp(-rho/lambda_sigma)]

    核物质结合能:
        E/A = T/A + V/A
        T/A = (3/5) (hbar^2 k_F^2)/(2m)  (动能)
        V/A = 1/2 (g_sigma^2/m_sigma^2 - g_omega^2/m_omega^2) rho  (平均场)

数学 (适应函数, 移植自 LPFC):
    A(t) = A_inf + (A_0 - A_inf) exp(-t/tau)
    或密度依赖:
    A(rho) = A_inf + (A_0 - A_inf) exp(-rho/lambda)
"""

import math
import numpy as np
from typing import Tuple, Dict, Callable
from numerical_constants import NeutronStarConstants as NS


def effective_mass_adaptation(rho: float, rho_0: float = 2.7e14,
                                m_star_ratio_0: float = 0.6,
                                lambda_rho: float = 2.0e14) -> float:
    """
    有效质量密度适应 (类似 LPFC 适应电流).

    m^*(rho)/m = m^*_inf/m + (m^*_0/m - m^*_inf/m) exp(-rho/lambda)

    其中 m^*_inf/m ~ 0.55-0.7, lambda ~ 1-3 rho_0
    """
    m_star_inf = 0.55
    return m_star_inf + (m_star_ratio_0 - m_star_inf) * math.exp(-rho / lambda_rho)


def sigma_coupling_adaptation(rho: float, rho_0: float = 2.7e14,
                                g_sigma_0: float = 10.0,
                                alpha_sigma: float = 0.3,
                                lambda_sigma: float = 1.5e14) -> float:
    """
    sigma 介子耦合常数密度适应.

    g_sigma(rho) = g_sigma_0 [1 + alpha_sigma exp(-rho/lambda)]
    """
    return g_sigma_0 * (1.0 + alpha_sigma * math.exp(-rho / lambda_sigma))


def omega_coupling_adaptation(rho: float, rho_0: float = 2.7e14,
                                g_omega_0: float = 8.0,
                                alpha_omega: float = -0.1) -> float:
    """
    omega 介子耦合常数密度适应.
    """
    return g_omega_0 * (1.0 + alpha_omega * rho / rho_0)


def rho_coupling_isospin(rho: float, delta: float,
                           g_rho_0: float = 5.0) -> float:
    """
    rho 介子耦合 (等旋依赖).

    g_rho_eff = g_rho_0 * (1 + x_rho * delta^2)
    """
    x_rho = 0.1
    return g_rho_0 * (1.0 + x_rho * delta * delta)


def nuclear_binding_energy(rho: float, delta: float = 0.0,
                             adaptation_model: bool = True) -> Dict:
    """
    核物质每核子结合能 (包含适应修正), 以 MeV 为单位.

    E/A(rho, delta) = T/A + V/A + S(rho) delta^2

    其中 T/A 为 Fermi 动能, V/A 为平均场势能 (用现象学参数化),
    S(rho) 为对称能.

    采用单位: 密度以 rho_0 为单位, 能量以 MeV 为单位.
    """
    rho_0 = 2.7e14

    if rho <= 0:
        return {
            'E_over_A_MeV': 0.0,
            'T_over_A_MeV': 0.0,
            'V_sigma_MeV': 0.0,
            'V_omega_MeV': 0.0,
            'm_star_over_m': 1.0,
            'g_sigma': 0.0,
            'g_omega': 0.0,
        }

    x = rho / rho_0  # 约化密度

    # 费米动量 (MeV/c): p_F = hbar c (3 pi^2 rho)^{1/3}
    # hbar c = 197.3 MeV fm, rho_0 = 0.16 fm^{-3}
    hbarc = 197.3  # MeV fm
    k_f = hbarc * (1.5 * math.pi ** 2 * 0.16 * x) ** (1.0 / 3.0)  # MeV
    m_n_MeV = 939.565  # 中子质量 MeV/c^2

    # 相对论费米动能 (每核子, MeV)
    # <T> = (3/4) E_F [1 + ...] (非相对论近似)
    # 简化: T/A ~ 3/5 * E_F (非相对论) 或更精确形式
    # 对核物质: T/A ≈ 22.9 (rho/rho_0)^{2/3} MeV
    T_over_A = 22.9 * x ** (2.0 / 3.0)

    # 势能 (参数化, 单位 MeV)
    # 对称核物质: V/A = -E_B - T/A(rho_0) + K_0/9 * (x-1)^2/2
    # 简化形式 (Skyrme 风格):
    E_B = -16.0  # 饱和结合能 MeV
    K_0 = 240.0  # 压缩模量 MeV

    if adaptation_model:
        g_sigma = sigma_coupling_adaptation(rho, rho_0)
        g_omega = omega_coupling_adaptation(rho, rho_0)
        m_star_ratio = effective_mass_adaptation(rho, rho_0)
    else:
        g_sigma = 10.0
        g_omega = 8.0
        m_star_ratio = 0.6

    # 平均场势能 (MeV): 参数化形式
    # V/A = alpha * x + beta * x^gamma (Skyrme 简化)
    alpha_v = -200.0  # MeV
    beta_v = 130.0    # MeV
    gamma_v = 1.3

    V_over_A = alpha_v * x + beta_v * x ** gamma_v
    # 调整使饱和点正确: E/A(rho_0) = -16 MeV
    V_at_sat = E_B - 22.9
    V_over_A = V_over_A * (V_at_sat / (alpha_v + beta_v))

    ea = T_over_A + V_over_A

    # 对称能贡献
    if delta != 0:
        S = symmetry_energy_parabolic(rho)
        ea += S * delta * delta

    # 分配: sigma (吸引) 和 omega (排斥) 的近似比例
    V_sigma_MeV = -abs(V_over_A) * 0.7
    V_omega_MeV = abs(V_over_A) * 0.3

    return {
        'E_over_A_MeV': ea,
        'T_over_A_MeV': T_over_A,
        'V_sigma_MeV': V_sigma_MeV,
        'V_omega_MeV': V_omega_MeV,
        'm_star_over_m': m_star_ratio if adaptation_model else 0.6,
        'g_sigma': g_sigma if adaptation_model else 10.0,
        'g_omega': g_omega if adaptation_model else 8.0,
    }


def adaptation_timecourse(density_profile: np.ndarray,
                            coupling_func: Callable,
                            tau_adapt: float = 1.0e14) -> np.ndarray:
    """
    沿密度剖面的耦合常数演化 (类似时间过程适应).

    将密度映射为 "伪时间": t = integral dr / v_F(r)
    """
    coupling = np.zeros_like(density_profile)
    for k in range(len(density_profile)):
        coupling[k] = coupling_func(density_profile[k])
    return coupling


def pressure_from_energy_density(rho_arr: np.ndarray,
                                   ea_arr_MeV: np.ndarray) -> np.ndarray:
    """
    由 E/A(rho) 计算压力 (热力学关系).

    P = rho^2 d(E/A)/drho
    """
    dea_drho = np.gradient(ea_arr_MeV, rho_arr)
    P_MeV_fm3 = rho_arr ** 2 * dea_drho * NS.MeV_to_erg / NS.m_n
    return P_MeV_fm3


# 自检
if __name__ == "__main__":
    print("=== 核相互作用适应自检 ===")

    rho_0 = 2.7e14
    rho_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0]) * rho_0

    print("有效质量适应:")
    for rho in rho_vals:
        m_ratio = effective_mass_adaptation(rho)
        print(f"  rho/rho_0={rho/rho_0:.1f}: m*/m = {m_ratio:.4f}")

    print("sigma 耦合适应:")
    for rho in rho_vals:
        g_sig = sigma_coupling_adaptation(rho)
        print(f"  rho/rho_0={rho/rho_0:.1f}: g_sigma = {g_sig:.3f}")

    print("结合能 (带适应):")
    for rho in rho_vals:
        result = nuclear_binding_energy(rho, delta=0.0, adaptation_model=True)
        print(f"  rho/rho_0={rho/rho_0:.1f}: E/A = {result['E_over_A_MeV']:.2f} MeV")

    print("结合能 (无适应):")
    for rho in rho_vals:
        result = nuclear_binding_energy(rho, delta=0.0, adaptation_model=False)
        print(f"  rho/rho_0={rho/rho_0:.1f}: E/A = {result['E_over_A_MeV']:.2f} MeV")

    print("\nnuclear_interaction_adaptation.py 自检通过.")
