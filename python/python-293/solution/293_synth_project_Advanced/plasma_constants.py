"""
plasma_constants.py — 空间等离子体物理常数与特征频率模块

本模块封装空间等离子体(磁层/辐射带/太阳风)中波粒相互作用研究所需的全部
物理常数与无量纲参数。所有量均采用 SI 单位制。

核心物理量:
  - 电子质量 m_e = 9.10938e-31 kg
  - 质子质量 m_p = 1.67262e-27 kg
  - 基本电荷 e   = 1.60218e-19 C
  - 真空介电常数 epsilon_0 = 8.85419e-12 F/m
  - 真空磁导率 mu_0 = 4*pi*1e-7 H/m
  - 光速 c = 2.99792e8 m/s
  - 玻尔兹曼常数 k_B = 1.38065e-23 J/K

等离子体特征尺度:
  - 电子等离子体频率: omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
  - 电子回旋频率:     omega_ce = e * B0 / m_e
  - 电子惯性长度:     d_e = c / omega_pe
  - 电子热速度:       v_th = sqrt(k_B * T_e / m_e)
  - 德拜长度:         lambda_D = v_th / omega_pe
  - 等离子体 beta:    beta = 2 * mu_0 * n_e * k_B * T_e / B0^2
"""

import math
import numpy as np


# ============================================================
# 基本物理常数 (SI)
# ============================================================
ELECTRON_MASS = 9.1093837015e-31       # kg
PROTON_MASS = 1.67262192369e-27        # kg
ELEMENTARY_CHARGE = 1.602176634e-19    # C
EPSILON_0 = 8.8541878128e-12           # F/m
MU_0 = 4.0 * math.pi * 1e-7           # H/m
SPEED_OF_LIGHT = 2.99792458e8          # m/s
K_BOLTZMANN = 1.380649e-23            # J/K
EV_TO_JOULE = ELEMENTARY_CHARGE        # 1 eV = 1.602e-19 J


# ============================================================
# 等离子体特征量计算
# ============================================================
def electron_plasma_frequency(n_e: float) -> float:
    """
    电子等离子体频率 (rad/s):
        omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
    参数: n_e 电子数密度 (m^{-3})
    """
    if n_e < 0.0:
        raise ValueError(f"电子密度不能为负: n_e={n_e}")
    return math.sqrt(n_e * ELEMENTARY_CHARGE**2 / (EPSILON_0 * ELECTRON_MASS))


def electron_cyclotron_frequency(B0: float) -> float:
    """
    电子回旋频率 (rad/s):
        omega_ce = e * B0 / m_e
    参数: B0 背景磁场 (T)
    """
    if B0 < 0.0:
        raise ValueError(f"磁场不能为负: B0={B0}")
    return ELEMENTARY_CHARGE * abs(B0) / ELECTRON_MASS


def proton_cyclotron_frequency(B0: float) -> float:
    """质子回旋频率: omega_ci = e * B0 / m_p"""
    return ELEMENTARY_CHARGE * abs(B0) / PROTON_MASS


def electron_inertial_length(n_e: float) -> float:
    """
    电子惯性长度 (m):
        d_e = c / omega_pe
    """
    omega_pe = electron_plasma_frequency(n_e)
    if omega_pe < 1e-30:
        raise ValueError("等离子体频率过低")
    return SPEED_OF_LIGHT / omega_pe


def debye_length(T_e_eV: float, n_e: float) -> float:
    """
    德拜长度 (m):
        lambda_D = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))
    """
    T_e_J = T_e_eV * EV_TO_JOULE
    if n_e <= 0.0 or T_e_J < 0.0:
        raise ValueError(f"无效参数: n_e={n_e}, T_e={T_e_eV}")
    return math.sqrt(EPSILON_0 * T_e_J / (n_e * ELEMENTARY_CHARGE**2))


def thermal_velocity(T_e_eV: float) -> float:
    """电子热速度 (m/s): v_th = sqrt(k_B * T_e / m_e)"""
    T_e_J = max(T_e_eV * EV_TO_JOULE, 0.0)
    return math.sqrt(T_e_J / ELECTRON_MASS)


def plasma_beta(n_e: float, T_e_eV: float, B0: float) -> float:
    """
    等离子体 beta (热压/磁压比):
        beta = 2 * mu_0 * n_e * k_B * T_e / B0^2
    """
    if B0 < 1e-30:
        raise ValueError("磁场趋于零, beta 发散")
    T_e_J = T_e_eV * EV_TO_JOULE
    return 2.0 * MU_0 * n_e * T_e_J / (B0**2)


def whistler_dispersion(omega_ce: float, omega_pe: float,
                         k_parallel: float, theta_deg: float = 0.0) -> complex:
    """
    Whistler 模色散关系 (冷等离子体近似):
        omega = omega_ce * cos(theta) * (k*c/omega_pe)^2 / (1 + (k*c/omega_pe)^2)
    低频近似 (omega << omega_ce) 成立.
    """
    theta_rad = math.radians(theta_deg)
    cos_theta = math.cos(theta_rad)
    kc_wp = k_parallel * SPEED_OF_LIGHT / omega_pe
    omega_real = omega_ce * cos_theta * kc_wp**2 / (1.0 + kc_wp**2)
    return complex(omega_real, 0.0)


def cyclotron_resonance_energy(omega_wave: float, omega_ce: float,
                                k_parallel: float, n_harmonic: int = -1) -> float:
    """
    回旋共振能量 (eV):
        共振条件: omega - k_para * v_para = n * omega_ce
        E_res = 0.5 * m_e * ((omega - n*omega_ce)/k_para)^2 / e
    """
    if abs(k_parallel) < 1e-30:
        return float('inf')
    v_res = (omega_wave - n_harmonic * omega_ce) / k_parallel
    E_res_J = 0.5 * ELECTRON_MASS * v_res**2
    return E_res_J / EV_TO_JOULE


def radiation_belt_parameters(L_shell: float = 4.5,
                               B_eq_nT: float = 300.0,
                               n_e_cm3: float = 1.0,
                               T_e_keV: float = 5.0) -> dict:
    """
    辐射带典型参数集 (偶极磁场模型):
        B(L) = B_eq / L^3
    返回包含所有特征物理量的字典.
    """
    B0 = B_eq_nT * 1e-9
    n_e = n_e_cm3 * 1e6   # cm^{-3} -> m^{-3}
    T_e_eV = T_e_keV * 1e3

    omega_pe = electron_plasma_frequency(n_e)
    omega_ce = electron_cyclotron_frequency(B0)
    d_e = electron_inertial_length(n_e)
    lam_D = debye_length(T_e_eV, n_e)
    v_th = thermal_velocity(T_e_eV)
    beta_val = plasma_beta(n_e, T_e_eV, B0)

    return {
        'L_shell': L_shell,
        'B0_T': B0,
        'n_e_m3': n_e,
        'T_e_eV': T_e_eV,
        'omega_pe': omega_pe,
        'omega_ce': omega_ce,
        'omega_ratio': omega_pe / max(omega_ce, 1e-30),
        'd_e_m': d_e,
        'lambda_D_m': lam_D,
        'v_th_m_s': v_th,
        'beta': beta_val,
    }


DEFAULT_RADIATION_BELT = radiation_belt_parameters()
