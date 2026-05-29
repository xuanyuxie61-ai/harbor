"""
state_equation.py
=================
等离子体状态方程（Equation of State, EOS）模块。

本模块为 ICF 内爆提供电子/离子热力学量计算，包含：
1. 理想气体状态方程
2. 电子简并压修正（Fermi-Dirac 统计）
3. 库仑相互作用修正（Debye-Huckel 近似）
4. 辐射压贡献（Stefan-Boltzmann 定律）

核心公式：
- 总压强: P = P_ion + P_e + P_rad
- 内能密度: epsilon = epsilon_ion + epsilon_e + epsilon_rad
- 声速: c_s = sqrt(gamma * P / rho)

数值积分采用 quadrature_rules 模块生成的高斯求积公式（基于原项目 608_jacobi_rule）。
"""

import numpy as np
from typing import Tuple
from icf_parameters import PC, EOS
from quadrature_rules import compute_fermi_dirac_integral


def electron_thermal_debroglie(T: float) -> float:
    """
    电子热德布罗意波长:
        lambda_T = h / sqrt(2*pi*m_e*k_B*T)
    """
    if T <= 0.0:
        return 1.0e30
    return PC.PLANCK / np.sqrt(2.0 * np.pi * PC.ELECTRON_MASS * PC.BOLTZMANN * T)


def electron_number_density(rho, Z_eff, A_avg):
    """
    电子数密度:
        n_e = Z_eff * rho * N_A / A_avg  [m^-3]
    支持标量或 numpy 数组输入。
    """
    A_avg = np.asarray(A_avg)
    result = np.zeros_like(np.asarray(rho), dtype=float)
    mask = A_avg > 0.0
    result[mask] = np.asarray(Z_eff)[mask] * np.asarray(rho)[mask] * PC.AVOGADRO / (A_avg[mask] * 1.0e-3)
    return result


def fermi_energy(n_e: float) -> float:
    """
    零温费米能:
        E_F = (h^2 / (2*m_e)) * (3*n_e / (8*pi))^(2/3)
    """
    if n_e <= 0.0:
        return 0.0
    prefactor = PC.PLANCK**2 / (2.0 * PC.ELECTRON_MASS)
    return prefactor * (3.0 * n_e / (8.0 * np.pi))**(2.0 / 3.0)


def degeneracy_parameter(n_e: float, T: float) -> float:
    """
    电子简并参数 theta = k_B*T / E_F。
    theta >> 1 : 经典极限
    theta << 1 : 强简并极限
    """
    ef = fermi_energy(n_e)
    if ef <= 1.0e-30 or T <= 0.0:
        return 1.0e30
    return PC.BOLTZMANN * T / ef


def electron_pressure_ideal(n_e: float, T: float) -> float:
    """电子理想气体压强: P_e,ideal = n_e * k_B * T"""
    if n_e <= 0.0 or T <= 0.0:
        return 0.0
    return n_e * PC.BOLTZMANN * T


def electron_pressure_degenerate(n_e: float) -> float:
    """
    强简并电子压强（零温近似）:
        P_F = (2/5) * n_e * E_F
    """
    if n_e <= 0.0:
        return 0.0
    ef = fermi_energy(n_e)
    return (2.0 / 5.0) * n_e * ef


def electron_pressure_full(n_e: float, T: float) -> float:
    """
    电子压强（理想+简并过渡）。
    采用 Padé 近似平滑过渡:
        P_e = P_deg * sqrt(1 + (P_ideal/P_deg)^2)
    等价于: P_e = sqrt(P_deg^2 + P_ideal^2)
    """
    p_ideal = electron_pressure_ideal(n_e, T)
    p_deg = electron_pressure_degenerate(n_e)
    return np.sqrt(p_deg**2 + p_ideal**2)


def electron_internal_energy(n_e: float, T: float) -> float:
    """
    电子内能密度 [J/m^3]。
    经典: epsilon_e = (3/2) * n_e * k_B * T
    简并修正: 采用 Fermi-Dirac 积分数值结果
    """
    if n_e <= 0.0 or T <= 0.0:
        return 0.0

    theta = degeneracy_parameter(n_e, T)
    if theta > 10.0:
        # 经典极限
        return 1.5 * n_e * PC.BOLTZMANN * T
    elif theta < 0.1:
        # 强简并极限
        ef = fermi_energy(n_e)
        return 0.6 * n_e * ef * (1.0 + 1.25 * theta**2)
    else:
        # 过渡区: 使用 Fermi-Dirac 积分 F_{3/2}(eta)
        # 近似 eta ≈ ln(n_e * lambda_T^3 / 2)
        lambda_t = electron_thermal_debroglie(T)
        fugacity = n_e * lambda_t**3 / 2.0
        eta = np.log(max(fugacity, 1.0e-30))
        f32 = compute_fermi_dirac_integral(1, eta, n_quad=32)
        return (3.0 / 2.0) * n_e * PC.BOLTZMANN * T * (2.0 * f32
              / (3.0 * np.sqrt(np.pi) * max(fugacity, 1.0e-30)))


def coulomb_correction_pressure(n_e: float, Z_eff: float, T: float) -> float:
    """
    库仑压强修正（Debye-Huckel 极限）:
        P_coul = -(1/3) * (e^2 / (4*pi*epsilon_0)) * (k_B*T)^(1/2) * n_e^(3/2) * Z_eff^(1/2)
    适用条件: Gamma < 1（弱耦合等离子体）
    """
    if n_e <= 0.0 or T <= 0.0 or Z_eff <= 0.0:
        return 0.0

    # 等离子体耦合参数 Gamma
    a_ws = (3.0 / (4.0 * np.pi * n_e))**(1.0 / 3.0)  # Wigner-Seitz 半径
    coulomb_energy = PC.ELEMENTARY_CHARGE**2 / (4.0 * np.pi * PC.VACUUM_PERMITTIVITY * a_ws)
    gamma = coulomb_energy / (PC.BOLTZMANN * T)

    if gamma >= 1.0:
        # 强耦合区，修正系数减小
        gamma = min(gamma, 10.0)
        corr = -0.3 * gamma * n_e * PC.BOLTZMANN * T
    else:
        corr = -EOS.COULOMB_CORRECTION * gamma * n_e * PC.BOLTZMANN * T

    return corr


def radiation_pressure(T: float) -> float:
    """
    辐射压强:
        P_rad = (4/3) * sigma_SB * T^4 / c
    """
    if T <= 0.0:
        return 0.0
    return (4.0 / 3.0) * PC.STEFAN_BOLTZMANN * T**4 / PC.SPEED_OF_LIGHT


def radiation_energy_density(T: float) -> float:
    """
    辐射能量密度:
        epsilon_rad = sigma_SB * T^4 / c
    """
    if T <= 0.0:
        return 0.0
    return PC.STEFAN_BOLTZMANN * T**4 / PC.SPEED_OF_LIGHT


def total_pressure(rho: float, T_e: float, T_i: float,
                   Z_eff: float, A_avg: float) -> float:
    """
    总压强 [Pa]:
        P = P_ion + P_e + P_coul + P_rad
    """
    if rho <= 0.0 or T_e < 0.0 or T_i < 0.0:
        return 0.0

    n_e = electron_number_density(rho, Z_eff, A_avg)
    n_i = n_e / max(Z_eff, 1.0e-10)

    p_ion = n_i * PC.BOLTZMANN * T_i
    p_e = electron_pressure_full(n_e, T_e)
    p_coul = coulomb_correction_pressure(n_e, Z_eff, T_e)
    p_rad = radiation_pressure(T_e)

    return max(p_ion + p_e + p_coul + p_rad, 1.0e-20)


def total_internal_energy(rho: float, T_e: float, T_i: float,
                          Z_eff: float, A_avg: float) -> float:
    """
    总比内能 [J/kg]:
        e = e_ion + e_e + e_rad
    """
    if rho <= 0.0:
        return 0.0

    n_e = electron_number_density(rho, Z_eff, A_avg)
    n_i = n_e / max(Z_eff, 1.0e-10)

    eps_ion_vol = 1.5 * n_i * PC.BOLTZMANN * T_i
    eps_e_vol = electron_internal_energy(n_e, T_e)
    eps_rad_vol = radiation_energy_density(T_e)

    eps_total_vol = eps_ion_vol + eps_e_vol + eps_rad_vol
    return eps_total_vol / rho


def sound_speed(rho: float, T_e: float, T_i: float,
                Z_eff: float, A_avg: float) -> float:
    """
    绝热声速:
        c_s = sqrt( gamma * P / rho )
    其中 gamma 为有效绝热指数。
    """
    if rho <= 0.0:
        return 1.0e-10
    p = total_pressure(rho, T_e, T_i, Z_eff, A_avg)
    gamma_eff = EOS.GAMMA_IDEAL
    # 辐射主导区修正
    p_rad = radiation_pressure(T_e)
    if p_rad > 0.5 * p:
        gamma_eff = 4.0 / 3.0
    return np.sqrt(gamma_eff * p / rho)


def ionization_state_Saha(rho: float, T: float, Z_nuc: float,
                          ionization_energy: float) -> float:
    """
    Saha 方程计算平均电离度（简化单能级模型）。

    n_{Z+1} * n_e / n_Z = (2*pi*m_e*k_B*T/h^2)^(3/2) * 2 * exp(-chi/(k_B*T))

    返回平均电离度 Z_eff。

    TODO: 实现 Saha 电离方程。需要根据密度、温度、核电荷数和电离能
    计算等离子体的平均电离度。涉及核心等离子体物理公式：
    - Saha 平衡常数的计算
    - 二次方程求解 x^2/(1-x) = ratio
    - 考虑简并压修正（可选）
    """
    # TODO: 实现 Saha 电离度计算
    raise NotImplementedError("Saha ionization model not implemented")


def compute_eos_table(rho_vals: np.ndarray, T_vals: np.ndarray,
                      Z_nuc: float, A_avg: float,
                      ionization_energy: float) -> dict:
    """
    构建 EOS 查找表，返回压强、内能、声速的二维数组。
    """
    nr, nt = len(rho_vals), len(T_vals)
    P_table = np.zeros((nr, nt))
    E_table = np.zeros((nr, nt))
    C_table = np.zeros((nr, nt))
    Z_table = np.zeros((nr, nt))

    for i, rho in enumerate(rho_vals):
        for j, T in enumerate(T_vals):
            Z_eff = ionization_state_Saha(rho, T, Z_nuc, ionization_energy)
            Z_table[i, j] = Z_eff
            P_table[i, j] = total_pressure(rho, T, T, Z_eff, A_avg)
            E_table[i, j] = total_internal_energy(rho, T, T, Z_eff, A_avg)
            C_table[i, j] = sound_speed(rho, T, T, Z_eff, A_avg)

    return {
        "pressure": P_table,
        "energy": E_table,
        "sound_speed": C_table,
        "ionization": Z_table,
        "rho": rho_vals,
        "T": T_vals,
    }
