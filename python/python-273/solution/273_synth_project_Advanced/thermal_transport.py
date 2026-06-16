"""
thermal_transport.py — 声子热输运与 Boltzmann 输运方程
=====================================================

融合种子项目:
  - 1149_BanerjeeLab: ODE 系统积分 + 稳态搜索 + 扰动分析
  - 1242_ce335805_PhotonDosReference: DOS 积分 + 模式求和
  - 195_coin_simulation: 运行平均 + 统计收敛

物理背景:
  晶格热导率由 Boltzmann 输运方程 (BTE) 确定:
    kappa_{ab} = (1/V) * sum_{lambda} C_lambda * v_{lambda,a} * v_{lambda,b} * tau_lambda

  其中:
    C_lambda = hbar * omega_lambda * dn_BE/dT  (模式热容)
    v_lambda = grad_k omega_lambda  (群速度)
    tau_lambda = 散射弛豫时间

  散射机制:
    1. Umklapp: 1/tau_U = A_U * omega^2 * T * exp(-Theta_D / (3T))
    2. Normal:  1/tau_N = A_N * omega * T^3 (低 T 主导)
    3. 边界:    1/tau_B = v_s / L (L 为样品尺寸)
    4. 同位素:  1/tau_iso = (pi/2) * Gamma * omega^2 * DOS(omega)
    5. 缺陷:    1/tau_def = (V_0/4*pi*v_s^3) * (delta_M/M)^2 * omega^4

  Bose-Einstein 分布:
    n_BE(omega, T) = 1 / (exp(hbar*omega / k_B*T) - 1)
"""

import numpy as np
from typing import Tuple, Dict, List


def bose_einstein(omega: np.ndarray, temperature: float) -> np.ndarray:
    """
    Bose-Einstein 分布函数。
    n_BE = 1 / (exp(hbar*omega / k_B*T) - 1)

    边界处理: omega -> 0 时 n_BE -> k_B*T / (hbar*omega)
    """
    hbar = 1.0546e-34
    k_B = 1.3806e-23
    if temperature < 1e-10:
        return np.zeros_like(omega)
    x = hbar * np.maximum(omega, 1e-20) / (k_B * temperature)
    x = np.minimum(x, 500.0)  # 防止 exp 溢出
    n_BE = np.where(x > 1e-10, 1.0 / (np.exp(x) - 1.0 + 1e-30), 0.0)
    return n_BE


def mode_heat_capacity(
    omega: np.ndarray, temperature: float,
) -> np.ndarray:
    """
    模式热容: C_lambda = k_B * x^2 * exp(x) / (exp(x) - 1)^2
    其中 x = hbar*omega / (k_B*T)

    高温极限: C -> k_B (Dulong-Petit)
    低温极限: C -> k_B * x^2 * exp(-x) -> 0
    """
    hbar = 1.0546e-34
    k_B = 1.3806e-23
    if temperature < 1e-10:
        return np.zeros_like(omega)
    x = hbar * np.maximum(omega, 1e-20) / (k_B * temperature)
    x = np.minimum(x, 500.0)
    exp_x = np.exp(np.minimum(x, 500.0))
    C = k_B * x ** 2 * exp_x / (exp_x - 1.0 + 1e-30) ** 2
    C = np.where(x > 1e-10, C, k_B)  # omega->0 极限
    return C


def umklapp_scattering_rate(
    omega: np.ndarray,
    temperature: float,
    A_U: float = 1e-18,
    theta_D: float = 645.0,
) -> np.ndarray:
    """
    Umklapp 散射率: 1/tau_U = A_U * omega^2 * T * exp(-Theta_D / (3T))

    参数:
        omega: (Nm,) 声子频率 (rad/s)
        temperature: 温度 (K)
        A_U: Umklapp 系数 (s^2/K)
        theta_D: 德拜温度 (K)
    """
    if temperature < 1e-10:
        return np.zeros_like(omega)
    gamma = A_U * omega ** 2 * temperature * np.exp(-theta_D / (3.0 * temperature))
    return gamma


def normal_scattering_rate(
    omega: np.ndarray,
    temperature: float,
    A_N: float = 1e-19,
) -> np.ndarray:
    """
    Normal 散射率: 1/tau_N = A_N * omega * T^3
    (低温下主导, 不产生热阻但影响分布函数)
    """
    return A_N * np.abs(omega) * temperature ** 3


def boundary_scattering_rate(
    v_sound: float,
    L: float = 1e-6,
) -> float:
    """
    边界散射率: 1/tau_B = v_s / L

    参数:
        v_sound: 声速 (m/s)
        L: 样品特征尺寸 (m)
    """
    return v_sound / max(L, 1e-15)


def isotope_scattering_rate(
    omega: np.ndarray,
    gamma_iso: float = 1e-4,
    v_sound: float = 5000.0,
    volume_atom: float = 2e-29,
) -> np.ndarray:
    """
    同位素散射率 (Tamura 公式):
    1/tau_iso = (pi/2) * Gamma * omega^2 * DOS(omega) * V_0 / v_s^3

    简化模型: 1/tau_iso = B_iso * omega^2
    B_iso = (V_0 / 4*pi*v_s^3) * Gamma

    参数:
        gamma_iso: 同位素质量涨落参数 Gamma
        v_sound: 平均声速
        volume_atom: 原子体积
    """
    B_iso = (volume_atom / (4.0 * np.pi * v_sound ** 3)) * gamma_iso
    return B_iso * omega ** 2


def total_scattering_rate(
    omega: np.ndarray,
    temperature: float,
    A_U: float = 1e-18,
    A_N: float = 1e-19,
    theta_D: float = 645.0,
    v_sound: float = 5000.0,
    L: float = 1e-6,
    gamma_iso: float = 1e-4,
) -> np.ndarray:
    """
    总散射率 (Matthiessen 规则):
    1/tau_total = 1/tau_U + 1/tau_N + 1/tau_B + 1/tau_iso
    """
    gamma_U = umklapp_scattering_rate(omega, temperature, A_U, theta_D)
    gamma_N = normal_scattering_rate(omega, temperature, A_N)
    gamma_B = boundary_scattering_rate(v_sound, L) * np.ones_like(omega)
    gamma_iso = isotope_scattering_rate(omega, gamma_iso, v_sound)
    return gamma_U + gamma_N + gamma_B + gamma_iso


def compute_lattice_thermal_conductivity(
    omega_all: np.ndarray,
    v_group_all: np.ndarray,
    temperatures: np.ndarray,
    volume: float,
    A_U: float = 1e-18,
    theta_D: float = 645.0,
    v_sound: float = 5000.0,
) -> np.ndarray:
    """
    计算晶格热导率 kappa(T)。

    kappa = (1/V) * sum_lambda C_lambda * v_lambda^2 * tau_lambda
           = (1/V) * sum_lambda C_lambda * v_lambda^2 / gamma_lambda

    融合 BanerjeeLab 的运行平均收敛策略: 对 k 点求和使用
    运行平均值监测收敛。

    参数:
        omega_all: (N_k * N_modes,) 所有 k 点的声子频率
        v_group_all: (N_k * N_modes, 3) 群速度
        temperatures: (N_T,) 温度数组
        volume: 超胞体积
        A_U: Umklapp 系数
        theta_D: 德拜温度
        v_sound: 平均声速

    返回:
        kappa: (N_T,) 热导率 (W/m/K)
    """
    hbar = 1.0546e-34
    v_sq = np.sum(v_group_all ** 2, axis=1)  # |v_g|^2
    n_modes_total = len(omega_all)

    kappa = np.zeros(len(temperatures))
    for ti, T in enumerate(temperatures):
        if T < 1e-10:
            kappa[ti] = 0.0
            continue

        C_modes = mode_heat_capacity(omega_all, T)
        gamma = total_scattering_rate(omega_all, T, A_U=A_U, theta_D=theta_D,
                                      v_sound=v_sound)
        gamma = np.maximum(gamma, 1e-30)  # 防止除零
        tau = 1.0 / gamma

        # 对模式和 k 点求和
        # kappa_ab = (1/3V) * sum C * v^2 * tau (各向同性近似)
        kappa[ti] = np.sum(C_modes * v_sq * tau) / (3.0 * volume)

    return kappa


def callaway_model(
    temperature: float,
    theta_D: float,
    v_sound: float,
    volume_atom: float,
    A_U: float = 1e-18,
    A_N: float = 1e-19,
) -> float:
    """
    Callaway 模型热导率 (解析近似)。

    kappa = (k_B/2*pi^2*v_s) * (k_B*T/hbar)^3 * integral_0^{Theta_D/T}
            tau_c * x^4 * exp(x) / (exp(x) - 1)^2 dx

    其中 x = hbar*omega/(k_B*T), tau_c^{-1} = tau_U^{-1} + tau_N^{-1} + ...

    使用简化数值积分 (Simpson 法则)。
    """
    hbar = 1.0546e-34
    k_B = 1.3806e-23
    if temperature < 1e-10:
        return 0.0

    x_D = theta_D / temperature
    n_quad = 200
    x = np.linspace(1e-6, x_D, n_quad)
    dx = x[1] - x[0]

    omega = x * k_B * temperature / hbar
    tau_U = 1.0 / np.maximum(umklapp_scattering_rate(omega, temperature, A_U, theta_D), 1e-30)
    tau_N = 1.0 / np.maximum(normal_scattering_rate(omega, temperature, A_N), 1e-30)
    tau_B = 1.0 / boundary_scattering_rate(v_sound)
    tau_c = 1.0 / (1.0 / tau_U + 1.0 / tau_N + 1.0 / tau_B)

    exp_x = np.exp(np.minimum(x, 500.0))
    integrand = tau_c * x ** 4 * exp_x / (exp_x - 1.0 + 1e-30) ** 2

    # Simpson 积分
    integral = np.trapz(integrand, x)

    prefactor = k_B / (2.0 * np.pi ** 2 * v_sound) * (k_B * temperature / hbar) ** 3
    return prefactor * integral
