"""
laser_propagation.py
====================
激光在等离子体中的传播与能量沉积模块。

融合原项目 1061_schroedinger_nonlinear_pde（非线性薛定谔方程离散）与
597_iplot（函数采样与插值）的核心思想，
近似描述激光束在冕区等离子体中的逆轫致辐射吸收与折射传播。

物理模型：
1. 激光电场包络满足近轴波动方程（非线性 Schrödinger 近似）：
     2*i*k0 * dE/dz + del_perp^2 E + (k^2 - k0^2) * E = 0
   其中 k0 = omega/c, k = n*omega/c, n 为等离子体折射率。

2. 能量沉积采用逆轫致辐射（Inverse Bremsstrahlung）:
     dI/ds = -kappa_IB * I
   其中吸收系数:
     kappa_IB = (n_e^2 * Z * e^2) / (n_c * m_e * epsilon_0 * c * nu_ei)

3. 临界密度:
     n_c = epsilon_0 * m_e * omega^2 / e^2

数值方法：
- 径向传播采用一维离散（利用 iplot 的均匀采样思想）
- 非线性折射率修正采用中心差分（借鉴 NLSE 的空间离散）
"""

import numpy as np
from typing import Tuple
from icf_parameters import LP, PC
from utils import safe_divide, clamp_array


def critical_density(wavelength: float) -> float:
    """
    激光临界密度:
        n_c = epsilon_0 * m_e * (2*pi*c/lambda)^2 / e^2
    """
    omega = 2.0 * np.pi * PC.SPEED_OF_LIGHT / wavelength
    nc = PC.VACUUM_PERMITTIVITY * PC.ELECTRON_MASS * omega**2 / PC.ELEMENTARY_CHARGE**2
    return nc


def plasma_refractive_index(n_e: float, n_c: float) -> float:
    """
    冷等离子体折射率:
        n^2 = 1 - n_e / n_c
    当 n_e >= n_c 时，n = 0（截止）
    """
    ratio = n_e / max(n_c, 1.0e-10)
    if ratio >= 1.0:
        return 0.0
    return np.sqrt(max(1.0 - ratio, 0.0))


def electron_ion_collision_freq(n_e: float, T_e: float, Z_eff: float) -> float:
    """
    电子-离子碰撞频率（Spitzer）:
        nu_ei = (Z * n_e * e^4 * lnLambda) / (3 * (2*pi)^(3/2) * epsilon_0^2 * m_e^(1/2) * (k_B*T_e)^(3/2))
    简化形式 [Hz]:
        nu_ei ≈ 2.9e-12 * Z * n_e * lnLambda / T_e^(3/2)   [n_e in m^-3, T_e in K]
    """
    if T_e <= 0.0 or n_e <= 0.0:
        return 0.0
    ln_lambda = max(23.5 - np.log(np.sqrt(n_e) / max(T_e, 1.0)), 2.0)
    nu = 2.9e-12 * Z_eff * n_e * ln_lambda / (T_e**1.5)
    return max(nu, 0.0)


def inverse_bremsstrahlung_coeff(n_e: float, T_e: float, Z_eff: float,
                                  wavelength: float) -> float:
    """
    逆轫致辐射吸收系数 [m^-1]。

    kappa_IB = (n_e^2 * Z_eff * e^2 * nu_ei) / (n_c * m_e * epsilon_0 * c * omega^2)
    利用 omega^2 = n_c * e^2 / (epsilon_0 * m_e) 简化得:
    kappa_IB = (n_e^2 / n_c^2) * (Z_eff * nu_ei) / c
    """
    nc = critical_density(wavelength)
    if nc <= 1.0e-10 or n_e <= 0.0:
        return 0.0

    nu_ei = electron_ion_collision_freq(n_e, T_e, Z_eff)
    ratio = n_e / nc
    kappa = ratio**2 * Z_eff * nu_ei / PC.SPEED_OF_LIGHT
    return max(kappa, 0.0)


def nlse_envelope_discrete(E: np.ndarray, z: float, dz: float,
                           n_e_profile: np.ndarray, n_c: float,
                           dr: float) -> np.ndarray:
    """
    非线性薛定谔方程（NLSE）包络传播的一步离散。

    基于原项目 1061_schroedinger_nonlinear_pde 的空间离散思想，
    对径向 Laplacian 采用中心差分:
        del^2 E = (E_{j+1} - 2*E_j + E_{j-1}) / dr^2 + (1/r) * (E_{j+1} - E_{j-1}) / (2*dr)

    近轴方程:  dE/dz = (i / (2*k0)) * del^2 E + i * (k - k0) * E
    """
    n = len(E)
    dEdz = np.zeros(n, dtype=complex)
    k0 = 2.0 * np.pi / LP.WAVELENGTH

    for j in range(n):
        r_j = (j + 0.5) * dr
        ne = n_e_profile[j]
        n_index = plasma_refractive_index(ne, n_c)
        k = k0 * n_index

        # 径向 Laplacian
        if j == 0:
            laplace = (E[j + 1] - 2.0 * E[j] + E[j]) / dr**2
        elif j == n - 1:
            laplace = (E[j] - 2.0 * E[j] + E[j - 1]) / dr**2
        else:
            second_deriv = (E[j + 1] - 2.0 * E[j] + E[j - 1]) / dr**2
            first_deriv = (E[j + 1] - E[j - 1]) / (2.0 * dr)
            laplace = second_deriv + first_deriv / max(r_j, 1.0e-15)

        dEdz[j] = 1j / (2.0 * k0) * laplace + 1j * (k - k0) * E[j]

    # 一步 Euler 推进
    return E + dz * dEdz


def compute_laser_deposition_1d(r_cells: np.ndarray, r_nodes: np.ndarray,
                                rho: np.ndarray, T_e: np.ndarray,
                                Z_eff: np.ndarray,
                                beam_power: float,
                                total_time: float,
                                n_samples: int = 101) -> np.ndarray:
    """
    计算激光在一维径向等离子体中的能量沉积率 [W/m^3]。

    物理模型：
    1. 激光从外边界向内传播，在临界密度面处截止/反射。
    2. 每个球壳单元的功率衰减服从 Beer-Lambert 定律:
           P_{i+1} = P_i * exp(-kappa_i * dr_i)
    3. 单元沉积功率 = P_in - P_out
    4. 体沉积率 = 沉积功率 / 单元体积

    参数
    ----
    r_cells, r_nodes : np.ndarray
        单元中心与节点坐标
    rho, T_e, Z_eff : np.ndarray
        等离子体状态
    beam_power : float
        激光功率 [W]
    total_time : float
        当前时刻 [s]（保留用于后续多脉冲模型）
    n_samples : int
        保留参数（iplot 采样思想的接口兼容性）

    返回
    ----
    deposition : np.ndarray
        每个单元的能量沉积率 [W/m^3]
    """
    n_cells = len(r_cells)
    deposition = np.zeros(n_cells)

    if beam_power <= 0.0:
        return deposition

    nc = critical_density(LP.WAVELENGTH)

    # 电子数密度
    n_e = np.zeros(n_cells)
    for i in range(n_cells):
        n_e[i] = Z_eff[i] * rho[i] * PC.AVOGADRO / (2.5 * 1.0e-3)

    # 找到临界密度面（从内向外搜索）
    critical_surface_idx = -1
    for i in range(n_cells - 1, -1, -1):
        if n_e[i] < nc:
            critical_surface_idx = i
            break

    if critical_surface_idx < 0:
        return deposition  # 全过密，无传播

    # 初始入射功率（考虑总吸收效率 ~0.8）
    absorption_efficiency = 0.8
    P_incident = beam_power * absorption_efficiency

    # 从外向内传播
    P_current = P_incident

    for i in range(n_cells - 1, critical_surface_idx - 1, -1):
        dr = r_nodes[i + 1] - r_nodes[i]
        kappa = inverse_bremsstrahlung_coeff(n_e[i], T_e[i], Z_eff[i], LP.WAVELENGTH)

        # Beer-Lambert 衰减
        attenuation = np.exp(-kappa * dr)
        P_out = P_current * attenuation
        P_absorbed = P_current - P_out

        # 体沉积率
        vol = 4.0 * np.pi / 3.0 * (r_nodes[i + 1]**3 - r_nodes[i]**3)
        if vol > 1.0e-30:
            deposition[i] = P_absorbed / vol

        P_current = P_out
        if P_current < 1.0e-6 * P_incident:
            break

    return deposition


def laser_power_time(t: float) -> float:
    """激光功率随时间变化 [W]。"""
    return LP.power_profile(t)
