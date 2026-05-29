"""
utils.py
物理常数、数值工具与多维网格生成模块

整合原项目 1175_subpak (grid generation utilities)
为表面催化反应分子动力学模拟提供基础数值基础设施
"""

import numpy as np
from typing import Tuple

# =============================================================================
# 物理常数 (SI 单位制)
# =============================================================================
BOLTZMANN_KB = 1.380649e-23      # J/K
AVOGADRO_NA = 6.02214076e23      # mol^-1
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PLANCK_H = 6.62607015e-34        # J·s
EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1.0e-10
FS_TO_S = 1.0e-15
AMU_TO_KG = 1.66053906660e-27

# Pt(111) 表面晶格常数
PT_LATTICE_CONSTANT = 3.924e-10  # m
PT_ATOMIC_MASS = 195.084         # amu

# 吸附物种质量 (amu)
MASS_CO = 28.010
MASS_O2 = 31.998


def kb_t_ev(temperature_k: float) -> float:
    """
    将热能 k_B * T 转换为 eV
    
    公式: E = k_B * T / e
    """
    if temperature_k < 0.0:
        raise ValueError("温度必须非负")
    return BOLTZMANN_KB * temperature_k / ELEMENTARY_CHARGE


def maxwell_boltzmann_speed(mass_amu: float, temperature_k: float) -> float:
    """
    Maxwell-Boltzmann 分布的最概然速率
    
    公式:
        v_p = sqrt(2 * k_B * T / m)
    
    参数:
        mass_amu: 粒子质量 (amu)
        temperature_k: 温度 (K)
    
    返回:
        最概然速率 (m/s)
    """
    if mass_amu <= 0.0 or temperature_k < 0.0:
        raise ValueError("mass_amu > 0 且 temperature_k >= 0")
    m_kg = mass_amu * AMU_TO_KG
    return np.sqrt(2.0 * BOLTZMANN_KB * temperature_k / m_kg)


def de_broglie_thermal_wavelength(mass_amu: float, temperature_k: float) -> float:
    """
    热德布罗意波长
    
    公式:
        Λ = h / sqrt(2 * π * m * k_B * T)
    
    该波长决定量子效应在表面吸附中的重要性
    """
    if mass_amu <= 0.0 or temperature_k <= 0.0:
        raise ValueError("mass_amu > 0 且 temperature_k > 0")
    m_kg = mass_amu * AMU_TO_KG
    return PLANCK_H / np.sqrt(2.0 * np.pi * m_kg * BOLTZMANN_KB * temperature_k)


def grid_uniform_1d(xmin: float, xmax: float, nstep: int) -> np.ndarray:
    """
    一维均匀网格生成
    
    整合原项目 1175_subpak/grid1 思想
    在 [xmin, xmax] 上生成 nstep 个等距节点
    
    公式:
        x_i = ((nstep - i) * xmin + (i - 1) * xmax) / (nstep - 1)
    """
    if nstep < 2:
        raise ValueError("nstep 必须 >= 2")
    if xmax <= xmin:
        raise ValueError("xmax 必须 > xmin")
    i = np.arange(1, nstep + 1)
    return ((nstep - i) * xmin + (i - 1) * xmax) / (nstep - 1)


def grid_uniform_nd(ndim: int, nstep: int, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    多维均匀网格生成
    
    参数:
        ndim: 空间维度
        nstep: 每维步数 (总节点数 = nstep^ndim)
        x1: 起点坐标 (ndim,)
        x2: 终点坐标 (ndim,)
    
    返回:
        grid: (ndim, nstep^ndim) 网格节点坐标
    """
    if nstep < 2:
        raise ValueError("nstep 必须 >= 2")
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.shape != (ndim,) or x2.shape != (ndim,):
        raise ValueError("x1, x2 形状必须为 (ndim,)")
    # 使用 numpy meshgrid 生成多维网格
    linspaces = [np.linspace(x1[d], x2[d], nstep) for d in range(ndim)]
    if ndim == 2:
        mg = np.meshgrid(*linspaces, indexing='ij')
        grid = np.vstack([g.ravel() for g in mg])
    elif ndim == 3:
        mg = np.meshgrid(*linspaces, indexing='ij')
        grid = np.vstack([g.ravel() for g in mg])
    else:
        mg = np.meshgrid(*linspaces, indexing='ij')
        grid = np.vstack([g.ravel() for g in mg])
    return grid


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """安全除法，避免除以零"""
    result = np.full_like(a, fill_value, dtype=float)
    mask = np.abs(b) > 1e-300
    result[mask] = a[mask] / b[mask]
    return result


def morse_potential(r: np.ndarray, d_e: float, a_param: float, r_e: float) -> np.ndarray:
    """
    Morse 势能函数
    
    公式:
        V(r) = D_e * [1 - exp(-a * (r - r_e))]^2 - D_e
    
    参数:
        r: 原子间距 (m)
        d_e: 解离能 (eV)
        a_param: 势阱宽度参数 (m^-1)
        r_e: 平衡键长 (m)
    
    返回:
        势能 (eV)
    """
    r = np.asarray(r, dtype=float)
    dr = r - r_e
    return d_e * (1.0 - np.exp(-a_param * dr)) ** 2 - d_e


def lennard_jones_potential(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """
    Lennard-Jones 12-6 势能
    
    公式:
        V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6]
    
    参数:
        r: 原子间距 (m)
        epsilon: 势阱深度 (eV)
        sigma: 有限距离参数 (m)
    """
    r = np.asarray(r, dtype=float)
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def arrhenius_rate(pre_exponential: float, activation_energy_ev: float,
                   temperature_k: float) -> float:
    """
    Arrhenius 反应速率常数
    
    公式:
        k = A * exp(-E_a / (k_B * T))
    
    参数:
        pre_exponential: 指前因子 A (s^-1)
        activation_energy_ev: 活化能 E_a (eV)
        temperature_k: 温度 (K)
    """
    if temperature_k <= 0.0:
        return 0.0
    kb_t = kb_t_ev(temperature_k)
    return pre_exponential * np.exp(-activation_energy_ev / kb_t)


def sticking_coefficient_langmuir(pressure_pa: float, temperature_k: float,
                                  alpha0: float, e_ads_ev: float) -> float:
    """
    Langmuir 吸附模型中的粘附系数
    
    公式:
        s = s_0 * (1 - θ) * exp(-E_ads / (k_B * T))
    
    其中 θ 为表面覆盖率
    """
    if temperature_k <= 0.0:
        return 0.0
    kb_t = kb_t_ev(temperature_k)
    return alpha0 * np.exp(-e_ads_ev / kb_t)
