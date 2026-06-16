"""
physical_constants.py — 高能物理常数与单位系统
==============================================

本模块定义计算高能物理中使用的物理常数、单位转换因子和相对论运动学公式。

所有计算采用自然单位制 (c = 1)，必要时转换回 SI/HEP 实用单位。

核心物理量:
    c           : 光速                    = 2.998e8 m/s
    e           : 元电荷                  = 1.602e-19 C
    m_e         : 电子质量                = 0.511 MeV/c^2
    m_p         : 质子质量                = 938.272 MeV/c^2
    m_mu        : μ子质量                 = 105.658 MeV/c^2
    m_pi        : 带电π介子质量           = 139.570 MeV/c^2
    X0_Si       : 硅辐射长度             = 9.370 cm
    I_Si        : 硅平均激发能           = 173 eV

相对论运动学:
    E^2 = (pc)^2 + (mc^2)^2
    β   = pc / E
    γ   = E / (mc^2)
    p_T = p · sin(λ)    (横向动量)
"""

import math

# ============================================================
# 自然单位制常数 (c = 1)
# ============================================================
C_LIGHT = 2.99792458e8          # 光速 [m/s]
E_CHARGE = 1.602176634e-19      # 元电荷 [C]
HBAR_C = 197.3269804            # ℏc [MeV·fm]
AVOGADRO = 6.02214076e23        # Avogadro 常数 [mol^{-1}]

# ============================================================
# 粒子质量 [MeV/c^2]
# ============================================================
MASS_ELECTRON = 0.510998950     # 电子质量
MASS_MUON = 105.6583745         # μ子质量
MASS_PION = 139.57039           # π±介子质量
MASS_KAON = 493.677             # K±介子质量
MASS_PROTON = 938.272088        # 质子质量
MASS_DEUTERON = 1875.612942     # 氘核质量

# ============================================================
# 常见粒子质量字典
# ============================================================
PARTICLE_MASSES = {
    'electron': MASS_ELECTRON,
    'muon': MASS_MUON,
    'pion': MASS_PION,
    'kaon': MASS_KAON,
    'proton': MASS_PROTON,
    'deuteron': MASS_DEUTERON,
}

# ============================================================
# 材料参数
# ============================================================
# 辐射长度 X0 [cm]
RADIATION_LENGTH = {
    'silicon': 9.370,
    'carbon': 18.80,
    'aluminum': 8.896,
    'iron': 1.760,
    'lead': 0.5612,
    'air': 36620.0,
    'beryllium': 35.28,
}

# 平均激发能 I [eV]
IONIZATION_POTENTIAL = {
    'silicon': 173.0,
    'carbon': 81.0,
    'aluminum': 166.0,
    'iron': 286.0,
    'lead': 823.0,
    'air': 85.7,
    'beryllium': 63.7,
}

# 材料密度 ρ [g/cm^3]
MATERIAL_DENSITY = {
    'silicon': 2.329,
    'carbon': 2.265,
    'aluminum': 2.699,
    'iron': 7.874,
    'lead': 11.35,
    'air': 0.001225,
    'beryllium': 1.848,
}

# 原子量 A [g/mol]
ATOMIC_MASS = {
    'silicon': 28.0855,
    'carbon': 12.011,
    'aluminum': 26.982,
    'iron': 55.845,
    'lead': 207.2,
    'air': 28.97,
    'beryllium': 9.012,
}

# 原子序数 Z
ATOMIC_NUMBER = {
    'silicon': 14,
    'carbon': 6,
    'aluminum': 13,
    'iron': 26,
    'lead': 82,
    'air': 7,
    'beryllium': 4,
}

# ============================================================
# 磁场参数
# ============================================================
B_FIELD_NOMINAL = 2.0           # 螺线管标称磁场 [Tesla]
B_FIELD_UNIFORM_TOLERANCE = 1e-6  # 均匀场容差


# ============================================================
# 相对论运动学函数
# ============================================================
def relativistic_energy(momentum_mev, mass_mev):
    """
    计算相对论总能量
        E = √(p²c² + m²c⁴)    (c = 1 自然单位)

    Parameters
    ----------
    momentum_mev : float
        粒子动量 [MeV/c]
    mass_mev : float
        粒子静止质量 [MeV/c^2]

    Returns
    -------
    float
        总能量 [MeV]
    """
    p2 = max(momentum_mev, 0.0) ** 2
    m2 = max(mass_mev, 0.0) ** 2
    return math.sqrt(p2 + m2)


def relativistic_beta(momentum_mev, mass_mev):
    """
    计算 β = v/c
        β = p / E = p / √(p² + m²)

    Parameters
    ----------
    momentum_mev : float
        粒子动量 [MeV/c]
    mass_mev : float
        粒子静止质量 [MeV/c^2]

    Returns
    -------
    float
        β ∈ [0, 1)
    """
    energy = relativistic_energy(momentum_mev, mass_mev)
    if energy < 1e-15:
        return 0.0
    return min(momentum_mev / energy, 1.0 - 1e-15)


def relativistic_gamma(momentum_mev, mass_mev):
    """
    计算 Lorentz 因子 γ
        γ = E / (mc²) = √(1 + (p/mc)²)

    Parameters
    ----------
    momentum_mev : float
        粒子动量 [MeV/c]
    mass_mev : float
        粒子静止质量 [MeV/c^2]

    Returns
    -------
    float
        γ ≥ 1
    """
    if mass_mev < 1e-15:
        return float('inf')
    return relativistic_energy(momentum_mev, mass_mev) / mass_mev


def relativistic_beta_gamma(momentum_mev, mass_mev):
    """
    计算 βγ = p / (mc)

    Parameters
    ----------
    momentum_mev : float
        粒子动量 [MeV/c]
    mass_mev : float
        粒子静止质量 [MeV/c^2]

    Returns
    -------
    float
        βγ ≥ 0
    """
    if mass_mev < 1e-15:
        return float('inf')
    return momentum_mev / mass_mev


def kinetic_energy(momentum_mev, mass_mev):
    """
    计算相对论动能
        T = E - mc² = √(p² + m²) - m

    Parameters
    ----------
    momentum_mev : float
        粒子动量 [MeV/c]
    mass_mev : float
        粒子静止质量 [MeV/c^2]

    Returns
    -------
    float
        动能 T [MeV]
    """
    return relativistic_energy(momentum_mev, mass_mev) - mass_mev


def momentum_from_kinetic_energy(kinetic_mev, mass_mev):
    """
    从动能反推动量
        p = √(T² + 2Tmc²) / c

    Parameters
    ----------
    kinetic_mev : float
        动能 [MeV]
    mass_mev : float
        粒子静止质量 [MeV/c^2]

    Returns
    -------
    float
        动量 [MeV/c]
    """
    t = max(kinetic_mev, 0.0)
    p2 = t * t + 2.0 * t * mass_mev
    return math.sqrt(max(p2, 0.0))


# ============================================================
# 单位转换
# ============================================================
def gev_to_mev(gev):
    """GeV → MeV"""
    return gev * 1000.0


def mev_to_gev(mev):
    """MeV → GeV"""
    return mev / 1000.0


def tesla_momentum(radius_m, bfield_t):
    """
    由曲率半径和磁场计算动量
        p [GeV/c] = 0.3 × B [T] × r [m]

    这是高能物理中横向动量的基本关系式，
    源自 Lorentz 力与向心力的平衡:
        qvB = γmv²/r  →  p_T = qBr

    Parameters
    ----------
    radius_m : float
        螺旋线曲率半径 [m]
    bfield_t : float
        磁场强度 [T]

    Returns
    -------
    float
        动量 [GeV/c]
    """
    return 0.3 * abs(bfield_t) * abs(radius_m)


def pt_from_curvature(curvature, bfield_t):
    """
    由曲率和磁场计算横向动量
        p_T = 0.3 · B / |κ|

    其中 κ = q/p_T 为 track parameter 中的横向曲率

    Parameters
    ----------
    curvature : float
        横向曲率 κ [1/mm]
    bfield_t : float
        磁场 [T]

    Returns
    -------
    float
        p_T [GeV/c]
    """
    if abs(curvature) < 1e-15:
        return float('inf')
    # κ 的单位为 [1/mm], B 为 [T]
    # p_T [GeV/c] = 0.3 * B / (|κ| * 1e-3) = 300 * B / |κ|
    return 0.3 * abs(bfield_t) / (abs(curvature) * 1e-3)
