"""
constants.py
物理常数与中微子振荡标准参数库

本模块包含中微子物理所需的所有基本常数、PMNS 矩阵参数以及
质量平方差。所有数值均基于 PDG 2024 推荐值。
"""

import numpy as np

# =============================================================================
# 基本物理常数 (SI 与粒子物理常用单位)
# =============================================================================

# 费米耦合常数 G_F [GeV^{-2}]
GF = 1.1663787e-5

# 电子伏特转换为焦耳 [J/eV]
EV_TO_JOULE = 1.602176634e-19

# 千米转换为电子伏特倒数 [eV^{-1}]
KM_TO_EV_INV = 5.067730889e9

# 电子平均数密度在标准岩石中的近似值 [cm^{-3}]
# 地球地壳典型值: 2.6 g/cm^3 -> N_e ≈ 1.3 * N_A / cm^3 * (Z/A)
EARTH_CRUST_NE = 6.02214076e23 * 2.6 * 0.5 / 14.0  # 约 5.6e23 cm^{-3}

# 物质势换算因子: V = sqrt(2) * G_F * N_e
# 单位: GeV, 需要将 N_e [cm^{-3}] 转换为 [GeV^3]
# 1 cm^{-3} = (197.3269804e-7 GeV)^3 * (1e-2 m)^3 / (ħc)^3
CM3_TO_GEV3 = (1.973269804e-14) ** 3  # cm^3 in GeV^{-3}
MATTER_POTENTIAL_FACTOR = np.sqrt(2.0) * GF  # [GeV^{-2}]

# =============================================================================
# PMNS 矩阵参数 (PDG 2024, 3σ 范围中值)
# =============================================================================

# 混合角 (弧度)
THETA_12 = np.deg2rad(33.45)   # θ_12, 太阳混合角
THETA_23 = np.deg2rad(47.7)    # θ_23, 大气混合角
THETA_13 = np.deg2rad(8.62)    # θ_13, 反应堆混合角

# CP 破坏相位 (弧度)
DELTA_CP = np.deg2rad(234.0)   # δ_CP

# 质量平方差 [eV^2]
# Δm²₂₁ = m²₂ - m²₁  (太阳)
DELTA_M2_21 = 7.42e-5          # [eV^2]

# Δm²₃₁ = m²₃ - m²₁  (大气)
DELTA_M2_31 = 2.510e-3         # [eV^2]  (Normal Hierarchy, NH)
DELTA_M2_31_IH = -2.490e-3     # [eV^2]  (Inverted Hierarchy, IH)

# =============================================================================
# 标准模型粒子质量 [GeV/c^2]
# =============================================================================

MASS_ELECTRON = 0.5109989461e-3  # 电子质量
MASS_MUON = 105.6583745e-3       # μ子质量
MASS_TAU = 1776.86e-3            # τ子质量

# =============================================================================
# 数值积分默认参数
# =============================================================================

DEFAULT_QUAD_NX = 64
DEFAULT_QUAD_NY = 64
DEFAULT_MC_SAMPLES = 100000
DEFAULT_ODE_STEPS = 2000

# =============================================================================
# 地球物质密度模型参数 (PREM 简化模型)
# =============================================================================

EARTH_RADIUS_KM = 6371.0
CORE_RADIUS_KM = 3480.0

# 地壳密度 [g/cm^3]
DENSITY_CRUST = 2.7
# 地幔密度 [g/cm^3]
DENSITY_MANTLE = 5.5
# 外核密度 [g/cm^3]
DENSITY_OUTER_CORE = 11.0
# 内核密度 [g/cm^3]
DENSITY_INNER_CORE = 13.0


def get_prem_density(radius_ratio):
    """
    返回 PREM (Preliminary Reference Earth Model) 简化密度剖面。

    参数:
        radius_ratio: r / R_earth, 0 <= radius_ratio <= 1

    返回:
        density: 密度 [g/cm^3]

    PREM 模型将地球分为四层:
        - 内核: 0 <= r < 1221 km
        - 外核: 1221 <= r < 3480 km
        - 地幔: 3480 <= r < 5701 km
        - 地壳: 5701 <= r <= 6371 km

    公式:
        ρ(r) = ρ_0 * [1 - a*(r/R) - b*(r/R)^2 - c*(r/R)^3]
    """
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")

    r_km = radius_ratio * EARTH_RADIUS_KM

    if r_km < 1221.0:
        # 内核
        return 13.0885 - 8.8381 * radius_ratio ** 2
    elif r_km < 3480.0:
        # 外核
        return 12.5815 - 1.2638 * radius_ratio - 3.6426 * radius_ratio ** 2 \
               - 5.5281 * radius_ratio ** 3
    elif r_km < 5701.0:
        # 下地幔
        return 7.9565 - 6.4761 * radius_ratio + 5.5283 * radius_ratio ** 2 \
               - 3.0807 * radius_ratio ** 3
    elif r_km < 5771.0:
        # 上地幔 (低速度区)
        return 5.3197 - 1.4836 * radius_ratio
    elif r_km < 5971.0:
        # 上地幔
        return 11.2494 - 8.0298 * radius_ratio
    elif r_km < 6151.0:
        # 上地幔过渡带
        return 7.1089 - 3.8045 * radius_ratio
    else:
        # 地壳
        return 2.6910 + 0.6924 * radius_ratio


def electron_fraction(radius_ratio):
    """
    电子丰度 Y_e = N_e / N_b (重子数密度比)。
    在地球物质中, 典型值 Y_e ≈ 0.5 (大致 Z/A ≈ 1/2)。
    """
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")
    # 内核中电子丰度略低 (铁核), 地壳中略高
    return 0.465 + 0.04 * radius_ratio


def matter_potential_eV(radius_ratio, energy_gev=None):
    """
    计算中微子在地球物质中传播时的有效物质势 V [eV]。

    在标准模型中, 电子中微子通过带电流 (CC) 与电子发生 forward scattering,
    产生一个额外的有效势:

        V_CC = sqrt(2) * G_F * N_e

    其中 N_e 为电子数密度。所有味的中微子还受到中性流 (NC) 散射的贡献,
    但 NC 贡献对所有味相同, 只产生一个共同的相位因子, 不影响振荡。

    参数:
        radius_ratio: r / R_earth
        energy_gev:   中微子能量 [GeV] (用于计算无量纲参数, 可选)

    返回:
        V: 物质势 [eV]
    """
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")

    rho = get_prem_density(radius_ratio)  # [g/cm^3]
    ye = electron_fraction(radius_ratio)

    # 电子数密度 N_e = ρ * Y_e * N_A / (A/Z) ≈ ρ * Y_e * N_A / 2
    # 更直接: N_e = ρ [g/cm^3] * N_A [1/mol] * Y_e / m_u [g/mol]
    # 对于普通物质, 平均摩尔质量 ≈ 2 * m_u / Y_e
    # 简化为: N_e ≈ ρ * N_A * Y_e / (1e-3 kg/mol * 1e6 cm^3/m^3)
    # 使用 g/cm^3: N_e [cm^{-3}] = ρ [g/cm^3] * N_A * Y_e / (平均原子量 [g/mol])
    # 对于岩石, 平均原子量 ≈ 20-30 g/mol, 取约 20 g/mol (硅氧为主)
    avg_molar_mass = 20.0  # [g/mol]
    avogadro = 6.02214076e23
    ne_cm3 = rho * avogadro * ye / avg_molar_mass  # [cm^{-3}]

    # 将 N_e [cm^{-3}] -> [GeV^3]
    # 1 cm = 1/(197.3269804e-7) GeV^{-1}
    # 1 cm^{-3} = (197.3269804e-7 GeV)^3 = (1.973e-14)^3 GeV^3
    hbarc = 0.1973269804  # GeV * fm
    hbarc_cm = hbarc * 1e-13  # GeV * cm
    ne_gev3 = ne_cm3 * hbarc_cm ** 3

    # V = sqrt(2) * G_F * N_e [GeV]
    v_gev = np.sqrt(2.0) * GF * ne_gev3

    # 转换为 eV
    v_ev = v_gev * 1e9

    return v_ev


def get_mass_squared_differences(hierarchy='normal'):
    """
    返回质量平方差向量 [Δm²₂₁, Δm²₃₁] (单位 eV²)。

    参数:
        hierarchy: 'normal' (NH) 或 'inverted' (IH)

    Normal Hierarchy (NH):
        m₁ < m₂ < m₃
        Δm²₂₁ = m²₂ - m²₁ > 0
        Δm²₃₁ = m²₃ - m²₁ > 0

    Inverted Hierarchy (IH):
        m₃ < m₁ < m₂
        Δm²₂₁ = m²₂ - m²₁ > 0
        Δm²₃₁ = m²₃ - m²₁ < 0
    """
    hierarchy = hierarchy.lower()
    if hierarchy == 'normal':
        return np.array([DELTA_M2_21, DELTA_M2_31], dtype=np.float64)
    elif hierarchy == 'inverted':
        return np.array([DELTA_M2_21, DELTA_M2_31_IH], dtype=np.float64)
    else:
        raise ValueError("hierarchy must be 'normal' or 'inverted'")


def get_pmns_angles():
    """
    返回 PMNS 混合角 [θ₁₂, θ₂₃, θ₁₃] (弧度)。
    """
    return np.array([THETA_12, THETA_23, THETA_13], dtype=np.float64)


def get_cp_phase():
    """
    返回 CP 破坏相位 δ_CP (弧度)。
    """
    return float(DELTA_CP)
