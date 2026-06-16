# -*- coding: utf-8 -*-
"""
plasma_parameters.py
====================
Fast Ignition Inertial Confinement Fusion (ICF) 物理参数与等离子体常数.

物理背景:
---------
在 fast ignition 方案 (Tabak et al., Phys. Plasmas 1, 1626, 1994) 中,
超短超强激光脉冲 (>10^20 W/cm^2) 在稠密等离子体临界面产生相对论电子束,
这些电子束穿越日冕等离子体并将能量沉积在稠密芯部, 点燃聚变反应.

核心方程:
---------
1) 能量沉积 PDE (非线性扩散 + 源项):
   ∂u/∂t = ∇·(κ(u)∇u) + S(x,t)
   其中 κ(u) = κ_0 · u^(5/2) 为 Spitzer-Härm 热导率,
   S(x,t) 为电子束能量沉积源项.

2) 相对论电子运动方程 (简化):
   dp/dt = -e·E - ν_ei·p + ξ(t)
   其中 ν_ei 为电子-离子碰撞频率, ξ(t) 为随机力.

3) Bethe  stopping power:
   -dE/dx = (4π·n_e·e^4)/(m_e·v^2) · ln(Λ)
   其中 Λ 为库仑对数.
"""

import math

# ============================================================
# 基础物理常数 (SI 单位制)
# ============================================================
ELECTRON_MASS = 9.1093837015e-31        # m_e  [kg]
ELECTRON_CHARGE = 1.602176634e-19       # e    [C]
SPEED_OF_LIGHT = 2.99792458e8           # c    [m/s]
BOLTZMANN = 1.380649e-23                # k_B  [J/K]
VACUUM_PERMITTIVITY = 8.8541878128e-12  # ε_0  [F/m]
VACUUM_PERMEABILITY = 1.25663706212e-6  # μ_0  [H/m]
PLANCK = 6.62607015e-34                 # h    [J·s]
AVOGADRO = 6.02214076e23                # N_A  [1/mol]
CLASSICAL_ELECTRON_RADIUS = 2.8179403262e-15  # r_e [m]
BOHR_RADIUS = 5.29177210903e-11         # a_0  [m]
RYDBERG_ENERGY = 2.1798723611035e-18    # Ry   [J]

# 电子伏特换算
EV_TO_JOULE = ELECTRON_CHARGE           # 1 eV = e J
KEV_TO_JOULE = 1.0e3 * EV_TO_JOULE
MEV_TO_JOULE = 1.0e6 * EV_TO_JOULE

# ============================================================
# Fast Ignition 等离子体参数
# ============================================================
# 日冕等离子体 (coronal plasma)
CORONA_ELECTRON_DENSITY = 1.0e25        # n_e,corona [1/m^3]  ~ 10 n_c
CORONA_ELECTRON_TEMPERATURE = 5.0e3     # T_e,corona [eV]      ~ 5 keV
CORONA_ION_CHARGE_Z = 10                # Z_eff (CH 靶)
CORONA_ION_MASS_NUMBER = 12.0           # A (碳)

# 稠密芯部 (dense core)
CORE_ELECTRON_DENSITY = 1.0e31          # n_e,core [1/m^3]     ~ 1000 ρ_solid
CORE_ELECTRON_TEMPERATURE = 1.0e4       # T_e,core [eV]        ~ 10 keV
CORE_ION_CHARGE_Z = 1.0                 # Z (DT)
CORE_ION_MASS_NUMBER = 2.5              # A (DT 平均)

# 激光与电子束参数
LASER_INTENSITY = 1.0e20                # I_L [W/m^2]
LASER_WAVELENGTH = 1.05e-6              # λ_L [m]  (1ω Nd:glass)
CRITICAL_DENSITY = (
    ELECTRON_MASS * VACUUM_PERMITTIVITY
    * (2.0 * math.pi * SPEED_OF_LIGHT / LASER_WAVELENGTH) ** 2
    / ELECTRON_CHARGE ** 2
)                                       # n_c [1/m^3]

FAST_ELECTRON_TEMPERATURE = 1.5e6       # T_hot [eV]  ~ 1.5 MeV
FAST_ELECTRON_TOTAL_ENERGY = 20.0e3     # 总能量 [J]   ~ 20 kJ
FAST_ELECTRON_PULSE_DURATION = 10.0e-12 # τ_pulse [s]  ~ 10 ps
FAST_ELECTRON_BEAM_RADIUS = 30.0e-6     # r_beam [m]   ~ 30 μm
FAST_ELECTRON_DIVERGENCE_HALF_ANGLE = math.radians(30.0)  # θ_div [rad]


# ============================================================
# 导出计算函数
# ============================================================
def coulomb_logarithm(n_e, T_e_ev):
    """
    计算库仑对数 ln(Λ).

    公式 (NRL Plasma Formulary):
    ln(Λ) = 23 - ln(n_e^{1/2} · T_e^{-3/2})    (cgs)
          ≈ 24 - ln(n_e^{1/2} [cm^{-3}] / T_e [eV])

    参数:
        n_e   : 电子数密度 [1/m^3]
        T_e_ev: 电子温度 [eV]
    返回:
        ln(Λ) : 库仑对数 (无量纲)
    """
    if T_e_ev <= 0.0 or n_e <= 0.0:
        return 10.0  # 典型下限保护
    n_e_cgs = n_e * 1.0e-6  # 转换到 cm^{-3}
    ln_lambda = 23.0 - math.log(math.sqrt(n_e_cgs) / (T_e_ev ** 1.5))
    return max(ln_lambda, 2.0)


def electron_plasma_frequency(n_e):
    """
    电子等离子体频率:
    ω_pe = sqrt(n_e · e^2 / (ε_0 · m_e))  [rad/s]
    """
    return math.sqrt(n_e * ELECTRON_CHARGE ** 2
                     / (VACUUM_PERMITTIVITY * ELECTRON_MASS))


def electron_thermal_velocity(T_e_ev):
    """
    电子热速度:
    v_th = sqrt(2 · T_e / m_e)
    T_e 以 eV 输入, 内部转换为 J.
    """
    T_e_J = T_e_ev * EV_TO_JOULE
    return math.sqrt(2.0 * T_e_J / ELECTRON_MASS)


def relativistic_gamma(energy_ev):
    """
    相对论 Lorentz 因子:
    γ = 1 + E_kin / (m_e c^2)
    其中 m_e c^2 ≈ 0.511 MeV.
    """
    m_e_c2_ev = ELECTRON_MASS * SPEED_OF_LIGHT ** 2 / ELECTRON_CHARGE
    return 1.0 + energy_ev / m_e_c2_ev


def relativistic_momentum(energy_ev):
    """
    相对论动量:
    p = m_e c sqrt(γ^2 - 1)
    """
    gamma = relativistic_gamma(energy_ev)
    return ELECTRON_MASS * SPEED_OF_LIGHT * math.sqrt(gamma ** 2 - 1.0)


def spitzer_collisional_frequency(n_e, T_e_ev, Z_eff=1.0):
    """
    Spitzer 电子-离子碰撞频率:
    ν_ei = n_e · Z · e^4 · ln(Λ) / (4π · ε_0^2 · m_e^2 · v_th^3)

    参数:
        n_e   : 电子密度 [1/m^3]
        T_e_ev: 电子温度 [eV]
        Z_eff : 有效离子电荷数
    """
    ln_lambda = coulomb_logarithm(n_e, T_e_ev)
    v_th = electron_thermal_velocity(T_e_ev)
    nu_ei = (n_e * Z_eff * ELECTRON_CHARGE ** 4 * ln_lambda /
             (4.0 * math.pi * VACUUM_PERMITTIVITY ** 2
              * ELECTRON_MASS ** 2 * v_th ** 3))
    return nu_ei


def bethe_stopping_power(energy_ev, n_e, Z_eff=1.0):
    """
    Bethe 碰撞能量损失率 (stopping power):
    -dE/dx = (4π · n_e · Z · e^4) / (m_e · v^2) · ln(Λ)

    参数:
        energy_ev: 电子动能 [eV]
        n_e      : 背景电子密度 [1/m^3]
        Z_eff    : 有效离子电荷
    返回:
        dE_dx [eV/m] : 正值, 表示 |dE/dx|
    """
    if energy_ev <= 100.0:
        return 1.0e6  # 低能保护
    gamma = relativistic_gamma(energy_ev)
    beta2 = 1.0 - 1.0 / (gamma ** 2)
    v2 = beta2 * SPEED_OF_LIGHT ** 2
    ln_lambda = coulomb_logarithm(n_e, energy_ev)
    # 转换到 SI, 然后回到 eV/m
    dedx_si = (4.0 * math.pi * n_e * Z_eff * ELECTRON_CHARGE ** 4
               * ln_lambda / (ELECTRON_MASS * v2))
    return dedx_si / ELECTRON_CHARGE


def spitzer_harm_conductivity(T_e_ev, n_e, Z_eff=1.0):
    """
    Spitzer-Härm 热导率系数:
    κ_SH = 1.84e-5 · T_e^{5/2} / (Z · ln Λ)  [W/m/K]  (cgs 换算)

    返回 κ_0 使得 κ(u) = κ_0 · u^{5/2}.
    """
    ln_lambda = coulomb_logarithm(n_e, T_e_ev)
    T_e_J = T_e_ev * EV_TO_JOULE
    # Braginskii 系数 (简化)
    kappa_0 = (1.84e-5 * (T_e_J / BOLTZMANN) ** 2.5
               / (Z_eff * max(ln_lambda, 2.0)))
    return kappa_0


# ============================================================
# 数值计算参数 (fast ignition 小尺度实验)
# ============================================================
# 空间网格
DOMAIN_LENGTH_X = 100.0e-6              # L_x [m]  ~ 100 μm
DOMAIN_LENGTH_Y = 60.0e-6               # L_y [m]  ~ 60 μm
NX = 64                                 # x 方向网格数
NY = 40                                 # y 方向网格数
MESH_ELEMENTS_X = NX - 1
MESH_ELEMENTS_Y = NY - 1

# 时间推进
TOTAL_TIME = 5.0e-12                    # T_total [s]  ~ 5 ps
CFL_NUMBER = 0.4                        # CFL 安全系数
MAX_TIME_STEPS = 200                    # 最大时间步数

# 高阶有限差分
FD_ORDER = 4                            # 空间精度阶数 (4阶中心差分)
FD_STENCIL_HALF_WIDTH = FD_ORDER // 2   # 半带宽

# 电子束采样
NUM_BEAM_ELECTRONS = 256                # 蒙特卡罗电子宏粒子数
BEAM_ENERGY_SPREAD_FWHM = 0.3           # 能散 ΔE/E (FWHM)

# 稳定性分析
VON_NEUMANNA_MODES = 32                 # von Neumann 分析模式数

# 收敛性测试
CONVERGENCE_REFINEMENT_LEVELS = 4       # 网格细化层次


def get_plasma_config():
    """返回完整的等离子体配置字典."""
    config = {
        # 物理
        "corona_ne": CORONA_ELECTRON_DENSITY,
        "corona_Te": CORONA_ELECTRON_TEMPERATURE,
        "core_ne": CORE_ELECTRON_DENSITY,
        "core_Te": CORE_ELECTRON_TEMPERATURE,
        "Z_corona": CORONA_ION_CHARGE_Z,
        "Z_core": CORE_ION_CHARGE_Z,
        "n_critical": CRITICAL_DENSITY,
        "T_hot_eV": FAST_ELECTRON_TEMPERATURE,
        "E_total_J": FAST_ELECTRON_TOTAL_ENERGY,
        "tau_pulse_s": FAST_ELECTRON_PULSE_DURATION,
        "r_beam_m": FAST_ELECTRON_BEAM_RADIUS,
        "theta_div_rad": FAST_ELECTRON_DIVERGENCE_HALF_ANGLE,
        # 数值
        "Lx_m": DOMAIN_LENGTH_X,
        "Ly_m": DOMAIN_LENGTH_Y,
        "nx": NX,
        "ny": NY,
        "T_total_s": TOTAL_TIME,
        "cfl": CFL_NUMBER,
        "max_steps": MAX_TIME_STEPS,
        "fd_order": FD_ORDER,
        "stencil_hw": FD_STENCIL_HALF_WIDTH,
        "n_beam": NUM_BEAM_ELECTRONS,
        "energy_spread": BEAM_ENERGY_SPREAD_FWHM,
        "vn_modes": VON_NEUMANNA_MODES,
        "refine_levels": CONVERGENCE_REFINEMENT_LEVELS,
    }
    return config


def print_plasma_header():
    """打印等离子体参数头."""
    print("=" * 72)
    print("Fast Ignition ICF : 电子束能量沉积高阶有限差分与稳定性分析")
    print("=" * 72)
    print("  临界密度 n_c      = {:.3e} /m^3".format(CRITICAL_DENSITY))
    print("  日冕密度 n_e      = {:.3e} /m^3".format(CORONA_ELECTRON_DENSITY))
    print("  日冕温度 T_e      = {:.3e} eV".format(CORONA_ELECTRON_TEMPERATURE))
    print("  稠密芯密度        = {:.3e} /m^3".format(CORE_ELECTRON_DENSITY))
    print("  快电子温度        = {:.3e} eV  (~ {:.2f} MeV)".format(
        FAST_ELECTRON_TEMPERATURE,
        FAST_ELECTRON_TEMPERATURE / 1.0e6))
    print("  库仑对数 (日冕)   = {:.3f}".format(
        coulomb_logarithm(CORONA_ELECTRON_DENSITY, CORONA_ELECTRON_TEMPERATURE)))
    print("  等离子体频率 ω_pe = {:.3e} rad/s".format(
        electron_plasma_frequency(CORONA_ELECTRON_DENSITY)))
    print("  电子热速度 v_th   = {:.3e} m/s".format(
        electron_thermal_velocity(CORONA_ELECTRON_TEMPERATURE)))
    print("  Spitzer ν_ei      = {:.3e} /s".format(
        spitzer_collisional_frequency(
            CORONA_ELECTRON_DENSITY, CORONA_ELECTRON_TEMPERATURE,
            CORONA_ION_CHARGE_Z)))
    print("  Bethe -dE/dx      = {:.3e} eV/m".format(
        bethe_stopping_power(FAST_ELECTRON_TEMPERATURE,
                             CORONA_ELECTRON_DENSITY,
                             CORONA_ION_CHARGE_Z)))
    print("  Spitzer κ_SH      = {:.3e} W/m/K".format(
        spitzer_harm_conductivity(
            CORONA_ELECTRON_TEMPERATURE, CORONA_ELECTRON_DENSITY,
            CORONA_ION_CHARGE_Z)))
    print("=" * 72)
