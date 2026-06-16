"""
plasma_physics.py - 核心物理常数、碰撞频率、输运系数与状态方程

本模块封装边界等离子体物理的所有基本公式，包括：
  - 国际单位制基本物理常数
  - 电子-离子碰撞频率（Coulomb对数、Spitzer电阻率）
  - 平行热传导系数（Spitzer-Harm）
  - 垂直扩散系数（经典扩散、Bohm扩散）
  - 偏滤器靶板热负荷公式（鞘层透射系数、声速、再循环）
  - 两点模型（two-point model）上下游关系
  - Bohm判据与鞘层条件

来源项目映射：
  - 1109_marekgluza_Fidelity_witnesses_example → 保真度度量方法借鉴
  - 832_ode_sweep_parfor → 参数扫描框架
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 基本物理常数 (SI单位制)
# =============================================================================
ELECTRON_CHARGE = 1.602176634e-19       # e [C]
ELECTRON_MASS = 9.1093837015e-31        # m_e [kg]
DEUTERIUM_MASS = 3.3435837724e-27       # m_D [kg] (2*m_p)
PROTON_MASS = 1.67262192369e-27         # m_p [kg]
VACUUM_PERMITTIVITY = 8.8541878128e-12  # epsilon_0 [F/m]
BOLTZMANN = 1.380649e-23                # k_B [J/K]
PLANCK = 6.62607015e-34                 # h [J*s]
SPEED_OF_LIGHT = 2.99792458e8           # c [m/s]
EV_TO_JOULE = 1.602176634e-19           # 1 eV -> J


# =============================================================================
# Coulomb对数 (Trubnikov公式)
# =============================================================================
def coulomb_logarithm(n_e: float, T_e_eV: float) -> float:
    """
    计算等离子体Coulomb对数 ln(Lambda)

    对于T_e > 10 eV的热等离子体：
      ln(Lambda) = 24 - ln(sqrt(n_e [cm^-3]) / T_e [eV])

    对于T_e < 10 eV的冷等离子体：
      ln(Lambda) = 23 - ln(sqrt(n_e [cm^-3]) * T_e [eV]^(-3/2))

    物理约束：ln(Lambda) >= 2 (弱耦合条件)

    参数:
        n_e: 电子密度 [m^-3]
        T_e_eV: 电子温度 [eV]

    返回:
        Coulomb对数 (无量纲)
    """
    n_e_cm3 = n_e * 1.0e-6  # 转换为 cm^-3

    if T_e_eV > 10.0:
        ln_Lambda = 24.0 - np.log(np.sqrt(n_e_cm3) / T_e_eV)
    elif T_e_eV > 0.1:
        ln_Lambda = 23.0 - np.log(np.sqrt(n_e_cm3) * T_e_eV**(-1.5))
    else:
        ln_Lambda = 10.0  # 低温极限，强耦合修正

    # 物理约束：Coulomb对数必须为正且足够大
    ln_Lambda = max(ln_Lambda, 2.0)
    return ln_Lambda


# =============================================================================
# 电子热速度
# =============================================================================
def electron_thermal_velocity(T_e_eV: float) -> float:
    """
    电子热速度: v_te = sqrt(2 * T_e / m_e)

    参数:
        T_e_eV: 电子温度 [eV]
    返回:
        v_te [m/s]
    """
    T_e_J = T_e_eV * EV_TO_JOULE
    return np.sqrt(2.0 * T_e_J / ELECTRON_MASS)


# =============================================================================
# 离子热速度与声速
# =============================================================================
def ion_thermal_velocity(T_i_eV: float, m_i: float = DEUTERIUM_MASS) -> float:
    """
    离子热速度: v_ti = sqrt(2 * T_i / m_i)

    参数:
        T_i_eV: 离子温度 [eV]
        m_i: 离子质量 [kg]，默认为氘
    返回:
        v_ti [m/s]
    """
    T_i_J = T_i_eV * EV_TO_JOULE
    return np.sqrt(2.0 * T_i_J / m_i)


def sound_speed(T_e_eV: float, T_i_eV: float, m_i: float = DEUTERIUM_MASS) -> float:
    """
    离子声速（Bohm速度）: c_s = sqrt((T_e + T_i) / m_i)

    这是平行流必须满足的Bohm判据的临界速度

    参数:
        T_e_eV: 电子温度 [eV]
        T_i_eV: 离子温度 [eV]
        m_i: 离子质量 [kg]
    返回:
        c_s [m/s]
    """
    T_total_J = (T_e_eV + T_i_eV) * EV_TO_JOULE
    return np.sqrt(T_total_J / m_i)


# =============================================================================
# 电子-离子碰撞频率 (Braginskii)
# =============================================================================
def electron_ion_collision_frequency(n_e: float, T_e_eV: float) -> float:
    """
    电子-离子碰撞频率 (Braginskii 1958):

      nu_ei = n_e * e^4 * ln(Lambda) /
              (4*pi*epsilon_0^2 * m_e^2 * v_te^3)

    等价于:
      nu_ei = 3.44e-11 * n_e[cm^-3] * ln(Lambda) / T_e[eV]^(3/2) [s^-1]

    参数:
        n_e: 电子密度 [m^-3]
        T_e_eV: 电子温度 [eV]
    返回:
        nu_ei [s^-1]
    """
    if T_e_eV < 1.0e-3:
        return 1.0e30  # 防止除零

    ln_Lambda = coulomb_logarithm(n_e, T_e_eV)
    v_te = electron_thermal_velocity(T_e_eV)

    # 标准公式
    e4 = ELECTRON_CHARGE**4
    denom = 4.0 * np.pi * VACUUM_PERMITTIVITY**2 * ELECTRON_MASS**2 * v_te**3
    nu_ei = n_e * e4 * ln_Lambda / denom

    return max(nu_ei, 1.0e-10)


def ion_ion_collision_frequency(n_i: float, T_i_eV: float,
                                 m_i: float = DEUTERIUM_MASS, Z_i: float = 1.0) -> float:
    """
    离子-离子碰撞频率:

      nu_ii = 4*sqrt(pi)/3 * n_i * Z_i^4 * e^4 * ln(Lambda) /
              (4*pi*epsilon_0)^2 * m_i^(1/2) * (2*T_i)^(3/2)

    参数:
        n_i: 离子密度 [m^-3]
        T_i_eV: 离子温度 [eV]
        m_i: 离子质量 [kg]
        Z_i: 离子电荷数
    返回:
        nu_ii [s^-1]
    """
    if T_i_eV < 1.0e-3:
        return 1.0e30

    ln_Lambda = coulomb_logarithm(n_i, T_i_eV)
    T_i_J = T_i_eV * EV_TO_JOULE

    # Braginskii离子碰撞频率
    prefactor = 4.0 * np.sqrt(np.pi) / 3.0
    numerator = prefactor * n_i * (Z_i * ELECTRON_CHARGE)**4 * ln_Lambda
    epsilon_factor = (4.0 * np.pi * VACUUM_PERMITTIVITY)**2
    denominator = epsilon_factor * np.sqrt(m_i) * (2.0 * T_i_J)**1.5

    nu_ii = numerator / denominator
    return max(nu_ii, 1.0e-10)


# =============================================================================
# Spitzer-Harm平行热传导系数
# =============================================================================
def spitzer_harm_conductivity(n_e: float, T_e_eV: float) -> float:
    """
    Spitzer-Harm平行电子热传导系数 (W/m/eV):

      kappa_|| = 3.2 * n_e * T_e / (m_e * nu_ei) * (k_B / e)

    或者等价表达（更常用的形式）：
      kappa_|| = 3.16 * T_e^(5/2) / (e^2 * m_e^(1/2) * ln(Lambda) * Lambda_e)

    其中 Lambda_e = 12*pi^(3/2) * (epsilon_0 * T_e / (n_e^(1/2) * e^3))^(1/2)
    是电子平均自由程。

    简化形式（数值系数）：
      kappa_|| ≈ 2.56e3 * T_e[eV]^(5/2) / ln(Lambda) [W/(m*eV)]

    参数:
        n_e: 电子密度 [m^-3]
        T_e_eV: 电子温度 [eV]
    返回:
        kappa_|| [W/(m*eV)]
    """
    if T_e_eV < 1.0e-3:
        return 1.0e-30

    ln_Lambda = coulomb_logarithm(n_e, T_e_eV)

    # 使用Bragskii系数 delta_e = 0.16 (Z=1)
    # kappa_|| = delta_e * n_e * k_B * T_e * tau_e / m_e
    # tau_e = 3*sqrt(m_e) * (k_B*T_e)^(3/2) / (4*sqrt(2*pi) * n_e * e^4 * ln_Lambda / (4*pi*eps0)^2)

    nu_ei = electron_ion_collision_frequency(n_e, T_e_eV)
    tau_e = 1.0 / nu_ei

    T_e_J = T_e_eV * EV_TO_JOULE
    # kappa_|| = 3.2 * n_e * k_B^2 * T_e * tau_e / m_e [W/m/K]
    # 转换为 W/m/eV: kappa_|| [W/m/eV] = kappa_|| [W/m/K] / e
    kappa_SI = 3.2 * n_e * BOLTZMANN**2 * T_e_J * tau_e / ELECTRON_MASS
    kappa_eV = kappa_SI / ELECTRON_CHARGE  # 转为 per eV

    return max(kappa_eV, 1.0e-30)


# =============================================================================
# 离子Larmor半径与垂直扩散
# =============================================================================
def ion_larmor_radius(T_i_eV: float, B: float, m_i: float = DEUTERIUM_MASS) -> float:
    """
    离子Larmor半径（回旋半径）:

      rho_i = sqrt(m_i * T_i) / (e * B) = v_ti / (2 * omega_ci)

    其中 omega_ci = e*B/m_i 是离子回旋频率

    参数:
        T_i_eV: 离子温度 [eV]
        B: 磁场强度 [T]
        m_i: 离子质量 [kg]
    返回:
        rho_i [m]
    """
    if B < 1.0e-10:
        return 1.0  # 无磁场极限

    T_i_J = T_i_eV * EV_TO_JOULE
    omega_ci = ELECTRON_CHARGE * B / m_i
    v_ti = np.sqrt(2.0 * T_i_J / m_i)
    rho_i = v_ti / omega_ci
    return rho_i


def classical_perpendicular_diffusion(n_i: float, T_i_eV: float, B: float,
                                       m_i: float = DEUTERIUM_MASS) -> float:
    """
    经典垂直扩散系数:

      D_⊥^classical = rho_i^2 * nu_ii

    在强磁场下，这是一个非常小的量（~10^-4 m²/s），
    远小于实验观测到的反常输运

    参数:
        n_i: 离子密度 [m^-3]
        T_i_eV: 离子温度 [eV]
        B: 磁场强度 [T]
        m_i: 离子质量 [kg]
    返回:
        D_⊥ [m²/s]
    """
    rho_i = ion_larmor_radius(T_i_eV, B, m_i)
    nu_ii = ion_ion_collision_frequency(n_i, T_i_eV, m_i)
    D_perp = rho_i**2 * nu_ii
    return D_perp


def bohm_diffusion(T_e_eV: float, B: float) -> float:
    """
    Bohm扩散系数（经验公式）:

      D_Bohm = T_e / (16 * e * B)

    这是磁约束等离子体中观测到的最大扩散率上限

    参数:
        T_e_eV: 电子温度 [eV]
        B: 磁场强度 [T]
    返回:
        D_Bohm [m²/s]
    """
    if B < 1.0e-10:
        return 1.0e6  # 无磁场极限

    D_Bohm = T_e_eV / (16.0 * B)  # 自然得到 m²/s
    return D_Bohm


def gyro_bohm_diffusion(T_e_eV: float, B: float, a: float = 0.5) -> float:
    """
    Gyro-Bohm扩散系数（漂移波湍流标度）:

      D_gB = rho_s^2 * c_s / a = rho_s * v_ti / (2*a)

    其中 rho_s = c_s / omega_ci 是声速Larmor半径
    a 是特征长度尺度（如小半径）

    参数:
        T_e_eV: 电子温度 [eV]
        B: 磁场强度 [T]
        a: 特征长度 [m]
    返回:
        D_gB [m²/s]
    """
    c_s = sound_speed(T_e_eV, T_e_eV)
    omega_ci = ELECTRON_CHARGE * B / DEUTERIUM_MASS
    rho_s = c_s / omega_ci
    D_gB = rho_s**2 * c_s / a
    return D_gB


# =============================================================================
# 鞘层边界条件与偏滤器靶板热负荷
# =============================================================================
def sheath_heat_transmission(Z_i: float = 1.0, gamma_se: float = 0.0) -> float:
    """
    鞘层热透射系数:

      gamma_sheath = 2 + delta_sec + (m_e/m_i)^(1/2) * ln(m_i/(2*pi*m_e))^(1/2)
                     + (1 + T_i/T_e) * (1 + gamma_se)

    对于冷鞘层（T_i = T_e, Z=1, 二次电子发射gamma_se=0）:
      gamma_sheath ≈ 7.0 ~ 7.5

    参数:
        Z_i: 离子电荷数
        gamma_se: 二次电子发射系数
    返回:
        gamma_sheath (无量纲)
    """
    mass_ratio = ELECTRON_MASS / DEUTERIUM_MASS

    # 鞘层电位降对应的能量
    # Phi_sheath = T_e * (1 + gamma_se) / 2 (对于Z=1)
    potential_drop = 0.5 * (1.0 + gamma_se)

    # 平行电子热流贡献
    electron_part = np.sqrt(mass_ratio) * np.log(1.0 / (2.0 * np.pi * mass_ratio + 1e-30))
    electron_part = max(electron_part, 0.0)

    # 离子动能贡献
    ion_part = 2.0 + potential_drop  # 2*Te (热流) + Te*(1+gamma_se)/2 (势能)

    gamma_total = ion_part + electron_part

    return gamma_total


def divertor_target_heat_flux(n_target: float, T_target_eV: float,
                                T_e_upstream_eV: float = 100.0,
                                Z_i: float = 1.0) -> float:
    """
    偏滤器靶板热负荷 (W/m²):

      q_target = gamma_sheath * n_target * T_target * c_s

    其中 c_s = sqrt((T_e + T_i)/m_i) 是声速

    参数:
        n_target: 靶板处等离子体密度 [m^-3]
        T_target_eV: 靶板处电子温度 [eV]
        T_e_upstream_eV: 上游电子温度 [eV]
        Z_i: 离子电荷数
    返回:
        q_target [W/m²]
    """
    gamma = sheath_heat_transmission(Z_i)
    c_s = sound_speed(T_target_eV, T_target_eV)

    T_target_J = T_target_eV * EV_TO_JOULE
    q_target = gamma * n_target * T_target_J * c_s

    return q_target


# =============================================================================
# 两点模型 (Two-Point Model)
# =============================================================================
def two_point_model(q_parallel: float, n_upstream: float, T_upstream_eV: float,
                     L_parallel: float, n_e: float = 1.0e19,
                     recycling_R: float = 0.9) -> dict:
    """
    SOL两点模型解析求解:

    上游-靶板关系：
      q_|| = kappa_0 * T_u^(7/2) / L_|| (纯传导极限)
      或 q_|| = gamma * n_t * T_t * c_s,t (靶板热流)

    压力平衡: n_t * T_t = n_u * T_u * (1 + R) / 2
    粒子守恒: Gamma_target = R * n_u * c_s,u

    参数:
        q_parallel: 平行热流密度 [W/m²]
        n_upstream: 上游密度 [m^-3]
        T_upstream_eV: 上游温度 [eV]
        L_parallel: 平行连接长度 [m]
        n_e: 参考密度 [m^-3]
        recycling_R: 再循环系数 (0-1)

    返回:
        dict包含靶板温度、密度、热负荷等
    """
    # Spitzer-Harm热传导系数 (前因子)
    kappa_0 = 2.56e3  # W/(m*eV^(7/2)) 近似

    # 纯传导极限: q_|| = 2/7 * kappa_0 * T_u^(7/2) / L_||
    T_u_J = T_upstream_eV * EV_TO_JOULE
    q_conduction = (2.0 / 7.0) * kappa_0 * T_upstream_eV**3.5 / L_parallel

    # 靶板温度估计（迭代求解能量平衡）
    # 简化：T_t ≈ T_u * (1 - q_||/q_conduction)
    ratio = min(q_parallel / (q_conduction + 1.0e-30), 0.99)
    T_target_eV = T_upstream_eV * (1.0 - ratio)**(2.0 / 7.0)
    T_target_eV = max(T_target_eV, 0.5)  # 防止过低

    # 靶板密度（压力守恒 + 再循环）
    pressure_factor = (1.0 + recycling_R) / 2.0
    n_target = n_upstream * (T_upstream_eV / T_target_eV) * pressure_factor

    # 靶板热负荷
    q_target = divertor_target_heat_flux(n_target, T_target_eV, T_upstream_eV)

    # 声速马赫数
    c_s = sound_speed(T_target_eV, T_target_eV)

    return {
        'T_target_eV': T_target_eV,
        'n_target': n_target,
        'q_target_Wm2': q_target,
        'q_parallel_input': q_parallel,
        'q_conduction_limit': q_conduction,
        'c_s': c_s,
        'L_parallel': L_parallel,
        'recycling_R': recycling_R,
        'T_upstream_eV': T_upstream_eV,
        'n_upstream': n_upstream,
    }


# =============================================================================
# 等离子体β参数
# =============================================================================
def plasma_beta(n_e: float, T_e_eV: float, B: float) -> float:
    """
    等离子体β参数（热压/磁压之比）:

      beta = 2 * mu_0 * n_e * (T_e + T_i) / B^2

    参数:
        n_e: 电子密度 [m^-3]
        T_e_eV: 电子温度 [eV]
        B: 磁场强度 [T]
    返回:
        beta (无量纲)
    """
    mu_0 = 4.0 * np.pi * 1.0e-7  # [H/m]
    T_total_J = 2.0 * T_e_eV * EV_TO_JOULE  # T_e + T_i ≈ 2*T_e
    p = n_e * T_total_J  # 总热压 [Pa]
    p_mag = B**2 / (2.0 * mu_0)  # 磁压 [Pa]
    beta = p / (p_mag + 1.0e-30)
    return beta


# =============================================================================
# Debye长度与等离子体频率
# =============================================================================
def debye_length(n_e: float, T_e_eV: float) -> float:
    """
    Debye屏蔽长度: lambda_D = sqrt(epsilon_0 * T_e / (n_e * e^2))
    """
    T_e_J = T_e_eV * EV_TO_JOULE
    return np.sqrt(VACUUM_PERMITTIVITY * T_e_J / (n_e * ELECTRON_CHARGE**2 + 1.0e-60))


def electron_plasma_frequency(n_e: float) -> float:
    """
    电子等离子体频率: omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
    """
    return np.sqrt(n_e * ELECTRON_CHARGE**2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS))


# =============================================================================
# 保真度度量（从量子保真度思想借鉴到等离子体模拟误差度量）
# =============================================================================
def plasma_state_fidelity(state_computed: np.ndarray,
                           state_reference: np.ndarray) -> float:
    """
    等离子体状态保真度度量:

      F = 1 - ||u_num - u_ref||^2 / (2 * ||u_ref||^2)

    借鉴自量子保真度witness的思想，用于衡量数值解
    相对于解析解或高精度参考解的接近程度

    F = 1.0 表示完美匹配
    F = 0.0 表示完全偏离

    参数:
        state_computed: 计算得到的等离子体状态向量
        state_reference: 参考状态向量
    返回:
        保真度 F ∈ (-inf, 1.0]
    """
    diff = state_computed - state_reference
    norm_ref = np.sum(state_reference**2)

    if norm_ref < 1.0e-30:
        return 1.0 if np.sum(diff**2) < 1.0e-30 else 0.0

    fidelity = 1.0 - np.sum(diff**2) / (2.0 * norm_ref)
    return float(fidelity)


def L2_error_norm(field_computed: np.ndarray, field_exact: np.ndarray,
                   dx: float = 1.0) -> float:
    """
    L2误差范数: ||e||_2 = sqrt(sum((u_h - u)^2 * dx))
    """
    diff = field_computed - field_exact
    return np.sqrt(np.sum(diff**2) * dx)


def H1_seminorm(field_computed: np.ndarray, field_exact: np.ndarray,
                 dx: float = 1.0) -> float:
    """
    H1半范数: |e|_H1 = sqrt(sum((de/dx)^2 * dx))
    衡量梯度的误差
    """
    diff = field_computed - field_exact
    grad_diff = np.gradient(diff, dx)
    return np.sqrt(np.sum(grad_diff**2) * dx)


# =============================================================================
# 源项模型
# =============================================================================
def volumetric_source(s: float, L: float, S_0: float = 1.0e20) -> float:
    """
    体积源项（沿场线方向的粒子/能量注入）:

      S(s) = S_0 * exp(-(s - L/2)^2 / (2*sigma^2))

    其中sigma = L/6，模拟中心注入

    参数:
        s: 沿场线位置 [m]
        L: 总连接长度 [m]
        S_0: 源项幅值
    返回:
        S(s) [m^-3 s^-1]
    """
    sigma = L / 6.0
    center = L / 2.0
    return S_0 * np.exp(-(s - center)**2 / (2.0 * sigma**2 + 1.0e-30))


def neutral_fuel_rate(T_e_eV: float, n_e: float, E_ion: float = 13.6) -> float:
    """
    中性粒子电离速率（ADAS参数化）:

      <sigma*v>_ion = S_0 * T_e^alpha * exp(-E_ion/T_e)

    简化参数化用于氘的电离

    参数:
        T_e_eV: 电子温度 [eV]
        n_e: 电子密度 [m^-3]
        E_ion: 电离能 [eV]（氘=13.6 eV）
    返回:
        电离速率 [s^-1]
    """
    if T_e_eV < 0.1:
        return 0.0

    S_0 = 2.0e-14  # m^3/s
    alpha = 0.5
    rate = S_0 * T_e_eV**alpha * np.exp(-E_ion / T_e_eV)
    return rate * n_e
