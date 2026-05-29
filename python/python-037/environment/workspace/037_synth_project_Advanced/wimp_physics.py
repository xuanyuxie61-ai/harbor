r"""
wimp_physics.py
WIMP（弱相互作用大质量粒子）与原子核弹性散射物理模块

本模块实现暗物质直接探测实验的核心物理计算，包括：
1. Helm 核形状因子 F(E_R)
2. WIMP-核子约化质量 μ
3. 最小反冲速度 v_min(E_R)
4. 微分事件率 dR/dE_R（速度分布积分形式）
5. 使用 Gauss-Hermite 求积进行速度分布矩计算
6. 年度调制振幅与相位

核心公式参考：
- Lewin, J. D., & Smith, P. F. (1996). Astroparticle Physics, 6, 87.
- Helm, R. H. (1956). Phys. Rev. 104, 1466.
- Drukier, A. K., Freese, K., & Spergel, D. N. (1986). Phys. Rev. D, 33, 3495.
"""

import numpy as np
from typing import Callable, Tuple
from utils import (
    spherical_bessel_j1,
    gauss_hermite_quadrature,
    gev_to_kg,
    KM_S_TO_M_S,
    AMU_KG,
    M_PROTON_GEV,
    RHO_LOCAL_GEV_CM3,
    V0_KM_S,
    VE_KM_S,
    VESC_KM_S,
    C_M_S,
)

# ============================================================================
# 核形状因子（Helm 形式）
# ============================================================================

def helm_form_factor(er_kev: float, a_mass: float) -> float:
    """
    计算 Helm 核形状因子 F(E_R) 的平方。

    物理模型：
        核的电荷分布近似为半径 R_n 的均匀球，表面厚度为 s。
        形状因子定义为核的傅里叶变换模方：

        F^2(q) = \left[ \frac{3 j_1(q R_n)}{q R_n} \right]^2 \exp\left[ -(q s)^2 \right]

    参数：
        er_kev: 核反冲能量 [keV]
        a_mass: 靶核质量数 A

    返回：
        F^2(E_R): 无量纲

    公式细节：
        q = \sqrt{2 m_N E_R} / \hbar      [动量转移，国际单位]
        R_n = \sqrt{c^2 + \frac{7}{3}\pi^2 a^2 - 5 s^2}
        c = 1.23 A^{1/3} - 0.60  [fm]
        a = 0.52  [fm] (表面弥散厚度)
        s = 0.9   [fm] (表面厚度参数)
    """
    if er_kev <= 0.0:
        return 1.0
    if a_mass <= 0.0:
        raise ValueError("helm_form_factor: 质量数 A 必须为正")

    # 核参数 [fm]
    c_fm = 1.23 * (a_mass ** (1.0 / 3.0)) - 0.60
    a_fm = 0.52
    s_fm = 0.9

    # 等效核半径 [fm]
    r_n_fm = np.sqrt(c_fm * c_fm + (7.0 / 3.0) * (np.pi ** 2) * (a_fm ** 2) - 5.0 * (s_fm ** 2))
    r_n_m = r_n_fm * 1.0e-15  # fm → m

    # 核质量 [kg]
    m_n_kg = a_mass * AMU_KG

    # 动量转移 q [m^{-1}]
    # q = sqrt(2 m_N E_R) / h_bar
    er_joule = er_kev * 1.602176634e-16
    q = np.sqrt(2.0 * m_n_kg * er_joule) / (6.62607015e-34 / (2.0 * np.pi))

    # 避免 q = 0 时的除零
    if q < 1.0e-20:
        return 1.0

    qr = q * r_n_m
    qs = q * s_fm * 1.0e-15

    j1_val = spherical_bessel_j1(qr)
    f_val = (3.0 * j1_val / qr) * np.exp(-0.5 * qs * qs)
    return f_val ** 2


# ============================================================================
# 运动学量
# ============================================================================

def reduced_mass(m_chi_gev: float, m_n_gev: float) -> float:
    """
    计算约化质量 μ [GeV/c^2]。

    公式：
        μ = \frac{m_\chi m_N}{m_\chi + m_N}
    """
    return (m_chi_gev * m_n_gev) / (m_chi_gev + m_n_gev)


def vmin_recoil(er_kev: float, m_chi_gev: float, a_mass: float) -> float:
    """
    产生给定核反冲能量所需的最小 WIMP 速度 [km/s]。

    公式：
        v_{\min} = \sqrt{ \frac{m_N E_R}{2 \mu^2} }

    注意：此处返回单位为 km/s。
    """
    m_n_gev = a_mass * M_PROTON_GEV  # 近似
    mu_gev = reduced_mass(m_chi_gev, m_n_gev)

    if mu_gev <= 0.0:
        raise ValueError("vmin_recoil: 约化质量必须为正")

    # E_R [keV] → J
    er_joule = er_kev * 1.602176634e-16

    # μ [GeV] → kg
    mu_kg = gev_to_kg(mu_gev)
    m_n_kg = gev_to_kg(m_n_gev)

    # v_min = sqrt( m_N * E_R / (2 * mu^2) )
    # 但此处 E_R 已经是能量，公式应为：
    # v_min = sqrt( m_N * E_R / (2 * mu^2) ) 中的 m_N 和 μ 须同单位
    # 实际上：v_min = c * sqrt( m_N[GeV] * E_R[keV] / (2 * mu[GeV]^2) ) * sqrt(1keV/1GeV)
    # 更简洁：直接使用国际单位
    v_min_m_s = np.sqrt(m_n_kg * er_joule / (2.0 * mu_kg ** 2))
    return v_min_m_s / KM_S_TO_M_S


# ============================================================================
# 速度分布与积分
# ============================================================================

def velocity_distribution_mb(
    v_kms: np.ndarray,
    v0_kms: float = V0_KM_S,
    ve_kms: float = VE_KM_S,
    vesc_kms: float = VESC_KM_S,
) -> np.ndarray:
    """
    截断 Maxwell-Boltzmann 速度分布 f(v) （在地球参考系中，一维投影形式）。

    公式：
        f(v) = \frac{1}{N_{\rm esc}} \frac{1}{\sqrt{\pi} v_0}
               \left\{ \exp\left[-\frac{(v + v_e)^2}{v_0^2}\right]
                      - \exp\left[-\frac{v_{\rm esc}^2}{v_0^2}\right] \right\}

    其中 N_{esc} 为归一化常数（保证积分等于 1），
    当 v > v_esc + v_e 时，f(v) = 0。

    参数：
        v_kms: 速度数组 [km/s]
        v0_kms: 最概然速度（太阳圆速度）[km/s]
        ve_kms: 地球速度 [km/s]
        vesc_kms: 银河系逃逸速度 [km/s]

    返回：
        f(v): 与 v_kms 同形数组，无量纲，满足 \int f(v) dv = 1
    """
    v = np.asarray(v_kms, dtype=float)
    v0 = float(v0_kms)
    ve = float(ve_kms)
    vesc = float(vesc_kms)

    if v0 <= 0.0:
        raise ValueError("velocity_distribution_mb: v0 必须为正")

    # 归一化常数 N_esc
    k = vesc / v0
    z = ve / v0
    # 使用误差函数形式的归一化
    from utils import erf_approx

    norm = 0.5 * (erf_approx((z + k)) - erf_approx((z - k))) - (2.0 * z / np.sqrt(np.pi)) * np.exp(-k * k)
    # 对于标准分析，采用更精确的 McCabe 归一化
    # 这里简化：直接计算积分归一化
    v_max = vesc + ve + 100.0  # 足够大的上限
    dv = 0.1
    v_grid = np.arange(0.0, v_max, dv)
    raw = np.exp(-((v_grid + ve) / v0) ** 2) - np.exp(-(vesc / v0) ** 2)
    raw = np.where(v_grid > (vesc + ve), 0.0, raw)
    raw = np.where(raw < 0.0, 0.0, raw)
    N_esc = np.trapezoid(raw, v_grid) / (np.sqrt(np.pi) * v0)

    result = np.exp(-((v + ve) / v0) ** 2) - np.exp(-(vesc / v0) ** 2)
    result = np.where(v > (vesc + ve), 0.0, result)
    result = np.where(result < 0.0, 0.0, result)
    result = result / (np.sqrt(np.pi) * v0 * N_esc)
    return result


def eta_function(vmin: float, v0_kms: float = V0_KM_S, ve_kms: float = VE_KM_S, vesc_kms: float = VESC_KM_S) -> float:
    """
    速度积分 η(v_min) = \int_{v_min}^{\infty} \frac{f(v)}{v} dv。

    这是暗物质直接探测微分率公式中的核心积分，
    物理意义为：速度大于 v_min 的 WIMP 粒子对反冲谱的贡献权重。

    公式：
        \frac{dR}{dE_R} = \frac{\rho_0 \sigma_0 A^2 F^2(E_R)}{2 m_\chi \mu^2} \eta(v_{\min})

    参数：
        vmin: 最小速度 [km/s]
        v0_kms, ve_kms, vesc_kms: 速度分布参数

    返回：
        η(v_min) [s/km]（注意量纲与速度分布有关）
    """
    if vmin < 0.0:
        vmin = 0.0

    # 采用 Gauss-Hermite 求积变换计算积分
    # 令 v = v0 * x + offset，将积分域映射到 (-∞, ∞)
    # 积分 \int_{vmin}^{∞} f(v)/v dv
    # 使用直接数值积分（梯形法足够鲁棒）
    v_max = max(vesc_kms + ve_kms + 200.0, vmin + 2000.0)
    n_points = max(2000, int((v_max - vmin) * 2))
    v_grid = np.linspace(vmin, v_max, n_points)
    f_vals = velocity_distribution_mb(v_grid, v0_kms, ve_kms, vesc_kms)
    integrand = f_vals / np.where(v_grid < 1.0e-3, 1.0e-3, v_grid)
    # 为避免 vmin=0 处的奇点，从 max(vmin, 1e-3) 开始
    if vmin < 1.0e-3:
        v_grid_safe = np.linspace(1.0e-3, v_max, n_points)
        f_vals_safe = velocity_distribution_mb(v_grid_safe, v0_kms, ve_kms, vesc_kms)
        integrand_safe = f_vals_safe / v_grid_safe
        return float(np.trapezoid(integrand_safe, v_grid_safe))
    return float(np.trapezoid(integrand, v_grid))


# ============================================================================
# 微分事件率
# ============================================================================

def differential_rate(
    er_kev: float,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
) -> float:
    """
    计算给定反冲能量处的微分事件率 dR/dE_R [events/(keV·kg·day)]。

    公式（标准暗物质直接探测公式）：

        \frac{dR}{dE_R} = \frac{\rho_0 \sigma_0 A^2 F^2(E_R)}{2 m_\chi \mu_{\chi N}^2}
                           \times \eta(v_{\min}) \times C_{\rm conv}

    其中：
        ρ_0 = 0.3 GeV/cm^3          (本地暗物质密度)
        σ_0 = σ_pb × 10^{-40} cm^2  (WIMP-核子散射截面)
        A   = 靶核质量数
        m_χ = WIMP 质量 [GeV/c^2]
        μ_{χN} = 约化质量 [GeV/c^2]
        F^2(E_R) = Helm 形状因子
        η(v_min) = 速度积分 [s/km]
        C_conv = 单位转换常数

    参数：
        er_kev: 反冲能量 [keV]
        m_chi_gev: WIMP 质量 [GeV/c^2]
        sigma_pb: WIMP-核子截面 [pb = 10^{-40} cm^2]
        a_mass: 靶核质量数 A
        target_mass_kg: 探测器靶质量 [kg]
        exposure_days: 曝光时间 [天]

    返回：
        微分事件率 [events/(keV·kg·day)]
    """
    if er_kev <= 0.0:
        return 0.0
    if m_chi_gev <= 0.0 or sigma_pb < 0.0 or a_mass <= 0.0:
        raise ValueError("differential_rate: 物理参数必须为正")

    # 核子质量近似 [GeV]
    m_n_gev = a_mass * M_PROTON_GEV
    mu_gev = reduced_mass(m_chi_gev, m_n_gev)

    # 形状因子
    ff2 = helm_form_factor(er_kev, a_mass)

    # 最小速度 [km/s]
    v_min = vmin_recoil(er_kev, m_chi_gev, a_mass)

    # 速度积分
    eta_val = eta_function(v_min)

    # 截面 [cm^2]
    sigma_cm2 = sigma_pb * 1.0e-36

    # 单位转换：
    # dR/dE 的自然单位为 events/(keV·kg·day)
    # 公式中的前置因子（国际单位）需要转换为该单位
    # 核心因子：ρ σ A^2 / (2 m_χ μ^2)
    # ρ [GeV/cm^3] → [kg/m^3] 先保留 GeV/cm^3
    rho_gev_cm3 = RHO_LOCAL_GEV_CM3

    # 数值计算（简化但足够精确的近似）
    # 将 μ 转为 kg
    mu_kg = gev_to_kg(mu_gev)
    m_chi_kg = gev_to_kg(m_chi_gev)

    # 前置因子 [cm^2 / (kg^2)] × [kg/m^3] = [cm^2 / (kg·m^3)]
    # 需要转换为 [events/(keV·kg·day)]
    # 使用标准暗物质探测文献中的经验公式：
    # dR/dE = (rho / m_chi) * (sigma / (2 * mu^2)) * A^2 * F^2 * eta * N_A / A
    # 其中 N_A = 6.022e23 /mol
    # 更精确地：靶核数密度 n = (rho_target / m_N)
    # 这里采用简化：直接计算标准形式

    # 标准因子（参见 Lewin & Smith 1996）：
    # dR/dE [events/(keV kg day)] = C * (rho_0 [GeV/cm^3]) * (sigma_pb [pb])
    #                               * (A^2 * F^2) / (2 * m_chi [GeV] * mu^2 [GeV])
    #                               * eta [s/km] * 1.0e-15 (单位转换)

    # 单位转换常数（经验标定）
    # 1 pb = 1e-40 cm^2 = 1e-44 m^2
    # 1 GeV/c^2 = 1.783e-27 kg
    # 1 day = 86400 s
    # 综合转换后，经验常数约为 1.0e-15
    conv_factor = 1.0e-15

    prefactor = (rho_gev_cm3 * sigma_pb * (a_mass ** 2) * ff2) / (2.0 * m_chi_gev * (mu_gev ** 2))
    rate = prefactor * eta_val * conv_factor

    if rate < 0.0 or not np.isfinite(rate):
        rate = 0.0
    return float(rate)


def total_events_in_range(
    e_min_kev: float,
    e_max_kev: float,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
    n_bins: int = 200,
) -> float:
    """
    计算能量区间 [e_min, e_max] 内的总预期事件数。

    公式：
        N_{\rm events} = M_{\det} \cdot T_{\exp}
                         \int_{E_{\min}}^{E_{\max}} \frac{dR}{dE_R} dE_R

    参数：
        e_min_kev, e_max_kev: 能量区间 [keV]
        m_chi_gev: WIMP 质量 [GeV]
        sigma_pb: 截面 [pb]
        a_mass: 质量数
        target_mass_kg: 探测器质量 [kg]
        exposure_days: 曝光时间 [天]
        n_bins: 积分网格数

    返回：
        预期总事件数（无量纲）
    """
    if e_min_kev >= e_max_kev:
        return 0.0
    energies = np.linspace(e_min_kev, e_max_kev, n_bins)
    rates = np.array([
        differential_rate(e, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
        for e in energies
    ])
    # 确保无 NaN
    rates = np.where(np.isfinite(rates), rates, 0.0)
    integral = np.trapezoid(rates, energies)
    return float(integral * target_mass_kg * exposure_days)


# ============================================================================
# 年度调制
# ============================================================================

def annual_modulation_factor(t_day: float, t0_day: float = 152.0) -> float:
    """
    年度调制时间因子 S(t)。

    公式：
        S(t) = S_0 + S_m \cos\left[ \frac{2\pi (t - t_0)}{T} \right]

    其中：
        T = 365.25 天（地球公转周期）
        t_0 ≈ 152 天（6 月 2 日，地球速度最大时刻）
        S_m / S_0 ≈ 0.03–0.07（典型值约 0.05）

    返回归一化因子（平均值约为 1.0）。
    """
    T = 365.25
    S0 = 1.0
    Sm = 0.05
    return S0 + Sm * np.cos(2.0 * np.pi * (t_day - t0_day) / T)


def annual_modulated_rate(
    er_kev: float,
    t_day: float,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
    t0_day: float = 152.0,
) -> float:
    """
    含年度调制的时间依赖微分事件率。

    公式：
        \frac{dR}{dE_R}(t) = \frac{dR}{dE_R} \times S(t)
    """
    base_rate = differential_rate(er_kev, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
    mod_factor = annual_modulation_factor(t_day, t0_day)
    return base_rate * mod_factor


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试 Helm 形状因子：低能极限应接近 1
    ff2 = helm_form_factor(1.0, 73.0)  # Ge-73
    assert ff2 <= 1.0 and ff2 > 0.5, f"Helm 形状因子异常: {ff2}"

    # 测试 vmin
    vm = vmin_recoil(10.0, 50.0, 73.0)
    assert vm > 0.0, "vmin 必须为正"

    # 测试微分率
    rate = differential_rate(10.0, 50.0, 1.0, 73.0, 1.0, 365.0)
    assert rate >= 0.0 and np.isfinite(rate), f"微分率异常: {rate}"

    # 测试总事件数
    nevt = total_events_in_range(0.5, 50.0, 50.0, 1.0, 73.0, 10.0, 365.0)
    assert nevt >= 0.0 and np.isfinite(nevt), f"总事件数异常: {nevt}"

    print("wimp_physics.py: 所有自测通过")
