"""
aerosol_activation.py
气溶胶活化成云凝结核 (CCN) 动力学模块

整合原项目:
  - 702_logistic_ode: Logistic 增长方程

功能:
  模拟过饱和环境下气溶胶粒子作为 CCN 的激活与增长过程。
  将经典 Köhler 理论与 Logistic 增长动力学结合，描述活化分数
  随过饱和度 S 的演化。

核心公式:
  1. Köhler 方程 (临界过饱和度):
       S_crit = sqrt( 4 A^3 / (27 B) )
     其中 A = 2 σ_w M_w / (R T ρ_w)   (曲率效应)
           B = ν M_w ρ_s / (M_s ρ_w)   (溶质效应)

  2. Logistic 活化动力学:
       df_act/dt = r * f_act * (1 - f_act / K)
     其中 f_act 为活化分数，r 为增长率，K 为饱和活化分数 (承载量)。
     解析解:
       f_act(t) = K f0 exp(r(t-t0)) / (K + f0 (exp(r(t-t0)) - 1))

  3. 激活率参数化 (Abdul-Razzak & Ghan, 2000):
       K(S) = 0.5 * (1 - erf( ln(S_crit / S) / (sqrt(2) ln σ_g) ))
"""

import numpy as np
from math import sqrt, pi, exp, log, erf


class ActivationError(Exception):
    pass


def logistic_exact(t, r, K, t0, f0):
    """
    Logistic 方程的解析解。

    方程:
      df/dt = r f (1 - f/K)

    解析解:
      f(t) = K f0 exp(r(t-t0)) / (K + f0 (exp(r(t-t0)) - 1))

    参数:
      t: 时间 (s)
      r: 内禀增长率 (s^-1)
      K: 环境承载量 (最大活化分数)
      t0: 初始时间 (s)
      f0: 初始值

    返回:
      f(t)
    """
    dt = r * (t - t0)
    # 防止溢出
    if dt > 700:
        return float(K)
    exp_dt = exp(dt)
    numerator = K * f0 * exp_dt
    denominator = K + f0 * (exp_dt - 1.0)
    if abs(denominator) < 1e-15:
        return float(K)
    return numerator / denominator


def kohler_critical_supersaturation(
    temperature,        # K
    surface_tension,    # N/m
    molecular_weight_water,  # kg/mol
    density_water,      # kg/m^3
    molecular_weight_solute, # kg/mol
    density_solute,     # kg/m^3
    vanthoff_factor,    # -
    dry_radius,         # m
    mass_solute,        # kg
):
    """
    计算 Köhler 临界过饱和度 S_crit (%)。

    公式:
      A = 2 σ M_w / (R T ρ_w)
      B = ν ε M_w m_s / (M_s)   [体积等效]
      S_crit = exp( sqrt(4 A^3 / (27 B)) ) - 1 ≈ sqrt(4 A^3 / (27 B))

    参数:
      temperature: 温度 (K)
      surface_tension: 水的表面张力 (N/m)
      molecular_weight_water: 水的摩尔质量 (kg/mol)
      density_water: 水密度 (kg/m^3)
      molecular_weight_solute: 溶质摩尔质量 (kg/mol)
      density_solute: 溶质密度 (kg/m^3)
      vanthoff_factor: 范特霍夫因子
      dry_radius: 干粒子半径 (m)
      mass_solute: 溶质质量 (kg)

    返回:
      S_crit: 临界过饱和度 (小数形式, e.g., 0.002 = 0.2%)
    """
    R = 8.314  # J/(mol K)
    A = 2.0 * surface_tension * molecular_weight_water / (R * temperature * density_water)

    # B 参数 (溶质项)
    volume_solute = mass_solute / density_solute
    B = vanthoff_factor * molecular_weight_water * volume_solute / molecular_weight_solute

    if B <= 0:
        raise ActivationError("kohler_critical_supersaturation: B 参数必须为正")

    s_crit = sqrt(4.0 * A ** 3 / (27.0 * B))
    # 截断到合理范围
    s_crit = min(s_crit, 0.5)
    return float(s_crit)


def activated_fraction_logistic(
    time,
    supersaturation,     # 当前过饱和度 (小数)
    s_crit,              # 临界过饱和度 (小数)
    sigma_g,
    r_growth=0.01,       # s^-1
    f0=0.001,
):
    """
    基于 Logistic 方程和 Köhler 理论的 CCN 活化分数。

    步骤:
      1. 由 S 和 S_crit 计算饱和活化分数 K
      2. 使用 Logistic 解析解计算时间演化

    参数:
      time: 时间数组 (s)
      supersaturation: 环境过饱和度
      s_crit: 临界过饱和度
      sigma_g: 粒径几何标准差
      r_growth: 活化增长率
      f0: 初始活化分数

    返回:
      f_act: 活化分数数组
    """
    if s_crit <= 0 or sigma_g <= 1.0:
        raise ActivationError("activated_fraction_logistic: 参数非法")

    # Abdul-Razzak & Ghan 参数化
    ratio = s_crit / (supersaturation + 1e-12)
    log_ratio = log(ratio)
    log_sigma = log(sigma_g)
    if log_sigma < 1e-12:
        log_sigma = 1e-12

    K = 0.5 * (1.0 - erf(log_ratio / (sqrt(2.0) * log_sigma)))
    K = np.clip(K, 0.0, 1.0)

    t = np.asarray(time, dtype=np.float64)
    f_act = np.array([logistic_exact(ti, r_growth, K, 0.0, f0) for ti in t])
    return np.clip(f_act, 0.0, 1.0)


def ccn_spectrum_derivative(supersaturation, N_total, s_crit, sigma_g):
    """
    CCN 谱的导数 dN/dlnS (即活化数浓度对过饱和度的敏感性)。

    公式:
      dN/dlnS = N_total / (sqrt(2π) ln σ_g) * exp( - (ln S - ln S_crit)^2 / (2 ln^2 σ_g) )
    """
    if supersaturation <= 0 or s_crit <= 0:
        return 0.0
    ln_s = log(supersaturation)
    ln_sc = log(s_crit)
    ln_sigma = log(sigma_g)
    coeff = N_total / (sqrt(2.0 * pi) * ln_sigma)
    exponent = -0.5 * ((ln_s - ln_sc) / ln_sigma) ** 2
    return coeff * exp(exponent)


def compute_ccn_number_concentration(
    supersaturation_percent,
    N_total,
    r_median,
    sigma_g,
    temperature=298.0,
    surface_tension=0.072,
    molecular_weight_water=0.018,
    density_water=1000.0,
    molecular_weight_solute=0.132,
    density_solute=1760.0,
    vanthoff_factor=3.0,
):
    """
    综合计算给定过饱和度下的 CCN 数浓度。

    步骤:
      1. 由干粒子参数估算 S_crit
      2. 计算活化分数
      3. 返回 CCN = N_total * f_act
    """
    dry_radius = r_median * 1e-6  # μm -> m
    mass_solute = (4.0 / 3.0) * pi * dry_radius ** 3 * density_solute

    s_crit = kohler_critical_supersaturation(
        temperature,
        surface_tension,
        molecular_weight_water,
        density_water,
        molecular_weight_solute,
        density_solute,
        vanthoff_factor,
        dry_radius,
        mass_solute,
    )

    s = supersaturation_percent / 100.0
    # 使用稳态活化分数 (t -> inf)
    ratio = s_crit / (s + 1e-12)
    log_ratio = log(ratio)
    log_sigma = log(sigma_g)
    if log_sigma < 1e-12:
        log_sigma = 1e-12
    K = 0.5 * (1.0 - erf(log_ratio / (sqrt(2.0) * log_sigma)))
    K = np.clip(K, 0.0, 1.0)

    return N_total * K
