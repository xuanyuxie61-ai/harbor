"""
flamelet_core.py
================
湍流燃烧火焰面模型的核心物理方程与参数定义。

核心科学模型：
--------------
1. 稳态层流火焰面方程（Steady Laminar Flamelet Model, SLFM）

   在混合分数空间 Z ∈ [0, 1] 中，温度 T(Z) 与组分质量分数 Y_k(Z) 满足：

       ρ(Z) * χ(Z) / 2 * d²T/dZ² + ω̇_T(T, Y_k) = 0        ... (1)
       ρ(Z) * χ(Z) / 2 * d²Y_k/dZ² + ω̇_k(T, Y_k) = 0      ... (2)

   其中 χ(Z) = 2 D |∇Z|² 为标量耗散率（scalar dissipation rate），
   D 为分子扩散系数，ρ 为密度，ω̇ 为化学反应源项。

2. 标量耗散率分布（反误差函数形式，Peters, 1984）：

       χ(Z) = χ_st * exp{ -2 [erf⁻¹(2Z - 1)]² } / exp{ -2 [erf⁻¹(2Z_st - 1)]² }  ... (3)

   其中 Z_st 为化学计量混合分数，χ_st 为化学计量点标量耗散率。

3. 理想气体状态方程：

       p = ρ R_u T / W                              ... (4)

   其中 W 为混合气体平均分子量，R_u = 8.314 J/(mol·K) 为通用气体常数。

4. 一步总包反应速率（Arrhenius 形式）：

       ω̇_F = -A ρ² Y_F Y_O exp(-E_a / (R_u T))      ... (5)

   其中 A 为指前因子，E_a 为活化能。

5. 混合分数与平均分子量：

       W(Z) = 1 / ( Z/W_F + (1-Z)/W_O )             ... (6)

   其中 W_F, W_O 分别为燃料与氧化剂分子量。

边界条件：
----------
   T(0) = T_oxidizer,   Y_F(0) = 0,   Y_O(0) = Y_{O,0}
   T(1) = T_fuel,       Y_F(1) = Y_{F,0},   Y_O(1) = 0

作者: 博士级科学代码合成系统
"""

import numpy as np
from scipy.special import erfinv

# 物理常数
R_UNIVERSAL = 8.314462618           # J/(mol·K), 通用气体常数
PRESSURE_ATM = 101325.0             # Pa, 标准大气压

# 燃料/氧化剂物性（以甲烷-空气为例）
MOL_WEIGHT_FUEL = 16.04e-3          # kg/mol, CH4
MOL_WEIGHT_OXIDIZER = 28.97e-3      # kg/mol, Air
STOICHIOMETRIC_RATIO = 17.16        # 氧化剂/燃料质量比

# 反应动力学参数
PRE_EXPONENTIAL = 2.0e6             # m³/(kg·s), 调整至合理量级
ACTIVATION_ENERGY = 8.0e4           # J/mol (降低活化能以改善点火特性)
HEAT_RELEASE = 5.0e7                # J/kg, 燃烧热

# 温度边界条件
T_OXIDIZER = 300.0                  # K
T_FUEL = 300.0                      # K
ADIA_TEMP_STOIC = 2226.0            # K, 化学计量绝热火焰温度

# 入口质量分数
Y_FUEL_INLET = 1.0
Y_OXIDIZER_INLET = 0.232            # 空气中氧气质量分数

# 化学计量混合分数
Z_STOICHIOMETRIC = 1.0 / (1.0 + STOICHIOMETRIC_RATIO)


def scalar_dissipation_rate(Z, chi_st, Z_st=None):
    """
    计算标量耗散率 χ(Z)。

    采用高斯型分布（Peters 修正形式），在化学计量点处取最大值：
        χ(Z) = χ_st * exp[ -(Z - Z_st)² / (2 σ_χ²) ]

    其中 σ_χ 为混合层厚度参数，典型值 0.05。

    Parameters
    ----------
    Z : float or ndarray
        混合分数，取值范围 [0, 1]。
    chi_st : float
        化学计量点标量耗散率，单位 s⁻¹，必须为正数。
    Z_st : float, optional
        化学计量混合分数，默认使用全局参数 Z_STOICHIOMETRIC。

    Returns
    -------
    chi : float or ndarray
        标量耗散率分布。
    """
    if Z_st is None:
        Z_st = Z_STOICHIOMETRIC

    if chi_st <= 0.0:
        raise ValueError("标量耗散率 chi_st 必须为正数。")

    Z = np.clip(Z, 0.0, 1.0)
    sigma_chi = 0.15  # 混合层厚度参数

    exponent = -((Z - Z_st) ** 2) / (2.0 * sigma_chi ** 2)
    exponent = np.clip(exponent, -700.0, 0.0)

    chi = chi_st * np.exp(exponent)
    return chi


def mixture_molecular_weight(Z):
    """
    计算混合气体的平均分子量 W(Z)。

    公式（6）:
        W(Z) = 1 / ( Z/W_F + (1-Z)/W_O )

    Parameters
    ----------
    Z : float or ndarray
        混合分数。

    Returns
    -------
    W : float or ndarray
        平均分子量，单位 kg/mol。
    """
    Z = np.clip(Z, 0.0, 1.0)
    denom = Z / MOL_WEIGHT_FUEL + (1.0 - Z) / MOL_WEIGHT_OXIDIZER
    # 防止除以零
    denom = np.where(np.abs(denom) < 1.0e-30, 1.0e-30, denom)
    return 1.0 / denom


def density_mixture(Z, T):
    """
    计算混合气体密度 ρ(Z, T)。

    公式（4）:
        ρ = p * W(Z) / (R_u * T)

    Parameters
    ----------
    Z : float or ndarray
        混合分数。
    T : float or ndarray
        温度，单位 K，必须为正数。

    Returns
    -------
    rho : float or ndarray
        密度，单位 kg/m³。
    """
    T = np.maximum(T, 1.0)
    W = mixture_molecular_weight(Z)
    rho = PRESSURE_ATM * W / (R_UNIVERSAL * T)
    return rho


def reaction_rate_one_step(T, Y_F, Y_O, Z=None):
    """
    计算一步总包反应的燃料消耗速率 ω̇_F。

    采用修正 Arrhenius 形式（含双限因子），确保数值稳定性：
        ω̇_F = A ρ² Y_F Y_O exp(-E_a / (R_u T)) * G(Z) * L(T)

    其中 L(T) 为温度双限因子：
        L(T) = max(T - T_ign, 0) * max(T_ad - T, 0) / C_norm

    该因子保证：
    - T < T_ignition (≈800K) 时无反应（点火门槛）
    - T > T_ad (≈2226K) 时反应自然衰减（自限性）
    - 中间温度区间反应最强

    Parameters
    ----------
    T : float or ndarray
        温度，单位 K。
    Y_F : float or ndarray
        燃料质量分数。
    Y_O : float or ndarray
        氧化剂质量分数。
    Z : float or ndarray, optional
        混合分数，用于空间限制。

    Returns
    -------
    omega : float or ndarray
        燃料消耗速率，单位 kg/(m³·s)。
    """
    T = np.maximum(T, 100.0)
    Y_F = np.clip(Y_F, 0.0, 1.0)
    Y_O = np.clip(Y_O, 0.0, 1.0)

    # TODO: Hole 1 - 实现一步总包 Arrhenius 反应速率
    # 需要计算：
    # 1. 混合气体密度 rho(Z, T) = p * W(Z) / (R_u * T)
    # 2. Arrhenius 指数项 exp(-Ea / (R_u * T))
    # 3. 反应速率 omega_base = A * rho^2 * Y_F * Y_O * exp(-Ea / (R_u * T))
    # 4. 温度双限因子 L(T)：确保 T < T_ignition 时无反应，T > T_ad 时反应自然衰减
    # 5. 混合分数空间限制因子（高斯型，在 Z_st 附近取最大值）
    # 返回燃料消耗速率 omega，单位 kg/(m^3·s)
    raise NotImplementedError("Hole 1: 请实现 reaction_rate_one_step 函数")


def temperature_equation_rhs(Z, T, Y_F, Y_O, chi_st):
    """
    计算温度方程的右端源项与扩散系数。

    方程（1）可重写为：
        d²T/dZ² = - 2 ω̇_T / (ρ χ)

    其中 ω̇_T = -Q * ω̇_F / ρ 为温度源项（单位 K/s）。

    Parameters
    ----------
    Z : float or ndarray
        混合分数。
    T : float or ndarray
        温度。
    Y_F, Y_O : float or ndarray
        组分质量分数。
    chi_st : float
        化学计量标量耗散率。

    Returns
    -------
    kappa : float or ndarray
        等效扩散系数 ρ χ / 2。
    source : float or ndarray
        温度源项 ω̇_T。
    """
    chi = scalar_dissipation_rate(Z, chi_st)
    rho = density_mixture(Z, T)

    omega_f = reaction_rate_one_step(T, Y_F, Y_O, Z)
    # 温度源项: 反应放热 / (ρ c_p)，这里用简化处理
    omega_T = HEAT_RELEASE * omega_f / (rho * 1200.0)

    kappa = rho * chi / 2.0
    # 防止kappa过小导致数值不稳定
    kappa = np.maximum(kappa, 1.0e-12)

    return kappa, omega_T


def flamelet_boundary_conditions():
    """
    返回火焰面方程的边界条件字典。

    Returns
    -------
    bc : dict
        包含左右边界温度与质量分数的字典。
    """
    return {
        'T_left': T_OXIDIZER,
        'T_right': T_FUEL,
        'Y_F_left': 0.0,
        'Y_F_right': Y_FUEL_INLET,
        'Y_O_left': Y_OXIDIZER_INLET,
        'Y_O_right': 0.0,
        'Z_st': Z_STOICHIOMETRIC,
        'T_ad': ADIA_TEMP_STOIC,
    }


def thermal_diffusivity_ref():
    """
    参考热扩散系数 α = λ / (ρ c_p)，单位 m²/s。

    Returns
    -------
    alpha : float
        参考热扩散系数。
    """
    lambda_gas = 0.026   # W/(m·K)
    cp = 1200.0          # J/(kg·K)
    rho_ref = PRESSURE_ATM * MOL_WEIGHT_OXIDIZER / (R_UNIVERSAL * T_OXIDIZER)
    return lambda_gas / (rho_ref * cp)
