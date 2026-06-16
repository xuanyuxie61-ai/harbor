"""
gibbs_energy_calphad.py
=======================
Fe-C 二元体系各相的摩尔 Gibbs 能计算。

实现 CALPHAD  sublattice 模型:
  - 替代式相 (LIQUID, FCC_A1, BCC_A2): 置换固溶体模型
  - 间隙式相 (CEMENTITE, Fe3C): 化学计量化合物模型

Gibbs 能的一般形式 (替代式相):
    G_m(x,T) = x_Fe * G°_Fe(T) + x_C * G°_C(T)
              + R*T * [x_Fe*ln(x_Fe) + x_C*ln(x_C)]
              + G^E(x,T)
    其中 G^E = x_Fe * x_C * sum_n L^(n)(T) * (x_Fe - x_C)^n

铁磁性贡献采用 Inden-Hillert-Jarl 模型:
    G^mag = R*T * ln(beta_eff + 1) * f(tau)
    tau = T / T_C
    f(tau) = 1 - (1/A) * [tau^(p-1)/s + ...]  (分段函数)
"""

import math
import numpy as np
from calphad_fec_constants import (
    R_GAS, LN2_CORR, TAU_MAGIC, TAU_FCC_MAGIC,
    G_FE_LIQUID, G_FE_BCC, G_FE_FCC,
    TC_FE_BCC, BMAG_FE_BCC, TC_FE_FCC, BMAG_FE_FCC,
    G_C_LIQUID, G_C_FCC, G_C_BCC, G_C_CEMENT,
    G_FE_CEMENT, G_CEMENT_FE3C,
    L_FCC_FE_C, L_BCC_FE_C, L_LIQUID_FE_C,
    evaluate_poly, redlich_kister_sum,
)


# ============================================================
# Inden-Hillert-Jarl 铁磁模型
# ============================================================

def _ind_hj_f_tau(tau, p, A_factor):
    """
    Inden-Hillert-Jarl 铁磁无量纲函数 f(tau)。

    对于 tau = T/T_C < 1 (有序态):
        f(tau) = 1 - (1/A) * [
            (7*tau^(-1))/(15*s) * (1 - tau^2/28 + ...)
            + tau^(3p-1)/s + tau^(9p-1)/s + ...
        ]

    对于 tau > 1 (无序态):
        f(tau) = -(1/A) * [tau^(-3p)/s + tau^(-9p)/s + ...]

    这里使用简化形式:
        tau < 1: f = 1 - [7/(15*s)]*(tau^(-1) - 1)*(tau^2/28 + tau^6/140 + tau^14/...)
                         - tau^(3p)/(s) - ...
        tau > 1: f = -[tau^(-5p) + tau^(-15p)]/(30*s) - ...

    Parameters
    ----------
    tau : float
        约化温度 T/T_C
    p : float
        结构因子 (BCC: 0.4, FCC: 0.28)
    A_factor : float
        归一化因子, 确保 integral_0^1 f(tau) d(tau) 的一致性

    Returns
    -------
    float
        f(tau) 值
    """
    if abs(tau) < 1e-15:
        return 1.0  # T → 0 极限

    s_factor = (79.0 * tau ** 3) / (35.0 * p) - (47.0) / (15.0 * p) \
        - (675.0 * tau ** 13) / (6000.0 * p)

    if tau < 1.0:
        # 有序态展开
        term1 = (1.0 / A_factor) * (
            (7.0 * (1.0 - tau)) / (15.0 * p)
            + tau ** (3.0 / p) * (1.0 / 30.0)
            + tau ** (9.0 / p) * (1.0 / 150.0)
            + tau ** (15.0 / p) * (1.0 / 400.0)
        )
        # 截断保护
        term1 = min(term1, 2.0)
        return 1.0 - term1
    else:
        # 无序态
        term2 = (1.0 / A_factor) * (
            tau ** (-5.0 / p) * (1.0 / 30.0)
            + tau ** (-15.0 / p) * (1.0 / 150.0)
            + tau ** (-25.0 / p) * (1.0 / 400.0)
        )
        return -term2


def _A_factor(p):
    """
    计算 IHJ 模型的归一化因子 A:
        A = (518/1125) + (11692/15975)*(1/p - 1)
            * [(1/p - 1)/6 + ...]
    """
    A = (518.0 / 1125.0) + (11692.0 / 15975.0) * (1.0 / p - 1.0)
    return max(A, 1e-10)


def gibbs_magnetic(T, Tc, beta, p):
    """
    计算铁磁性对摩尔 Gibbs 能的贡献 (J/mol):
        G_mag = R * T * ln(beta + 1) * f(T/Tc)

    Parameters
    ----------
    T : float
        温度 (K)
    Tc : float
        Curie/Néel 温度 (K, 可为负值表示反铁磁)
    beta : float
        有效 Bohr 磁子数
    p : float
        结构因子

    Returns
    -------
    float
        G_mag (J/mol)
    """
    if abs(Tc) < 1.0 or abs(beta) < 1e-10:
        return 0.0
    tau = T / abs(Tc)
    A = _A_factor(p)
    f_val = _ind_hj_f_tau(tau, p, A)
    return R_GAS * T * math.log(abs(beta) + 1.0) * f_val


# ============================================================
# 各相的参考 Gibbs 能
# ============================================================

def G_ref_Fe(phase, T):
    """
    纯 Fe 在指定相中的参考 Gibbs 能 (J/mol)。

    Parameters
    ----------
    phase : str
        'LIQUID', 'FCC', 'BCC', 'CEMENT'
    T : float
        温度 (K)
    """
    T = max(T, 1.0)
    if phase == 'LIQUID':
        g = evaluate_poly(G_FE_LIQUID, T)
    elif phase == 'FCC':
        g = evaluate_poly(G_FE_FCC, T)
        g += gibbs_magnetic(T, TC_FE_FCC, BMAG_FE_FCC, TAU_FCC_MAGIC)
    elif phase == 'BCC':
        g = evaluate_poly(G_FE_BCC, T)
        g += gibbs_magnetic(T, TC_FE_BCC, BMAG_FE_BCC, TAU_MAGIC)
    elif phase == 'CEMENT':
        g = G_FE_CEMENT
    else:
        g = 0.0
    return g


def G_ref_C(phase, T):
    """
    纯 C 在指定相中的参考 Gibbs 能 (J/mol)。

    Parameters
    ----------
    phase : str
        'LIQUID', 'FCC', 'BCC', 'CEMENT'
    T : float
        温度 (K)
    """
    T = max(T, 1.0)
    if phase == 'LIQUID':
        g = evaluate_poly(G_C_LIQUID, T)
    elif phase == 'FCC':
        g = evaluate_poly(G_C_FCC, T)
    elif phase == 'BCC':
        g = evaluate_poly(G_C_BCC, T)
    elif phase == 'CEMENT':
        g = G_C_CEMENT
    else:
        g = 0.0
    return g


# ============================================================
# 替代式相 Gibbs 能
# ============================================================

def gibbs_substitutional(x_C, T, phase):
    """
    计算替代式固溶体相的摩尔 Gibbs 能 (J/mol):

    G_m = (1-x_C)*G°_Fe(T) + x_C*G°_C(T)
        + R*T*[(1-x_C)*ln(1-x_C) + x_C*ln(x_C)]
        + x_C*(1-x_C) * L(x_C, T)

    其中 L(x_C, T) = sum_n L^(n)(T) * (2*x_C - 1)^n

    Parameters
    ----------
    x_C : float or np.ndarray
        碳摩尔分数 (0 < x_C < 1)
    T : float
        温度 (K)
    phase : str
        'LIQUID', 'FCC', 'BCC'

    Returns
    -------
    float or np.ndarray
        G_m (J/mol)
    """
    scalar_input = np.isscalar(x_C)
    x_C = np.atleast_1d(np.asarray(x_C, dtype=float))

    # 边界保护: 避免 ln(0)
    x_C = np.clip(x_C, 1.0e-15, 1.0 - 1.0e-15)
    x_Fe = 1.0 - x_C

    # 机械混合项
    g_ref = x_Fe * G_ref_Fe(phase, T) + x_C * G_ref_C(phase, T)

    # 理想混合熵项
    # G_ideal = R*T * (x_Fe*ln(x_Fe) + x_C*ln(x_C))
    g_ideal = R_GAS * T * (x_Fe * np.log(x_Fe) + x_C * np.log(x_C))

    # 过剩项 (Redlich-Kister)
    if phase == 'FCC':
        L_params = L_FCC_FE_C
    elif phase == 'BCC':
        L_params = L_BCC_FE_C
    elif phase == 'LIQUID':
        L_params = L_LIQUID_FE_C
    else:
        L_params = []

    g_excess = np.zeros_like(x_C)
    if len(L_params) > 0:
        # 逐点计算 RK 多项式
        for i in range(len(x_C)):
            g_excess[i] = x_Fe[i] * x_C[i] * redlich_kister_sum(L_params, x_C[i])

    g_total = g_ref + g_ideal + g_excess

    if scalar_input:
        return float(g_total[0])
    return g_total


def gibbs_cementite(T):
    """
    渗碳体 Fe3C 的摩尔 Gibbs 能 (J/mol-formula-unit):

    G_Fe3C = 3*G°_Fe^ref + G°_C^ref + a + b*T

    其中 a, b 为经验参数, 相对于石墨+纯铁。

    Parameters
    ----------
    T : float
        温度 (K)

    Returns
    -------
    float
        G_Fe3C (J/mol)
    """
    T = max(T, 1.0)
    a, b = G_CEMENT_FE3C[0], G_CEMENT_FE3C[1]
    # 相对于 3*G_Fe(ref) + G_C(graphite)
    g_form = a + b * T  # 形成 Gibbs 能
    return g_form


# ============================================================
# 化学势 (偏摩尔 Gibbs 能)
# ============================================================

def chemical_potential_C(x_C, T, phase):
    """
    计算碳在指定相中的化学势 mu_C = dG/dx_C |_{T,P}:

    mu_C = G°_C - G°_Fe
         + R*T * [ln(x_C) - ln(1-x_C)]
         + d/dx_C [x_C*(1-x_C)*L(x_C)]

    其中 d/dx_C [x_C*(1-x_C)*L] = (1-2*x_C)*L + x_C*(1-x_C)*dL/dx_C

    Parameters
    ----------
    x_C : float or np.ndarray
        碳摩尔分数
    T : float
        温度 (K)
    phase : str
        'LIQUID', 'FCC', 'BCC'

    Returns
    -------
    float or np.ndarray
        mu_C (J/mol)
    """
    from calphad_fec_constants import rk_derivative
    scalar_input = np.isscalar(x_C)
    x_C = np.atleast_1d(np.asarray(x_C, dtype=float))
    x_C = np.clip(x_C, 1.0e-15, 1.0 - 1.0e-15)
    x_Fe = 1.0 - x_C

    # 参考化学势差
    dmu_ref = G_ref_C(phase, T) - G_ref_Fe(phase, T)

    # 理想混合贡献
    dmu_ideal = R_GAS * T * (np.log(x_C) - np.log(x_Fe))

    # 过剩贡献
    if phase == 'FCC':
        L_params = L_FCC_FE_C
    elif phase == 'BCC':
        L_params = L_BCC_FE_C
    elif phase == 'LIQUID':
        L_params = L_LIQUID_FE_C
    else:
        L_params = []

    dmu_excess = np.zeros_like(x_C)
    if len(L_params) > 0:
        for i in range(len(x_C)):
            xi = 2.0 * x_C[i] - 1.0
            L_val = redlich_kister_sum(L_params, x_C[i])
            dL_dx = rk_derivative(L_params, x_C[i], order=1)
            # d/dx [x*(1-x)*L(x)] = (1-2x)*L + x*(1-x)*dL/dx
            dmu_excess[i] = (1.0 - 2.0 * x_C[i]) * L_val \
                + x_Fe[i] * x_C[i] * dL_dx

    mu_C = dmu_ref + dmu_ideal + dmu_excess

    if scalar_input:
        return float(mu_C[0])
    return mu_C


def chemical_potential_Fe(x_C, T, phase):
    """
    计算铁的化学势 mu_Fe = G_m - x_C * mu_C (Gibbs-Duhem 关系)。

    mu_Fe = G_m - x_C * (dG_m/dx_C)

    Parameters
    ----------
    x_C : float or np.ndarray
        碳摩尔分数
    T : float
        温度 (K)
    phase : str
        'LIQUID', 'FCC', 'BCC'

    Returns
    -------
    float or np.ndarray
        mu_Fe (J/mol)
    """
    scalar_input = np.isscalar(x_C)
    x_C = np.atleast_1d(np.asarray(x_C, dtype=float))
    G_m = gibbs_substitutional(x_C, T, phase)
    mu_C = chemical_potential_C(x_C, T, phase)
    x_C_safe = np.clip(x_C, 1.0e-15, 1.0 - 1.0e-15)
    mu_Fe = G_m - x_C_safe * mu_C

    if scalar_input:
        return float(mu_Fe[0])
    return mu_Fe


def second_derivative_G(x_C, T, phase):
    """
    计算 d²G_m/dx_C² — 用于 Spinodal 条件判断和 Cahn-Hilliard 方程。

    d²G/dx² = R*T * [1/x_C + 1/(1-x_C)]
            + d²/dx² [x*(1-x)*L(x)]

    Parameters
    ----------
    x_C : float or np.ndarray
        碳摩尔分数
    T : float
        温度 (K)
    phase : str
        相名称

    Returns
    -------
    float or np.ndarray
        d²G/dx² (J/mol)
    """
    from calphad_fec_constants import rk_derivative
    scalar_input = np.isscalar(x_C)
    x_C = np.atleast_1d(np.asarray(x_C, dtype=float))
    x_C = np.clip(x_C, 1.0e-15, 1.0 - 1.0e-15)
    x_Fe = 1.0 - x_C

    # 理想混合: d²/dx² [R*T*(x*ln(x) + (1-x)*ln(1-x))]
    # = R*T * [1/x + 1/(1-x)]
    d2g_ideal = R_GAS * T * (1.0 / x_C + 1.0 / x_Fe)

    # 过剩项二阶导
    if phase == 'FCC':
        L_params = L_FCC_FE_C
    elif phase == 'BCC':
        L_params = L_BCC_FE_C
    elif phase == 'LIQUID':
        L_params = L_LIQUID_FE_C
    else:
        L_params = []

    d2g_excess = np.zeros_like(x_C)
    if len(L_params) > 0:
        for i in range(len(x_C)):
            L_val = redlich_kister_sum(L_params, x_C[i])
            dL_dx = rk_derivative(L_params, x_C[i], order=1)
            d2L_dx2 = rk_derivative(L_params, x_C[i], order=2)
            x = x_C[i]
            # d²/dx² [x*(1-x)*L] = -4*L + 2*(1-2x)*dL/dx + x*(1-x)*d²L/dx²
            d2g_excess[i] = -4.0 * L_val \
                + 2.0 * (1.0 - 2.0 * x) * dL_dx \
                + x * (1.0 - x) * d2L_dx2

    d2g = d2g_ideal + d2g_excess

    if scalar_input:
        return float(d2g[0])
    return d2g
