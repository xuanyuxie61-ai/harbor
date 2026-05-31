"""
================================================================================
空化概率模型模块 (cavitation_probability.py)
================================================================================
融合项目:
  - 1183_supreme_vacancy (vacancy_this_span): 概率累积模型

在可压缩CFD中，当局部压力低于饱和蒸汽压时发生空化（cavitation）。
本模块提供基于概率论的空化发生风险评估：
  1. 局部压力低于蒸汽压的概率计算
  2. 多位置联合空化概率（基于独立事件假设）
  3. 空化初生准则（nucleation theory）

物理基础:
    空化数 (Cavitation number):
        σ = (p_∞ - p_v) / (0.5 ρ U_∞²)

    当 σ < σ_crit 时发生空化。考虑压力脉动，局部空化概率为：

        P_cav(x) = P( p(x) < p_v ) = Φ( (p_v - ⟨p⟩) / σ_p )

    其中 Φ 为标准正态CDF，σ_p 为压力脉动标准差。
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def cavitation_probability_local(mean_p: float, p_vapor: float, std_p: float) -> float:
    """
    计算单点空化发生概率

    假设压力脉动服从正态分布 N(μ_p, σ_p²)，则：

        P_cav = P(p < p_v) = 0.5 · [1 + erf( (p_v - μ_p) / (√2 σ_p) )]

    参数:
        mean_p: 平均压力
        p_vapor: 饱和蒸汽压
        std_p: 压力脉动标准差

    返回:
        空化概率 [0, 1]
    """
    if std_p < 1e-14:
        return 1.0 if mean_p < p_vapor else 0.0

    z = (p_vapor - mean_p) / (np.sqrt(2.0) * std_p)
    # 防止溢出
    z = np.clip(z, -5.0, 5.0)

    # 误差函数近似
    prob = 0.5 * (1.0 + erf_approx(z))
    return float(np.clip(prob, 0.0, 1.0))


def erf_approx(x: float) -> float:
    """
    误差函数近似 (Abramowitz & Stegun, 7.1.26)

        erf(x) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵) exp(-x²)

    其中 t = 1 / (1 + px), p = 0.3275911
    """
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign_x = np.sign(x)
    x_abs = abs(x)

    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs ** 2)

    return sign_x * y


def joint_cavitation_probability(probabilities: np.ndarray, independence: bool = True) -> float:
    """
    多点联合空化概率

    独立事件假设:
        P(∪_i A_i) = 1 - ∏_i (1 - P_i)

    完全相关假设:
        P(∪_i A_i) = max_i P_i

    实际情况下使用Frank copula建模：
        P_union = 1 - C_θ(1-P_1, ..., 1-P_n)

    参数:
        probabilities: 各点空化概率数组
        independence: 是否假设独立

    返回:
        联合空化概率
    """
    probs = np.clip(np.asarray(probabilities), 0.0, 1.0)

    if len(probs) == 0:
        return 0.0

    if independence:
        p_no_cav = np.prod(1.0 - probs)
        p_union = 1.0 - p_no_cav
    else:
        # 完全正相关（保守估计）
        p_union = float(np.max(probs))

    return float(np.clip(p_union, 0.0, 1.0))


def cavitation_inception_criterion(Re: float, sigma: float, roughness: float = 1e-5) -> dict:
    """
    空化初生准则

    基于nucleation theory，空化初生临界条件：

        σ_i ≥ σ_crit = C · (Re · roughness/L)^{-1/5}

    其中 C 为经验常数，roughness 为表面粗糙度，L 为特征长度。

    参数:
        Re: 雷诺数
        sigma: 空化数
        roughness: 无量纲表面粗糙度

    返回:
        dict 包含判断结果、临界空化数、安全裕度
    """
    # 经验公式（基于Arndt & Ippen 1968）
    C_crit = 0.5
    sigma_critical = C_crit * (Re * roughness) ** (-0.2)

    margin = sigma - sigma_critical
    inception_risk = 1.0 / (1.0 + np.exp(5.0 * margin))  # logistic映射

    return {
        'cavitation_inception': sigma < sigma_critical,
        'sigma_critical': float(sigma_critical),
        'sigma_actual': float(sigma),
        'safety_margin': float(margin),
        'inception_risk': float(inception_risk)
    }


def analyze_pressure_field_for_cavitation(p_field: np.ndarray, p_vapor: float,
                                          u_field: np.ndarray = None, rho: float = 1.0) -> dict:
    """
    对全场压力进行空化风险评估

    分析内容:
      1. 全局最小压力与空化数
      2. 空化风险区域识别
      3. 空化概率空间分布
    """
    mean_p = np.mean(p_field)
    min_p = np.min(p_field)
    std_p = np.std(p_field)

    # 计算空化数
    if u_field is not None:
        u_max = np.max(np.abs(u_field))
        sigma_global = (mean_p - p_vapor) / (0.5 * rho * u_max ** 2 + 1e-14)
    else:
        sigma_global = (mean_p - p_vapor) / (0.5 * rho + 1e-14)

    # 逐点空化概率
    prob_field = np.zeros_like(p_field)
    for j in range(p_field.shape[0]):
        for i in range(p_field.shape[1]):
            prob_field[j, i] = cavitation_probability_local(p_field[j, i], p_vapor, std_p)

    # 高风险区域（概率>0.1）
    high_risk_mask = prob_field > 0.1
    high_risk_fraction = np.sum(high_risk_mask) / p_field.size

    # 联合概率（所有高风险点）
    high_risk_probs = prob_field[high_risk_mask]
    if len(high_risk_probs) > 0:
        joint_prob = joint_cavitation_probability(high_risk_probs[:100])  # 限制计算量
    else:
        joint_prob = 0.0

    return {
        'mean_pressure': float(mean_p),
        'min_pressure': float(min_p),
        'pressure_std': float(std_p),
        'cavitation_number': float(sigma_global),
        'probability_field': prob_field,
        'high_risk_fraction': float(high_risk_fraction),
        'joint_cavitation_probability': float(joint_prob),
        'max_local_probability': float(np.max(prob_field))
    }


def compute_nucleation_rate(p: float, p_vapor: float, T: float, surface_tension: float = 0.072) -> float:
    """
    基于经典成核理论计算气泡成核率

    均相成核率 (Classical Nucleation Theory, CNT):

        J = J₀ exp( -ΔG* / k_B T )

    临界成核功:
        ΔG* = (16π γ³) / (3 (p_v - p)²)

    参数:
        p: 局部压力
        p_vapor: 饱和蒸汽压
        T: 温度
        surface_tension: 表面张力系数

    返回:
        成核率 (events/m³/s)
    """
    k_b = 1.380649e-23  # J/K
    delta_p = max(p_vapor - p, 1e-10)

    # 临界半径
    r_star = 2.0 * surface_tension / delta_p

    # 临界成核功（无量纲化简化）
    delta_g_star = (16.0 * np.pi * surface_tension ** 3) / (3.0 * delta_p ** 2)

    # 指前因子（简化）
    j0 = 1e33  # m⁻³ s⁻¹

    # 成核率
    J = j0 * np.exp(-delta_g_star / max(T, 1e-10))

    return float(J)
