"""
concrete_creep_model.py
混凝土蠕变-徐变本构模型模块

融入种子项目:
  - 316_doughnut_exact: 非线性 ODE 精确解的思想（用于粘弹性演化方程）

功能:
  - B3 混凝土蠕变模型（老化粘弹性）
  - MC2010 fib 徐变与收缩模型
  - 复模量计算（对应频率域分析）
  - 老化系数与成熟度理论
  - 收缩应变预测
"""

import numpy as np
from typing import Optional, Tuple


def b3_compliance_function(
    t: float, t_prime: float,
    q1: float, q2: float, q3: float, q4: float,
    lambda_: float = 1.0
) -> float:
    """
    B3 模型混凝土柔量函数 J(t, t')。

    B3 模型（Bazant-Baweja）将柔量分解为:
        J(t, t') = q_1 + q_2 Q(t, t') + q_3 \\ln\\\left(1 + (t-t')^{n}\\right) + q_4 \\ln(t/t')

    其中:
      - q_1 = 1/E_0: 瞬时弹性柔量
      - q_2 Q(t, t'): 老化粘弹性部分
      - q_3: 非老化粘弹性（固体化理论）
      - q_4: 老化流变（塑性流动）

    参数:
        t: 观测时间 [day]
        t_prime: 加载龄期 [day]
        q1, q2, q3, q4: B3 模型参数 [1/MPa]
        lambda_: 环境湿度影响系数

    返回:
        柔量 J(t, t') [1/MPa]
    """
    if t <= t_prime:
        return q1

    dt = t - t_prime
    tau = t_prime

    # 老化粘弹性核 Q(t, t')
    # 简化的近似形式（基于 solidification 理论）
    Q = 1.0 - np.exp(-lambda_ * np.sqrt(dt / (tau + 1.0)))

    # 非老化对数项
    log_term = np.log1p(dt ** 0.3)

    # 老化流变项
    flow_term = np.log(t / tau) if t / tau > 1.0 else 0.0

    J = q1 + q2 * Q + q3 * log_term + q4 * flow_term
    return max(J, q1)


def b3_creep_coefficient(
    t: float, t_prime: float, E28: float,
    q1: float, q2: float, q3: float, q4: float
) -> float:
    """
    B3 模型蠕变系数 \\phi(t, t')。

    定义为:
        \\phi(t, t') = E(t') J(t, t') - 1

    其中 E(t') 为加载龄期时的弹性模量。

    参数:
        t: 观测时间
        t_prime: 加载龄期
        E28: 28天弹性模量 [MPa]
        q1..q4: B3 参数

    返回:
        蠕变系数
    """
    # 简化的弹性模量龄期关系
    E_tprime = E28 * np.sqrt(t_prime / 28.0) if t_prime < 28.0 else E28
    J = b3_compliance_function(t, t_prime, q1, q2, q3, q4)
    phi = E_tprime * J - 1.0
    return max(phi, 0.0)


def mc2010_creep_coefficient(
    t: float, t0: float,
    fcm: float, RH: float, h0: float,
    cement_type: str = "N"
) -> float:
    """
    fib Model Code 2010 蠕变系数模型。

    MC2010 蠕变系数:
        \\phi(t, t_0) = \\phi_0 \\cdot \\beta_c(t - t_0)

    其中基本蠕变系数:
        \\phi_0 = \\phi_{RH} \\cdot \\beta(f_{cm}) \\cdot \\beta(t_0)

    环境湿度影响:
        \\phi_{RH} = 1 + \\frac{1 - RH/100}{0.1 \\cdot \\sqrt[3]{h_0}}

    强度影响:
        \\beta(f_{cm}) = \\frac{5.3}{\\sqrt{f_{cm}/f_{cm0}}}

    加载龄期影响:
        \\beta(t_0) = \\frac{1}{0.1 + t_0^{0.20}}

    发展系数:
        \\beta_c(t - t_0) = \\\left[ \\frac{(t - t_0)}{\\beta_H + (t - t_0)} \\right]^{0.3}

    参数:
        t: 观测时间 [day]
        t0: 加载龄期 [day]
        fcm: 平均圆柱体抗压强度 [MPa]
        RH: 环境相对湿度 [%]
        h0: 名义厚度 [mm]
        cement_type: 水泥类型 ("N", "R", "SL")

    返回:
        蠕变系数
    """
    if t <= t0:
        return 0.0

    fcm0 = 10.0  # 参考强度 [MPa]

    # 湿度影响
    phi_RH = 1.0 + (1.0 - RH / 100.0) / (0.1 * (h0 ** (1.0 / 3.0)))

    # 强度影响
    beta_fcm = 5.3 / np.sqrt(fcm / fcm0)

    # 加载龄期影响
    beta_t0 = 1.0 / (0.1 + t0 ** 0.20)

    # 基本蠕变系数
    phi_0 = phi_RH * beta_fcm * beta_t0

    # 水泥类型修正
    alpha_factor = {"N": 1.0, "R": 1.25, "SL": 0.85}.get(cement_type, 1.0)

    # 湿度系数 beta_H
    beta_H = min(1.5 * h0 + 250.0 * alpha_factor, 1500.0 * alpha_factor)

    # 发展系数
    dt = t - t0
    beta_c = (dt / (beta_H + dt)) ** 0.3

    phi = phi_0 * beta_c
    return max(phi, 0.0)


def mc2010_shrinkage_strain(
    t: float, ts: float,
    fcm: float, RH: float, h0: float,
    cement_type: str = "N"
) -> float:
    """
    fib Model Code 2010 收缩应变模型。

    总收缩:
        \\varepsilon_{cs}(t, t_s) = \\varepsilon_{cas}(f_{cm}) \\cdot \\beta_{RH}(RH) \\cdot \\beta_s(t - t_s)

    其中:
        \\varepsilon_{cas}(f_{cm}) = -\\alpha_{as} \\\left( \\frac{0.1 \\cdot f_{cm}}{6 + 0.1 \\cdot f_{cm}} \\right)^{2.5} \\cdot 10^{-6}

    湿度函数:
        \\beta_{RH} = 1 - \\\left(\\frac{RH}{100}\\right)^3

    时间发展函数:
        \\beta_s(t - t_s) = \\sqrt{\\frac{(t - t_s)}{350 \\cdot (h_0/100)^2 + (t - t_s)}}

    参数:
        t: 观测时间 [day]
        ts: 收缩开始时间 [day]
        fcm: 平均抗压强度 [MPa]
        RH: 相对湿度 [%]
        h0: 名义厚度 [mm]
        cement_type: 水泥类型

    返回:
        收缩应变 [-]
    """
    if t <= ts:
        return 0.0

    alpha_as = {"N": 800e-6, "R": 700e-6, "SL": 900e-6}.get(cement_type, 800e-6)

    # 基本收缩
    eps_cas = alpha_as * ((0.1 * fcm) / (6.0 + 0.1 * fcm)) ** 2.5

    # 湿度影响
    beta_RH = 1.0 - (RH / 100.0) ** 3

    # 时间发展
    dt = t - ts
    beta_s = np.sqrt(dt / (350.0 * (h0 / 100.0) ** 2 + dt))

    eps_cs = eps_cas * beta_RH * beta_s
    return eps_cs


def aging_elastic_modulus(
    t: float, E28: float, s: float = 0.25
) -> float:
    """
    混凝土弹性模量的龄期发展函数。

    常用公式:
        E(t) = E_{28} \\sqrt{ \\frac{t}{a + b t} }

    或简化的:
        E(t) = E_{28} \\\left( \\frac{t}{4 + 0.85 t} \\right)^{0.5}

    参数:
        t: 龄期 [day]
        E28: 28天弹性模量 [MPa]
        s: 发展指数

    返回:
        E(t) [MPa]
    """
    if t <= 0:
        t = 0.01
    ratio = t / (4.0 + 0.85 * t)
    return E28 * (ratio ** s)


def kelvin_chain_compliance(
    t: float, t_prime: float,
    E0: float, E_k: np.ndarray, eta_k: np.ndarray
) -> float:
    """
    Kelvin 链模型的柔量函数。

    Kelvin 链由瞬时弹性元件和 N 个 Kelvin 单元串联组成:
        J(t, t') = 1/E_0 + \\\sum_{k=1}^{N} \\frac{1}{E_k} \\\left(1 - e^{-(t-t')/\\tau_k}\\right)

    其中松弛时间 \\tau_k = \\eta_k / E_k。

    参数:
        t: 观测时间
        t_prime: 加载龄期
        E0: 瞬时弹性模量
        E_k: Kelvin 单元弹性模量数组
        eta_k: Kelvin 单元粘滞系数数组

    返回:
        柔量
    """
    if t <= t_prime:
        return 1.0 / E0

    dt = t - t_prime
    J = 1.0 / E0
    for Ek, etak in zip(E_k, eta_k):
        if Ek > 0 and etak > 0:
            tau_k = etak / Ek
            J += (1.0 / Ek) * (1.0 - np.exp(-dt / tau_k))
    return J


def maxwell_chain_relaxation(
    t: float, t_prime: float,
    E0: float, E_i: np.ndarray, eta_i: np.ndarray
) -> float:
    """
    Maxwell 链模型的松弛模量。

    Maxwell 链由 N 个 Maxwell 单元并联组成:
        R(t, t') = E_0 + \\\sum_{i=1}^{N} E_i e^{-(t-t')/\\tau_i}

    其中松弛时间 \\tau_i = \\eta_i / E_i。

    参数:
        t: 观测时间
        t_prime: 加载龄期
        E0: 长期弹性模量
        E_i: Maxwell 单元弹性模量数组
        eta_i: Maxwell 单元粘滞系数数组

    返回:
        松弛模量
    """
    if t <= t_prime:
        # 瞬时模量为所有模量之和
        return E0 + np.sum(E_i)

    dt = t - t_prime
    R = E0
    for Ei, etai in zip(E_i, eta_i):
        if Ei > 0 and etai > 0:
            tau_i = etai / Ei
            R += Ei * np.exp(-dt / tau_i)
    return R


def complex_modulus_maxwell(
    omega: np.ndarray, E0: float, E_i: np.ndarray, eta_i: np.ndarray
) -> np.ndarray:
    """
    Maxwell 链的复模量 E^*(\\omega)。

    对于单个 Maxwell 单元:
        E_i^*(\\omega) = \\frac{E_i (i \\omega \\tau_i)}{1 + i \\omega \\tau_i}
                  = \\frac{E_i \\omega^2 \\tau_i^2}{1 + \\omega^2 \\tau_i^2}
                    + i \\frac{E_i \\omega \\tau_i}{1 + \\omega^2 \\tau_i^2}

    总复模量:
        E^*(\\omega) = E_0 + \\\sum_i E_i^*(\\omega)

    存储模量:
        E'(\\omega) = \\text{Re}[E^*(\\omega)]

    损耗模量:
        E''(\\omega) = \\text{Im}[E^*(\\omega)]

    损耗因子:
        \\tan \\delta = E'' / E'

    参数:
        omega: 角频率数组 [rad/s]
        E0: 长期弹性模量
        E_i, eta_i: Maxwell 参数

    返回:
        复模量数组
    """
    omega = np.asarray(omega)
    E_star = np.full_like(omega, E0, dtype=complex)

    for Ei, etai in zip(E_i, eta_i):
        if Ei > 0 and etai > 0:
            tau_i = etai / Ei
            iw_tau = 1j * omega * tau_i
            E_star += Ei * iw_tau / (1.0 + iw_tau)

    return E_star


def degree_of_hydration(
    t: float, T: float, alpha_inf: float = 0.85,
    tau_h: float = 24.0, beta_h: float = 0.7
) -> float:
    """
    水泥水化度（成熟度理论）。

    基于 Arrhenius 方程的温度-时间等效性:
        \\alpha(t, T) = \\alpha_\\\infty \\exp\\\left[-\\\left(\\frac{\\tau_h}{t_e}\\right)^{\\beta_h}\\right]

    等效龄期:
        t_e = \\\\int_0^t \\exp\\\left[\\frac{E_a}{R}\\\left(\\frac{1}{T_{ref}} - \\frac{1}{T(s)}\\right)\\right] ds

    这里使用简化的等温近似:
        t_e \\approx t \\cdot \\exp\\\left[\\frac{E_a}{R}\\\left(\\frac{1}{T_{ref}} - \\frac{1}{T}\\right)\\right]

    参数:
        t: 真实时间 [day]
        T: 温度 [K]
        alpha_inf: 最终水化度
        tau_h: 特征水化时间
        beta_h: 水化形状参数

    返回:
        水化度 [-]
    """
    T_ref = 293.15  # 20°C [K]
    Ea_R = 4000.0   # Ea/R [K]

    # 温度修正的等效龄期
    t_eq = t * np.exp(Ea_R * (1.0 / T_ref - 1.0 / T))

    alpha = alpha_inf * np.exp(-(tau_h / t_eq) ** beta_h)
    return min(alpha, alpha_inf)


def mature_compressive_strength(
    t: float, T: float, fcm28: float
) -> float:
    """
    考虑温度的混凝土抗压强度发展。

    基于水化度:
        f_{cm}(t) = f_{cm,28} \\cdot \\\left[\\frac{\\alpha(t)}{\\alpha_{28}}\\right]^n

    参数:
        t: 龄期 [day]
        T: 温度 [K]
        fcm28: 28天抗压强度 [MPa]

    返回:
        fcm(t) [MPa]
    """
    alpha = degree_of_hydration(t, T)
    alpha28 = degree_of_hydration(28.0, T)
    if alpha28 < 1e-10:
        return fcm28
    n = 1.0  # 强度-水化度关系指数
    return fcm28 * (alpha / alpha28) ** n


def equivalent_age_linear(
    t: float, T_history: np.ndarray, dt: float,
    Ea: float = 33500.0, R_gas: float = 8.314
) -> float:
    """
    计算线性温度历史下的等效龄期。

    等效龄期（Arrhenius 型）:
        t_{eq} = \\\sum  \Delta t_i \\cdot \\exp\\\left[\\frac{E_a}{R}\\\left(\\frac{1}{T_{ref}} - \\frac{1}{T_i}\\right)\\right]

    参数:
        t: 总时间
        T_history: 温度历史数组 [K]
        dt: 时间步长
        Ea: 活化能 [J/mol]
        R_gas: 气体常数 [J/(mol·K)]

    返回:
        等效龄期 [day]
    """
    T_ref = 293.15  # 20°C
    t_eq = 0.0
    for T in T_history:
        beta_T = np.exp((Ea / R_gas) * (1.0 / T_ref - 1.0 / T))
        t_eq += dt * beta_T
    return t_eq


def stress_strain_creep_integral(
    time_points: np.ndarray, strain_history: np.ndarray,
    E28: float, phi_func
) -> np.ndarray:
    """
    基于叠加原理的应力计算（Stieltjes 积分）。

    应力-应变关系（Boltzmann 叠加原理）:
        \\sigma(t) = \\\\int_0^t R(t, t') d\\varepsilon(t')

    离散形式:
        \\sigma_k = \\\sum_{i=1}^k R(t_k, t_i) \Delta\\varepsilon_i

    参数:
        time_points: 时间点数组
        strain_history: 应变历史数组
        E28: 28天弹性模量
        phi_func: 蠕变系数函数 phi(t, t')

    返回:
        应力历史数组
    """
    n = len(time_points)
    stress = np.zeros(n)

    for k in range(n):
        sigma_k = 0.0
        for i in range(k + 1):
            if i == 0:
                d_eps = strain_history[i]
            else:
                d_eps = strain_history[i] - strain_history[i - 1]

            t_k = time_points[k]
            t_i = time_points[i]
            phi = phi_func(t_k, t_i)
            E_ti = aging_elastic_modulus(t_i, E28)
            R = E_ti / (1.0 + phi) if phi >= 0 else E_ti
            sigma_k += R * d_eps

        stress[k] = sigma_k

    return stress


def effective_creep_modulus(
    t: float, t0: float, E28: float, phi: float
) -> float:
    """
    有效徐变模量（Age-Adjusted Effective Modulus, AAEM）。

    AAEM 方法:
        E_{eff}(t, t_0) = \\frac{E(t_0)}{1 + \\chi(t, t_0) \\phi(t, t_0)}

    其中老化系数 \\chi \\approx 0.8（对于缓慢加载）。

    参数:
        t: 观测时间
        t0: 加载龄期
        E28: 28天弹性模量
        phi: 蠕变系数

    返回:
        有效模量 [MPa]
    """
    # TODO(Hole_1): 实现 Age-Adjusted Effective Modulus (AAEM) 公式
    # 需要基于老化系数 chi、加载龄期弹性模量 E(t0) 和蠕变系数 phi
    # 计算有效徐变模量 E_eff(t, t0)
    # 科学公式: E_eff = E(t0) / (1 + chi * phi(t, t0))
    pass
