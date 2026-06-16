"""
time_evolution_cp.py
--------------------
B0-B0bar 混合的时间演化与 CP 不对称分析。
映射自种子项目 1411_weekday (日历日期 → Julian Ephemeris Date 转换)。

物理动机:
  在 B 工厂 (BaBar, Belle, Belle II) 中, B 介子对的衰变时间差 Delta t
  是通过测量两个 B 衰变顶点的空间距离 Delta z 和 Boost βγ 得到的:
      Delta t = Delta z / (βγ c)

  时间戳记录通常以 Julian Ephemeris Date (JED) 或 UNIX 时间戳给出,
  需要转换到 B 介子的 proper time (ps) 进行物理分析。

本模块实现:
  1) YMDF → JED 转换 (仿 1411_weekday, 使用 Gregorian 历法);
  2) B0-B0bar 混合的时间演化函数:
       f(t) = e^{-Gamma t} [cosh(Delta Gamma t / 4) ± cos(Delta m t)]  (未标记)
       f(t) = e^{-Gamma t} [sinh(Delta Gamma t / 4) ± sin(Delta m t)]  (未标记)
  3) CP 不对称的时间依赖:
       A_CP(t) = S sin(Delta m t) - C cos(Delta m t)
     其中 S = 2 Im(lambda_f) / (1 + |lambda_f|^2),
           C = (1 - |lambda_f|^2) / (1 + |lambda_f|^2)
  4) lambda_f = (q/p) (A_bar / A), 其中 (q/p) = sqrt((M_12^* - i/2 Gamma_12^*) /
                                                     (M_12 - i/2 Gamma_12))

公式来源:
  - Bigi, Sanda, "CP Violation", Cambridge Univ. Press, 2nd ed.
  - Nir, "CP Violation in and beyond the Standard Model", arXiv:hep-ph/0510237
  - PDG 2024, section "Mixing and CP violation parameters"
"""

from __future__ import annotations
import math
import cmath
from typing import List, Tuple

from b_physics_constants import (
    DELTA_M_D, DELTA_M_S, DELTA_GAMMA_D, DELTA_GAMMA_S,
    TAU_B0, TAU_BS, LAMBDA_W, A_W, RHOBAR_W, ETABAR_W,
    wolfenstein_to_ckm, unitarity_triangle_angles,
)


# ========================================================================== #
#              时间戳转换 (仿 1411_weekday: ymdf → JED)                     #
# ========================================================================== #
def y_common_to_astronomical(y: int) -> int:
    """
    将历史年份 (含负数, 无 year 0) 转换为天文年份 (含 year 0 = 1 BC)。
      y_common_to_astronomical(y) = y + 1 if y < 0 else y
    """
    return y + 1 if y < 0 else y


def ymdf_to_jed_gregorian(y: int, m: int, d: int, f: float = 0.0) -> float:
    """
    将 Gregorian 历法的 YMDF 日期转换为 Julian Ephemeris Date。
    算法 (仿 1411):
      y2 = y_common_to_astronomical(y)
      y' = y2 + 4716 - floor((14 - m) / 12)
      m' = (m + 9) mod 12
      d' = d - 1
      J1 = floor(1461 * y' / 4)
      J2 = floor((153 * m' + 2) / 5)
      g  = floor(3 * floor((y' + 184) / 100) / 4) - 38
      JED = J1 + J2 + d' - 1401 - g - 0.5 + f
    """
    y2 = y_common_to_astronomical(y)
    y_prime = y2 + 4716 - (14 - m) // 12
    m_prime = (m + 9) % 12
    d_prime = d - 1
    j1 = (1461 * y_prime) // 4
    j2 = (153 * m_prime + 2) // 5
    g = (3 * ((y_prime + 184) // 100)) // 4 - 38
    jed = j1 + j2 + d_prime - 1401 - g - 0.5 + f
    return float(jed)


def jed_to_b_proper_time(
        jed_production: float,
        jed_decay: float,
        boost_beta_gamma: float = 0.56,
) -> float:
    """
    将 JED 时间差转换为 B 介子的 proper time (ps):
        Delta t_sec = (jed_decay - jed_production) * 86400   # 1 day = 86400 s
        Delta z = beta_gamma * c * Delta t_sec
        proper_time = Delta z / (beta_gamma * c) = Delta t_sec * 1e12   # 秒 → ps

    实际上, 在 B 工厂中 proper time 由 Delta z / (beta_gamma c) 给出,
    此处简化为直接返回时间差 (ps), 假设 boost 因子已校正。
    """
    delta_days = jed_decay - jed_production
    delta_sec = delta_days * 86400.0
    delta_ps = delta_sec * 1.0e12
    return delta_ps


# ========================================================================== #
#                   B0-B0bar 混合: 时间演化函数                             #
# ========================================================================== #
def b_mixing_qp_ratio(
        delta_m: float,
        delta_gamma: float,
        a_sl: float = 0.0,
) -> complex:
    """
    计算 |q/p| 和 phase:
        (q/p)^2 = (M_12^* - i/2 Gamma_12^*) / (M_12 - i/2 Gamma_12)
    在标准参数化下:
        |q/p| = 1 + a_sl / 2   (a_sl 为半轻 CP 不对称)
        phi_qp = arg(q/p) ≈ 0 (在 SM 中, B_d 系统 phi_qp ≈ -beta ≈ -0.38 rad)
    简化: 返回 q/p = (1 + a_sl/2) * exp(i * 0) ≈ 1 + a_sl/2
    """
    if abs(a_sl) > 0.5:
        raise ValueError("|a_sl| 应远小于 1, 输入值物理上不合理")
    mod = 1.0 + 0.5 * a_sl
    # 微小相位 (SM 中近似为零)
    return complex(mod, 0.0)


def time_evolution_untagged(
        t: float,
        delta_m: float,
        delta_gamma: float,
        gamma_total: float,
) -> Tuple[float, float]:
    """
    未标记 (untagged) B → f 衰变的时间分布:
        P(B0 → f; t) ∝ e^{-Gamma t} [cosh(Delta Gamma t / 2) + cos(Delta m t)]
        P(B0bar → f; t) ∝ e^{-Gamma t} [cosh(Delta Gamma t / 2) - cos(Delta m t)]
    返回 (P_unmixed, P_mixed), 归一化到 t=0 时为 1。
    """
    if t < 0.0:
        return 0.0, 0.0
    exp_gt = math.exp(-gamma_total * t)
    cosh_term = math.cosh(0.5 * delta_gamma * t)
    cos_term = math.cos(delta_m * t)
    P_unmixed = exp_gt * (cosh_term + cos_term) / 2.0
    P_mixed   = exp_gt * (cosh_term - cos_term) / 2.0
    return P_unmixed, P_mixed


def time_evolution_tagged(
        t: float,
        delta_m: float,
        delta_gamma: float,
        gamma_total: float,
        lambda_f: complex,
        charge_initial: int = +1,
) -> float:
    """
    标记初态为 B0 或 B0bar 的时间分布 (到 CP 本征态 f):
    对初态 B0:
        Gamma(B0(t) → f) ∝ e^{-Gamma t} {
            (1 + |lambda_f|^2)/2 * cosh(Delta Gamma t / 2)
            + (1 - |lambda_f|^2)/2 * cos(Delta m t)
            - Re(lambda_f) * sinh(Delta Gamma t / 2)
            - Im(lambda_f) * sin(Delta m t)
        }
    对初态 B0bar (用 1/lambda_f 替代 lambda_f):
        等价于 lambda_f → 1/lambda_f

    charge_initial = +1 表示初态为 B0, -1 表示 B0bar。
    """
    if t < 0.0:
        return 0.0
    if charge_initial < 0:
        if abs(lambda_f) < 1.0e-30:
            return 0.0
        lam = 1.0 / lambda_f
    else:
        lam = lambda_f
    exp_gt = math.exp(-gamma_total * t)
    cosh_h = math.cosh(0.5 * delta_gamma * t)
    sinh_h = math.sinh(0.5 * delta_gamma * t)
    cos_t = math.cos(delta_m * t)
    sin_t = math.sin(delta_m * t)
    term1 = 0.5 * (1.0 + abs(lam) ** 2) * cosh_h
    term2 = 0.5 * (1.0 - abs(lam) ** 2) * cos_t
    term3 = -lam.real * sinh_h
    term4 = -lam.imag * sin_t
    return exp_gt * (term1 + term2 + term3 + term4)


# ========================================================================== #
#                   时间依赖 CP 不对称                                      #
# ========================================================================== #
def cp_asymmetry_time_dependent(
        t: float,
        S_f: float,
        C_f: float,
        delta_m: float,
) -> float:
    """
    时间依赖的 CP 不对称:
        A_CP(t) = [Gamma(B0bar(t) → f) - Gamma(B0(t) → f)] /
                  [Gamma(B0bar(t) → f) + Gamma(B0(t) → f)]
                = S_f sin(Delta m t) - C_f cos(Delta m t)
    其中:
        S_f = 2 Im(lambda_f) / (1 + |lambda_f|^2)
        C_f = (1 - |lambda_f|^2) / (1 + |lambda_f|^2)

    对 B0 → J/psi Ks (黄金道):
        lambda_f ≈ -e^{-2 i beta},  C_f ≈ 0,  S_f = sin(2 beta)
    """
    if t < 0.0:
        return 0.0
    return S_f * math.sin(delta_m * t) - C_f * math.cos(delta_m * t)


def lambda_f_from_weak_phases(
        phi_mix: float,
        phi_decay: float,
        ratio_direct: complex = 1.0 + 0.0j,
) -> complex:
    """
    计算 lambda_f:
        lambda_f = (q/p) * (A_bar_f / A_f)
    在只有单一衰变振幅时:
        A_bar_f / A_f = exp(-2 i phi_decay) * ratio_direct
        (q/p) ≈ exp(-i phi_mix)
    因此:
        lambda_f = exp(-i (phi_mix + 2 phi_decay)) * ratio_direct
    当 |ratio_direct| = 1 且 phi_mix + 2 phi_decay != 0, 有 |lambda_f| = 1
    (无直接 CPV, 仅混合诱导 CPV)。
    """
    return cmath.exp(-1j * (phi_mix + 2.0 * phi_decay)) * ratio_direct


def extract_sin2beta_from_time_data(
        times: List[float],
        asymmetries: List[float],
        delta_m: float = DELTA_M_D,
) -> Tuple[float, float]:
    """
    从时间依赖的 CP 不对称数据提取 sin(2 beta) 和 C_f。
    通过最小二乘拟合:
        A_CP(t_i) = S sin(Delta m t_i) - C cos(Delta m t_i)
    构建线性系统:
        [Σ sin^2,  -Σ sin cos] [S]   [Σ A sin]
        [-Σ sin cos, Σ cos^2 ] [C] = [-Σ A cos]
    返回 (S_f, C_f), 其中 S_f = sin(2 beta) 对黄金道。
    """
    if len(times) != len(asymmetries):
        raise ValueError("时间数据与不对称数据长度不一致")
    n = len(times)
    if n < 2:
        raise ValueError("至少需要 2 个数据点")
    sum_sin2 = 0.0
    sum_cos2 = 0.0
    sum_sincos = 0.0
    sum_A_sin = 0.0
    sum_A_cos = 0.0
    for k in range(n):
        t = times[k]
        A = asymmetries[k]
        s = math.sin(delta_m * t)
        c = math.cos(delta_m * t)
        sum_sin2 += s * s
        sum_cos2 += c * c
        sum_sincos += s * c
        sum_A_sin += A * s
        sum_A_cos += A * c
    # 解 2x2 线性系统
    det = sum_sin2 * sum_cos2 - sum_sincos * sum_sincos
    if abs(det) < 1.0e-30:
        return 0.0, 0.0
    S_f = (sum_cos2 * sum_A_sin - sum_sincos * sum_A_cos) / det
    C_f = (-sum_sincos * sum_A_sin + sum_sin2 * sum_A_cos) / det * (-1)
    return S_f, C_f


# ========================================================================== #
#                   B 工厂时间戳生成器 (用于模拟)                          #
# ========================================================================== #
def simulate_decay_times(
        n_events: int,
        tau_B: float,
        delta_m: float,
        delta_gamma: float,
        S_f: float,
        C_f: float,
        seed: int = 42,
) -> Tuple[List[float], List[int]]:
    """
    生成模拟的 B 衰变 proper time 和初态标记:
      1) 从 P(t) ∝ e^{-t/tau} 生成 proper time (指数分布);
      2) 对每个事件, 以 50% 概率标记为初态 B0 (+1) 或 B0bar (-1);
      3) 根据 time_evolution_tagged 计算衰变概率, 以 Metropolis 接受。
    返回 (times, initial_states)。
    """
    import random
    random.seed(seed)
    times = []
    states = []
    gamma_total = 1.0 / tau_B
    max_rate = 1.0   # P(t=0) = 1 为最大
    attempts = 0
    max_attempts = n_events * 50
    while len(times) < n_events and attempts < max_attempts:
        attempts += 1
        t = random.expovariate(gamma_total)
        if t > 10.0 * tau_B:
            continue
        q = random.random()
        charge = +1 if random.random() < 0.5 else -1
        P = time_evolution_tagged(
            t, delta_m, delta_gamma, gamma_total,
            lambda_f_from_weak_phases(
                phi_mix=0.0,
                phi_decay=math.atan2(S_f, math.sqrt(max(1.0 - S_f * S_f, 0.0))) / 2.0,
                ratio_direct=(1.0 + C_f) / (1.0 - C_f + 1.0e-30) + 0.0j if abs(1 - C_f) > 1.0e-10 else 1.0 + 0.0j,
            ),
            charge_initial=charge,
        )
        # 归一化: 最大 P ≈ 1 (对 t ≈ 0)
        accept_prob = min(max(P, 0.0), max_rate)
        if q < accept_prob:
            times.append(t)
            states.append(charge)
    return times, states
