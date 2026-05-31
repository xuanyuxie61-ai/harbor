"""
海洋平台疲劳累积损伤与可靠性分析模块

基于种子项目：
  - 1095_snakes_probability：马尔可夫链转移矩阵与矩阵幂方法

核心物理模型：
  1. 雨流计数法（Rainflow Counting）：
     从平台响应时程中提取应力循环的幅值-均值对，用于疲劳分析。
     采用简化三点比较法：
       给定峰谷序列 ... S_{i-1}, S_i, S_{i+1} ...
       若 |S_i - S_{i-1}| ≥ |S_{i+1} - S_i| 且 sign(S_i - S_{i-1}) ≠ sign(S_{i+1} - S_i)，
       则构成一个完整循环，幅值 ΔS = |S_i - S_{i-1}|。

  2. S-N 曲线疲劳损伤：
       N = a · S^{-m}    （Basquin 方程）
     其中 a, m 为材料常数（典型海洋钢：m = 3, a = 1e12）。
     Miner 线性累积损伤：
       D = Σ_i (n_i / N_i)
     当 D ≥ 1 时判定为疲劳失效。

  3. 海况马尔可夫链模型（源自 1095_snakes_probability）：
     将连续海况（由 Hs, Tp 表征）离散化为有限状态空间。
     转移矩阵 P 描述从一个海况状态转移到另一个状态的概率：
       π^{(t+1)} = π^{(t)} · P
     稳态分布满足 π = π · P。
     长期疲劳损伤期望：
       E[D] = Σ_s π_s · D_s
     其中 D_s 为海况 s 下的年疲劳损伤率。

  4. 可靠度指标：
       β = (μ_R - μ_S) / √(σ_R² + σ_S²)
     其中 R 为抗力，S 为荷载效应。
     失效概率：P_f ≈ Φ(-β)
"""

import numpy as np
import math
from typing import Tuple, List, Dict


# ======================================================================
# 1. 雨流计数法
# ======================================================================

def rainflow_count_cycles(
    signal: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    简化雨流计数：从应力/响应时程中提取循环幅值和均值。
    返回 [(amplitude, mean), ...] 列表。
    采用峰谷提取 + 三点比较法。
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return []

    # 提取峰谷点
    peaks_valleys = [signal[0]]
    for i in range(1, len(signal) - 1):
        if (signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]) or \
           (signal[i] <= signal[i - 1] and signal[i] < signal[i + 1]):
            peaks_valleys.append(signal[i])
    peaks_valleys.append(signal[-1])

    cycles = []
    stack = []
    for s in peaks_valleys:
        stack.append(s)
        while len(stack) >= 3:
            s1 = stack[-3]
            s2 = stack[-2]
            s3 = stack[-1]
            delta1 = abs(s2 - s1)
            delta2 = abs(s3 - s2)
            if delta1 >= delta2:
                # 构成一个循环
                amp = delta1 * 0.5
                mean = (s1 + s2) * 0.5
                cycles.append((amp, mean))
                stack.pop(-2)
                stack.pop(-2)
            else:
                break
    # 剩余点构成半循环
    for i in range(len(stack) - 1):
        amp = abs(stack[i + 1] - stack[i]) * 0.5
        mean = (stack[i] + stack[i + 1]) * 0.5
        cycles.append((amp, mean))
    return cycles


def rainflow_histogram(
    signal: np.ndarray, n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将雨流计数结果按幅值分箱统计，返回 (bin_edges, counts)。
    """
    cycles = rainflow_count_cycles(signal)
    if len(cycles) == 0:
        return np.zeros(n_bins + 1), np.zeros(n_bins)
    amplitudes = np.array([c[0] for c in cycles])
    min_amp = np.min(amplitudes)
    max_amp = np.max(amplitudes)
    if max_amp - min_amp < 1e-12:
        return np.zeros(n_bins + 1), np.zeros(n_bins)
    counts, edges = np.histogram(amplitudes, bins=n_bins, range=(min_amp, max_amp))
    return edges, counts


# ======================================================================
# 2. S-N 曲线疲劳损伤
# ======================================================================

def sn_curve_cycles(
    stress_range: float,
    a: float = 1.0e12,
    m: float = 3.0,
    threshold: float = 1.0,
) -> float:
    """
    根据 S-N 曲线计算给定应力幅值下的失效循环次数。
    N = a · S^{-m}，若 S < threshold 则取 N = ∞。
    """
    if stress_range < threshold:
        return float('inf')
    return a * (stress_range ** (-m))


def miner_damage(
    cycles: List[Tuple[float, float]],
    a: float = 1.0e12,
    m: float = 3.0,
    threshold: float = 1.0,
) -> float:
    """
    计算 Miner 线性累积损伤 D = Σ (n_i / N_i)。
    每个循环计为 1 次，n_i = 1。
    """
    D = 0.0
    for amp, _ in cycles:
        S = 2.0 * amp  # 应力范围 = 2 × 幅值
        N = sn_curve_cycles(S, a, m, threshold)
        if N < float('inf') and N > 0:
            D += 1.0 / N
    return D


def annual_fatigue_damage(
    stress_signal_annual: np.ndarray,
    a: float = 1.0e12,
    m: float = 3.0,
) -> float:
    """
    计算年度疲劳累积损伤。
    """
    cycles = rainflow_count_cycles(stress_signal_annual)
    return miner_damage(cycles, a, m)


# ======================================================================
# 3. 海况马尔可夫链（源自 1095_snakes_probability）
# ======================================================================

def build_seastate_markov_chain(
    n_states: int = 8,
    transition_exponent: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建离散海况状态的马尔可夫链转移矩阵。
    状态按 Hs 递增排列：状态 0（平静）→ 状态 n-1（极端）。
    转移概率设计为：
      - 大概率停留在当前状态或相邻状态。
      - 小概率跳跃到较远状态。
      - 从平静到极端的概率递减。
    返回 (P, steady_state, state_labels)。
    """
    np.random.seed(seed)
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            dist = abs(i - j)
            if dist == 0:
                P[i, j] = 0.6
            elif dist == 1:
                P[i, j] = 0.25 / transition_exponent
            else:
                P[i, j] = 0.15 * np.exp(-dist) / (n_states - 1)
        # 归一化
        row_sum = np.sum(P[i, :])
        if row_sum > 0:
            P[i, :] /= row_sum
        else:
            P[i, i] = 1.0

    # 计算稳态分布（矩阵幂方法）
    steady = np.zeros(n_states)
    steady[0] = 1.0
    for _ in range(200):
        steady_new = steady @ P
        if np.max(np.abs(steady_new - steady)) < 1e-12:
            steady = steady_new
            break
        steady = steady_new
    steady = np.maximum(steady, 0.0)
    steady /= np.sum(steady)

    # 状态标签：典型海况 Hs (m)
    state_labels = np.linspace(0.5, 12.0, n_states)
    return P, steady, state_labels


def compute_longterm_fatigue_damage_markov(
    P: np.ndarray,
    steady_state: np.ndarray,
    state_damage_rates: np.ndarray,
) -> float:
    """
    基于马尔可夫链稳态分布计算长期期望疲劳损伤率。
    E[D] = Σ_s π_s · D_s
    """
    if P.shape[0] != len(steady_state) or len(steady_state) != len(state_damage_rates):
        raise ValueError("马尔可夫链维度不匹配")
    return float(np.dot(steady_state, state_damage_rates))


def simulate_markov_chain_trajectory(
    P: np.ndarray,
    initial_state: int,
    n_steps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    模拟马尔可夫链状态轨迹。
    返回状态索引数组。
    """
    np.random.seed(seed)
    n_states = P.shape[0]
    traj = np.zeros(n_steps, dtype=int)
    state = initial_state
    traj[0] = state
    for t in range(1, n_steps):
        state = np.random.choice(n_states, p=P[state, :])
        traj[t] = state
    return traj


def markov_chain_n_step_distribution(
    P: np.ndarray,
    initial_dist: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    计算 n 步后的状态分布：π^{(n)} = π^{(0)} · P^n。
    使用迭代矩阵乘法避免直接求幂。
    """
    dist = initial_dist.copy()
    for _ in range(n):
        dist = dist @ P
    return dist


# ======================================================================
# 4. 可靠度分析
# ======================================================================

def reliability_index(
    mean_resistance: float,
    std_resistance: float,
    mean_load: float,
    std_load: float,
) -> float:
    """
    计算可靠度指标 β = (μ_R - μ_S) / √(σ_R² + σ_S²)。
    """
    denom = np.sqrt(std_resistance ** 2 + std_load ** 2)
    if denom < 1e-15:
        return float('inf')
    return (mean_resistance - mean_load) / denom


def failure_probability_from_beta(beta: float) -> float:
    """
    由可靠度指标计算近似失效概率：P_f ≈ Φ(-β)。
    使用误差函数近似标准正态 CDF。
    """
    return 0.5 * (1.0 - math.erf(beta / np.sqrt(2.0)))


def fatigue_life_prediction(
    annual_damage: float,
    design_life_years: float = 25.0,
    safety_factor: float = 10.0,
) -> Dict[str, float]:
    """
    基于 Miner 损伤预测疲劳寿命。
    返回包含预测寿命、安全系数校核等的字典。
    """
    if annual_damage <= 1e-15:
        predicted_life = float('inf')
    else:
        predicted_life = 1.0 / annual_damage
    allowable_damage = 1.0 / safety_factor
    years_to_failure = predicted_life
    is_safe = years_to_failure > design_life_years
    return {
        "annual_damage": annual_damage,
        "predicted_life_years": predicted_life,
        "design_life_years": design_life_years,
        "safety_factor": safety_factor,
        "allowable_damage": allowable_damage,
        "years_to_failure": years_to_failure,
        "is_safe": is_safe,
    }


# ======================================================================
# 5. 应力时程生成（从平台响应转换）
# ======================================================================

def stress_from_platform_response(
    surge: np.ndarray,
    heave: np.ndarray,
    pitch: np.ndarray,
    stress_factor_surge: float = 2.5e5,
    stress_factor_heave: float = 1.8e5,
    stress_factor_pitch: float = 3.2e6,
    noise_level: float = 0.02,
) -> np.ndarray:
    """
    将平台动力响应转换为关键节点应力时程。
    假设应力为位移/转角的线性组合加随机噪声：
        σ(t) = k_s · surge(t) + k_h · heave(t) + k_p · pitch(t) + ε(t)
    """
    stress = (
        stress_factor_surge * surge
        + stress_factor_heave * heave
        + stress_factor_pitch * pitch
    )
    noise = noise_level * np.std(stress) * np.random.randn(len(stress))
    return stress + noise
