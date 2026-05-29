"""
enso_predictor.py
=================
基于 bernoulli_dice (1415_will_you_be_alive) 的概率模拟思想，
实现 ENSO 事件的蒙特卡洛概率预测与不确定性量化。

科学背景
--------
ENSO 预测面临根本性的不确定性，来源于：
1. 混沌初值敏感性（Lorenz 型蝴蝶效应）；
2. 模式参数的不确定性（海气耦合强度、热通量系数等）；
3. 季节性随机强迫（大气内部变率、MJO 等）。

蒙特卡洛方法通过在参数空间和初值空间中大量采样，
统计性地估计未来 ENSO 状态的概率分布，为决策提供置信区间。

核心公式
--------
1. El Niño 发生的条件概率（基于历史统计）：
   
   P(Niño3.4 > 0.5°C | month = m) = N_events(m) / N_total(m)

2. 贝叶斯更新：
   给定观测 O 后，更新 ENSO 状态概率：
   
   P(S_i | O) = P(O | S_i) * P(S_i) / Σ_j P(O | S_j) * P(S_j)

3. 集合预测方差：
   对于 N 个集合成员 {x_k}：
   
   μ = (1/N) Σ x_k
   σ² = (1/(N-1)) Σ (x_k - μ)²

4. 预测 skill（均方根误差）：
   
   RMSE = √( (1/N_t) Σ_{t=1}^{N_t} (forecast_t - obs_t)² )

5. 持续概率（类比 Bernoulli 骰子问题）：
   若当前为 El Niño，下一季度仍为 El Niño 的概率：
   
   P(persist) = p_0 + (1 - p_0) * exp(-τ/τ_d)

   其中 p_0 为基准持续概率，τ_d 为衰减速率。
"""

import numpy as np
from typing import Tuple, List, Optional


class ENSOState:
    """ENSO 状态枚举"""
    STRONG_NINO = 2
    WEAK_NINO = 1
    NEUTRAL = 0
    WEAK_NINA = -1
    STRONG_NINA = -2


def classify_enso_state(nino34: float) -> int:
    """
    根据 Niño 3.4 指数分类 ENSO 状态。

    阈值（°C）：
    - 强 El Niño : > 1.5
    - 弱 El Niño : 0.5 ~ 1.5
    - 中性       : -0.5 ~ 0.5
    - 弱 La Niña : -1.5 ~ -0.5
    - 强 La Niña : < -1.5
    """
    if nino34 > 1.5:
        return ENSOState.STRONG_NINO
    elif nino34 > 0.5:
        return ENSOState.WEAK_NINO
    elif nino34 > -0.5:
        return ENSOState.NEUTRAL
    elif nino34 > -1.5:
        return ENSOState.WEAK_NINA
    else:
        return ENSOState.STRONG_NINA


def state_name(state: int) -> str:
    """返回 ENSO 状态的名称。"""
    names = {
        ENSOState.STRONG_NINO: "Strong El Nino",
        ENSOState.WEAK_NINO: "Weak El Nino",
        ENSOState.NEUTRAL: "Neutral",
        ENSOState.WEAK_NINA: "Weak La Nina",
        ENSOState.STRONG_NINA: "Strong La Nina",
    }
    return names.get(state, "Unknown")


def transition_probability(current_state: int,
                           month: int,
                           transition_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算 ENSO 状态转移概率。

    参数
    ----
    current_state : int
        当前 ENSO 状态。
    month : int
        当前月份（1-12），用于季节调制。
    transition_matrix : np.ndarray, optional
        5x5 转移矩阵。若未提供，使用经验估计。

    返回
    ----
    probs : np.ndarray, shape (5,)
        下一时刻各状态的概率。
    """
    if transition_matrix is None:
        # 经验转移矩阵（简化）：偏向持续当前状态，冬季锁相增强
        base_matrix = np.array([
            # StrongNino WeakNino Neutral WeakNina StrongNina
            [0.50, 0.30, 0.15, 0.04, 0.01],  # StrongNino
            [0.20, 0.45, 0.25, 0.08, 0.02],  # WeakNino
            [0.08, 0.18, 0.48, 0.18, 0.08],  # Neutral
            [0.02, 0.08, 0.25, 0.45, 0.20],  # WeakNina
            [0.01, 0.04, 0.15, 0.30, 0.50],  # StrongNina
        ])

        # 季节调制：冬季（11-1月）增强锁相
        if month in [11, 12, 1]:
            season_factor = 1.3
        elif month in [5, 6, 7]:
            season_factor = 0.8
        else:
            season_factor = 1.0

        # 调整对角元
        for i in range(5):
            base_matrix[i, i] *= season_factor
            # 归一化
            base_matrix[i] /= np.sum(base_matrix[i])

        transition_matrix = base_matrix

    state_idx = current_state + 2  # map -2..2 to 0..4
    state_idx = max(0, min(4, state_idx))
    return transition_matrix[state_idx]


def monte_carlo_enso_forecast(nino34_current: float,
                              month_current: int,
                              n_ensemble: int = 1000,
                              n_months: int = 12,
                              noise_std: float = 0.3) -> dict:
    """
    使用蒙特卡洛方法生成 ENSO 集合预测。

    参数
    ----
    nino34_current : float
        当前 Niño 3.4 指数（℃）。
    month_current : int
        当前月份（1-12）。
    n_ensemble : int
        集合成员数。
    n_months : int
        预测月数。
    noise_std : float
        随机强迫标准差。

    返回
    ----
    result : dict
        包含均值轨迹、方差、状态概率的预测结果。
    """
    if n_ensemble < 1:
        raise ValueError("n_ensemble must be positive")

    # 简化自回归模型：Niño3.4_{t+1} = ρ * Niño3.4_t + ε_t
    rho = 0.85  # 月际自相关

    forecasts = np.zeros((n_ensemble, n_months))
    states = np.zeros((n_ensemble, n_months), dtype=int)

    for e in range(n_ensemble):
        nino = nino34_current
        month = month_current
        for m in range(n_months):
            noise = np.random.normal(0.0, noise_std)
            nino = rho * nino + noise
            # 边界处理
            nino = np.clip(nino, -4.0, 4.0)
            forecasts[e, m] = nino
            states[e, m] = classify_enso_state(nino)
            month = month % 12 + 1

    # 统计
    mean_trajectory = np.mean(forecasts, axis=0)
    std_trajectory = np.std(forecasts, axis=0)

    # 各状态概率
    state_probs = np.zeros((5, n_months))
    state_labels = [ENSOState.STRONG_NINA, ENSOState.WEAK_NINA,
                    ENSOState.NEUTRAL, ENSOState.WEAK_NINO,
                    ENSOState.STRONG_NINO]
    for i, s in enumerate(state_labels):
        state_probs[i] = np.mean(states == s, axis=0)

    return {
        "mean_trajectory": mean_trajectory,
        "std_trajectory": std_trajectory,
        "ensemble": forecasts,
        "state_probabilities": state_probs,
        "state_labels": [state_name(s) for s in state_labels],
    }


def forecast_skill(forecasts: np.ndarray, observations: np.ndarray) -> dict:
    """
    计算预测 skill 指标。

    参数
    ----
    forecasts : np.ndarray, shape (n_times,)
        预测值。
    observations : np.ndarray, shape (n_times,)
        观测值。

    返回
    ----
    metrics : dict
        RMSE, MAE, 相关系数等。
    """
    if forecasts.shape != observations.shape:
        raise ValueError("Shape mismatch")

    diff = forecasts - observations
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))

    # 皮尔逊相关系数
    f_mean, o_mean = np.mean(forecasts), np.mean(observations)
    numerator = np.sum((forecasts - f_mean) * (observations - o_mean))
    denom = np.sqrt(np.sum((forecasts - f_mean) ** 2) * np.sum((observations - o_mean) ** 2))
    correlation = numerator / denom if denom > 1e-14 else 0.0

    # 持续性 skill
    persistence = observations[:-1]
    persist_obs = observations[1:]
    persist_rmse = np.sqrt(np.mean((persistence - persist_obs) ** 2))
    skill_score = 1.0 - rmse / persist_rmse if persist_rmse > 1e-14 else 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "skill_score": float(skill_score),
    }


def probabilistic_event_forecast(nino34_current: float,
                                 month_current: int,
                                 target_state: int,
                                 lead_months: int = 6,
                                 n_trials: int = 10000) -> float:
    """
    计算未来特定月份出现特定 ENSO 状态的概率（类似 Bernoulli 骰子模拟）。

    参数
    ----
    nino34_current : float
        当前 Niño 3.4。
    month_current : int
        当前月份。
    target_state : int
        目标 ENSO 状态。
    lead_months : int
        预测提前期（月）。
    n_trials : int
        蒙特卡洛试验次数。

    返回
    ----
    probability : float
        目标状态出现的概率。
    """
    rho = 0.85
    noise_std = 0.3
    count = 0

    for _ in range(n_trials):
        nino = nino34_current
        month = month_current
        for _ in range(lead_months):
            nino = rho * nino + np.random.normal(0.0, noise_std)
            nino = np.clip(nino, -4.0, 4.0)
            month = month % 12 + 1
        if classify_enso_state(nino) == target_state:
            count += 1

    return count / n_trials
