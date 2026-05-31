"""
脑血流动力学 — 血细胞动力学与统计变异性模块

整合 jai_alai_simulation（排队博弈模拟）与 asa111（正态分布分位数算法），
模拟血细胞在血管网络分支点处的竞争排队行为及血流的统计涨落。

科学背景:
- 红细胞在毛细血管分叉处存在竞争效应：流速较快的分支更容易"捕获"红细胞。
- 可用队列竞争模型描述：N 个血细胞在分叉点的选择行为类似于竞争性排队。
- 血流存在生理涨落，可用正态分布描述脉动压力与流速的随机性。
- Fahraeus-Lindqvist 效应: 在微血管中，表观粘度随管径减小而下降。
"""

import numpy as np


def fahraeus_lindqvist_viscosity(diameter_um, Hct=0.45):
    """
    Fahraeus-Lindqvist 效应：微血管中血液表观粘度随管径变化。
    经验公式 (Pries et al., 1992):
        μ_app = μ_plasma * [1 + (μ_0.45 - 1) * ((1 - Hct)^C - 1) / ((1 - 0.45)^C - 1)]
    其中:
        μ_0.45 = 220 * exp(-1.3 * d) + 3.2 - 2.44 * exp(-0.06 * d^0.645)
        C = (0.8 + exp(-0.075*d)) * (-1 + 1/(1+10^(-11)*d^12)) + 1/(1+10^(-11)*d^12)
    d: 血管直径 [μm]
    """
    d = max(diameter_um, 1e-3)
    mu_plasma = 1.2e-3  # Pa·s
    mu_045 = 220.0 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-0.06 * d ** 0.645)
    C = (0.8 + np.exp(-0.075 * d)) * (-1.0 + 1.0 / (1.0 + 10 ** (-11) * d ** 12)) + \
        1.0 / (1.0 + 10 ** (-11) * d ** 12)
    denom = (1.0 - 0.45) ** C - 1.0
    if abs(denom) < 1e-14:
        denom = 1e-14
    mu_app = mu_plasma * (1.0 + (mu_045 - 1.0) * ((1.0 - Hct) ** C - 1.0) / denom)
    return max(mu_app, mu_plasma)


def hematocrit_partition(Q_parent, Q_daughter1, Q_daughter2, Hct_parent,
                         D_parent, D_d1, D_d2):
    """
    分叉处红细胞压积分配 (Pries 两相模型):
        Hct_d1 / Hct_d2 = (Q_d1 / Q_d2)^n * (D_d1 / D_d2)^m
    其中 n≈1.0, m≈0.5。
    """
    if Q_parent < 1e-14:
        return 0.0, 0.0
    ratio = (Q_daughter1 / (Q_daughter2 + 1e-14)) ** 1.0 * (D_d1 / (D_d2 + 1e-14)) ** 0.5
    Hct_d2 = Hct_parent * Q_parent / (Q_daughter1 * ratio + Q_daughter2 + 1e-14)
    Hct_d1 = Hct_d2 * ratio
    return Hct_d1, Hct_d2


# ---- ASA111 正态分布分位数 ----
def ppnd(p):
    """
    正态分布下侧分位数（AS 111 算法，Beasley & Springer, 1977）。
    输入: 0 < p < 1
    返回: z ~ N(0,1), P(Z <= z) = p
    """
    p = float(p)
    a0 = 2.50662823884
    a1 = -18.61500062529
    a2 = 41.39119773534
    a3 = -25.44106049637
    b1 = -8.47351093090
    b2 = 23.08336743743
    b3 = -21.06224101826
    b4 = 3.13082909833
    c0 = -2.78718931138
    c1 = -2.29796479134
    c2 = 4.85014127135
    c3 = 2.32121276858
    d1 = 3.54388924762
    d2 = 1.63706781897
    split = 0.42

    if p <= 0.0 or p >= 1.0:
        return 0.0, 1

    if abs(p - 0.5) <= split:
        r = (p - 0.5) ** 2
        value = (p - 0.5) * (((a3 * r + a2) * r + a1) * r + a0) / \
                ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.0)
        return value, 0
    else:
        if p > 0.5:
            r = np.sqrt(-np.log(1.0 - p))
        else:
            r = np.sqrt(-np.log(p))
        value = (((c3 * r + c2) * r + c1) * r + c0) / ((d2 * r + d1) * r + 1.0)
        if p < 0.5:
            value = -value
        return value, 0


def blood_flow_variability(mean_flow, std_fraction, n_samples, seed=42):
    """
    基于 ASA111 逆正态 CDF 的反变换采样生成血流速率样本。
    """
    np.random.seed(seed)
    u = np.random.uniform(1e-6, 1.0 - 1e-6, n_samples)
    z = np.array([ppnd(ui)[0] for ui in u])
    return mean_flow * (1.0 + std_fraction * z)


# ---- Jai-Alai 竞争排队模型 → 血细胞分叉竞争 ----
def jai_alai_match(strength):
    """
    模拟一场血细胞分叉竞争。
    8 个"选手"代表不同流变学特性的血细胞群，按强度竞争。
    返回胜者索引与比赛场次。
    """
    n = len(strength)
    queue = list(range(n))
    games = 0
    while len(queue) >= 2:
        p1 = queue.pop(0)
        p2 = queue.pop(0)
        games += 1
        total = strength[p1] + strength[p2]
        if total < 1e-14:
            winner = p1
        else:
            winner = p1 if np.random.rand() < strength[p1] / total else p2
        queue.append(winner)
    return queue[0], games


def blood_cell_competition_simulation(strength, n_games):
    """
    多次模拟血细胞在分叉点的竞争结果。
    strength: 各细胞群竞争强度（与变形能力、流速正相关）
    n_games: 模拟次数
    返回各细胞群获胜次数统计。
    """
    stats = np.zeros(len(strength), dtype=int)
    for _ in range(n_games):
        winner, _ = jai_alai_match(strength)
        stats[winner] += 1
    return stats


def stochastic_pulsatile_flow(Q_mean, f_heart, t_array, amplitude=0.3, phase_noise_std=0.1):
    """
    随机脉动血流模型:
        Q(t) = Q_mean [1 + A sin(2π f t + φ)] + ε(t)
    其中相位 φ 与噪声 ε 由正态分布采样。
    """
    phi = np.random.normal(0.0, phase_noise_std)
    Q = Q_mean * (1.0 + amplitude * np.sin(2.0 * np.pi * f_heart * t_array + phi))
    noise = Q_mean * amplitude * 0.1 * np.random.randn(len(t_array))
    return np.maximum(Q + noise, 0.0)
