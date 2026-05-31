"""
stochastic_inlet.py
===================
基于 117_brc_data 改造的随机入口条件生成模块。

工业反应器的入口条件（温度、压力、流量、组分浓度）存在随机波动。
本模块利用正态分布随机扰动模型，生成带有统计不确定性的入口数据，
用于 CFD 模拟的敏感性分析与不确定性量化（UQ）。

核心公式
--------
1. 入口温度扰动：
       T_in(t) = T̄_in + σ_T · ξ(t)
       ξ(t) ~ N(0, 1)

2. 入口浓度扰动（Fischer-Tropsch 合成气）：
       y_CO(t) = ȳ_CO + σ_y · ξ_1(t)
       y_H2(t) = ȳ_H2 + σ_y · ξ_2(t)
       约束：y_CO + y_H2 + y_inert = 1

3. 流量扰动：
       Q_in(t) = Q̄_in · (1 + σ_Q · ξ_3(t))

4. 入口条件的统计特征：
       μ = 样本均值
       σ = 样本标准差
       变异系数 CV = σ / μ

5. 伪随机数生成（Box-Muller 变换）：
       Z = √(-2 ln U_1) · cos(2π U_2)
       其中 U_1, U_2 ~ Uniform(0,1)。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Random inlet data generation (from 117_brc_data)
# ---------------------------------------------------------------------------

def generate_inlet_conditions(n_samples, T_mean=523.0, T_std=5.0,
                              yCO_mean=0.30, yH2_mean=0.60, y_std=0.02,
                              Q_mean=0.01, Q_std=0.001,
                              seed=42):
    """
    生成随机入口条件样本。

    Parameters
    ----------
    n_samples : int
        样本数。
    T_mean, T_std : float
        入口温度均值 [K] 与标准差 [K]。
    yCO_mean, yH2_mean, y_std : float
        CO、H2 摩尔分数均值与标准差。
    Q_mean, Q_std : float
        体积流率均值 [m³/s] 与标准差。
    seed : int
        随机种子。

    Returns
    -------
    data : dict
        包含 'T', 'yCO', 'yH2', 'Q', 'time' 的字典。
    """
    rng = np.random.default_rng(seed)

    T = T_mean + T_std * rng.standard_normal(n_samples)
    yCO_raw = yCO_mean + y_std * rng.standard_normal(n_samples)
    yH2_raw = yH2_mean + y_std * rng.standard_normal(n_samples)
    Q = Q_mean + Q_std * rng.standard_normal(n_samples)

    # 归一化摩尔分数约束
    y_total = yCO_raw + yH2_raw
    mask = y_total > 0.95
    yCO_raw[mask] = yCO_raw[mask] * 0.95 / y_total[mask]
    yH2_raw[mask] = yH2_raw[mask] * 0.95 / y_total[mask]

    yCO = np.clip(yCO_raw, 0.05, 0.50)
    yH2 = np.clip(yH2_raw, 0.10, 0.70)

    Q = np.clip(Q, 0.5 * Q_mean, 1.5 * Q_mean)
    T = np.clip(T, T_mean - 3 * T_std, T_mean + 3 * T_std)

    time = np.arange(n_samples, dtype=float)

    return {
        'T': T,
        'yCO': yCO,
        'yH2': yH2,
        'Q': Q,
        'time': time,
        'statistics': {
            'T_mean': float(np.mean(T)),
            'T_std': float(np.std(T)),
            'yCO_mean': float(np.mean(yCO)),
            'yH2_mean': float(np.mean(yH2)),
            'Q_mean': float(np.mean(Q)),
            'Q_cv': float(np.std(Q) / np.mean(Q)),
        }
    }


def generate_perturbed_profile(base_profile, mu_perturb=0.0, sigma_perturb=0.05,
                               seed=42):
    """
    对基准一维分布（如温度或浓度轴向分布）施加随机扰动。

    Parameters
    ----------
    base_profile : ndarray
        基准分布。
    mu_perturb, sigma_perturb : float
        扰动的相对均值与标准差。
    seed : int

    Returns
    -------
    perturbed : ndarray
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(base_profile, dtype=float)
    noise = rng.normal(mu_perturb, sigma_perturb, size=base.shape)
    perturbed = base * (1.0 + noise)
    return np.clip(perturbed, 0.0, None)
