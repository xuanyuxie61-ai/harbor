"""
monte_carlo.py — 声子模式统计采样与 Monte Carlo 方法
====================================================

融合种子项目:
  - 195_coin_simulation: Bernoulli 试验, 运行统计,  streak 分析
    运行平均: a_j = (1/j) * sum_{i=1}^j v_i
    运行和: s_j = sum_{i=1}^j v_i
  - 1250_fjarri-attic_qsim_letter_2011: Wigner 表示 + 系综平均
    n_j(x) = <Psi_j^*(x) Psi_j(x)>_paths - M/(2V)

物理背景:
  声子占据数的统计分布:
    <n_lambda> = 1 / (exp(hbar*omega_lambda / k_B*T) - 1)  (Bose-Einstein)
    Var(n_lambda) = <n_lambda> * (<n_lambda> + 1)

  Monte Carlo 采样用于:
    1. 热力学平均: <A> = (1/N_samples) sum_s A(n^{(s)})
    2. 非谐声子散射率: 通过采样初始态计算散射矩阵元
    3. 热导率的统计估计

  Wigner 采样 (量子-经典对应):
    每个模式的初始振幅从 Wigner 分布采样:
    W(q,p) = (2/hbar) * tanh(hbar*omega/(2*k_B*T))
             * exp(-2*H(q,p)/(hbar*omega * coth(hbar*omega/(2*k_B*T))))
"""

import numpy as np
from typing import Tuple, Dict, List


def sample_phonon_occupations(
    omega: np.ndarray,
    temperature: float,
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Bose-Einstein 分布采样声子占据数。

    P(n) = n_BE^n / (1 + n_BE)^{n+1}  (几何分布)

    其中 n_BE = 1/(exp(hbar*omega/(k_B*T)) - 1)

    返回:
        samples: (n_samples, n_modes) 占据数样本
        mean_occupations: (n_modes,) 平均占据数
    """
    hbar = 1.0546e-34
    k_B = 1.3806e-23
    rng = np.random.RandomState(seed)
    n_modes = len(omega)

    # 计算平均占据数
    x = hbar * np.maximum(omega, 1e-20) / (k_B * max(temperature, 1e-10))
    x = np.minimum(x, 500.0)
    n_BE = np.where(x > 1e-10, 1.0 / (np.exp(x) - 1.0 + 1e-30), 0.0)

    # 几何分布采样
    samples = np.zeros((n_samples, n_modes))
    for m in range(n_modes):
        if n_BE[m] < 1e-10:
            samples[:, m] = 0
        else:
            p = 1.0 / (1.0 + n_BE[m])
            samples[:, m] = rng.geometric(p, n_samples) - 1

    return samples, n_BE


def compute_thermal_average(
    observable_func,
    omega: np.ndarray,
    temperature: float,
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo 计算热力学平均 <A>。

    <A> = (1/N) sum_s A({n_lambda^{(s)}})

    使用运行平均监测收敛 (融合 coin_simulation)。

    返回:
        mean_value: 平均值
        std_error: 标准误差
    """
    samples, n_BE = sample_phonon_occupations(omega, temperature, n_samples, seed)

    running_sum = 0.0
    running_sq_sum = 0.0
    values = np.zeros(n_samples)

    for s in range(n_samples):
        val = observable_func(samples[s], omega)
        values[s] = val
        running_sum += val
        running_sq_sum += val ** 2

    mean_val = running_sum / n_samples
    variance = running_sq_sum / n_samples - mean_val ** 2
    std_error = np.sqrt(max(variance, 0.0) / n_samples)

    return mean_val, std_error


def wigner_sample_initial_conditions(
    omega: np.ndarray,
    temperature: float,
    n_samples: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wigner 分布采样初始条件 (融合 qsim_letter 的 truncated Wigner)。

    对每个模式 lambda, 初始坐标和动量从 Wigner 函数采样:
      q_lambda ~ N(0, sigma_q^2)
      p_lambda ~ N(0, sigma_p^2)

    其中:
      sigma_q^2 = (hbar / (2*m*omega)) * coth(hbar*omega / (2*k_B*T))
      sigma_p^2 = (hbar*m*omega / 2) * coth(hbar*omega / (2*k_B*T))

    量子极限 (T->0): sigma_q^2 = hbar/(2*m*omega) (零点涨落)
    经典极限 (T->inf): sigma_q^2 = k_B*T / (m*omega^2)
    """
    hbar = 1.0546e-34
    k_B = 1.3806e-23
    rng = np.random.RandomState(seed)
    n_modes = len(omega)

    # 简化: 假设 m = 1 amu
    m = 1.6605e-27  # kg
    omega_safe = np.maximum(omega * 2 * np.pi * 1e12, 1e-10)  # THz -> rad/s

    x = hbar * omega_safe / (2.0 * k_B * max(temperature, 1e-10))
    x = np.minimum(x, 500.0)
    coth_x = np.where(x > 1e-10, 1.0 / np.tanh(x), 1.0 / np.maximum(x, 1e-10))

    sigma_q = np.sqrt(hbar / (2.0 * m * omega_safe) * coth_x)
    sigma_p = np.sqrt(hbar * m * omega_safe / 2.0 * coth_x)

    # 采样
    q_samples = np.zeros((n_samples, n_modes))
    p_samples = np.zeros((n_samples, n_modes))
    for m_idx in range(n_modes):
        q_samples[:, m_idx] = rng.normal(0, sigma_q[m_idx], n_samples)
        p_samples[:, m_idx] = rng.normal(0, sigma_p[m_idx], n_samples)

    return q_samples, p_samples


def running_statistics(
    values: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    运行统计量 (融合 coin_simulation 的 running_average, running_sum)。

    返回:
        running_mean: a_j = (1/j) * sum_{i=1}^j v_i
        running_sum: s_j = sum_{i=1}^j v_i
        running_std: 标准差的运行估计
    """
    n = len(values)
    running_sum = np.cumsum(values)
    running_mean = running_sum / np.arange(1, n + 1)
    running_sq_sum = np.cumsum(values ** 2)
    running_var = running_sq_sum / np.arange(1, n + 1) - running_mean ** 2
    running_var = np.maximum(running_var, 0.0)
    running_std = np.sqrt(running_var)

    return {
        'running_mean': running_mean,
        'running_sum': running_sum,
        'running_std': running_std,
    }


def streak_analysis(
    binary_sequence: np.ndarray,
) -> Dict[str, any]:
    """
    连续段分析 (融合 coin_simulation 的 streak 计算)。

    在声子 MC 中用于检测:
      - 连续占据事件 (热激发持续)
      - 连续空闲事件 (模式冻结)

    返回:
        max_streak: 最长连续段
        mean_streak: 平均连续段长度
        streaks: 所有连续段长度列表
    """
    if len(binary_sequence) == 0:
        return {'max_streak': 0, 'mean_streak': 0.0, 'streaks': []}

    streaks = []
    current = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] == binary_sequence[i - 1]:
            current += 1
        else:
            streaks.append(current)
            current = 1
    streaks.append(current)

    return {
        'max_streak': max(streaks),
        'mean_streak': np.mean(streaks),
        'streaks': streaks,
    }


def monte_carlo_thermal_conductivity(
    omega_all: np.ndarray,
    v_group_all: np.ndarray,
    temperature: float,
    volume: float,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo 估计热导率。

    kappa = <C * v^2 * tau> / (3*V)

    通过对声子占据数采样, 计算散射率的统计平均。
    """
    from thermal_transport import mode_heat_capacity, total_scattering_rate

    rng = np.random.RandomState(seed)
    n_total = len(omega_all)
    v_sq = np.sum(v_group_all ** 2, axis=1)

    samples, n_BE = sample_phonon_occupations(omega_all, temperature, n_samples, seed)

    kappa_samples = np.zeros(n_samples)
    for s in range(n_samples):
        # 对每个样本, 计算有效散射率 (考虑声子-声子散射增强)
        n_occ = samples[s]
        enhancement = 1.0 + 0.1 * n_occ  # 简化: 占据数增强散射
        gamma = total_scattering_rate(omega_all, temperature) * enhancement
        gamma = np.maximum(gamma, 1e-30)
        tau = 1.0 / gamma

        C = mode_heat_capacity(omega_all, temperature)
        kappa_samples[s] = np.sum(C * v_sq * tau) / (3.0 * volume)

    mean_kappa = np.mean(kappa_samples)
    std_kappa = np.std(kappa_samples) / np.sqrt(n_samples)

    return mean_kappa, std_kappa
