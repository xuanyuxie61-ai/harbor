"""
synaptic_encoding.py
突触编码与最优资源分配模块

融合 knapsack_rational (有理背包问题优化)
与 polynomial_multiply (离散卷积 / 多项式乘法)。

核心科学模型：
  脉冲时序编码:
    s(t) = sum_i w_i K(t - t_i)

  突触核函数 (alpha-synapse):
    K(t) = (t / tau_s) exp(-t / tau_s) * H(t)

  信息论目标：在能量约束下最大化编码信息量。
    max_{w} I(s; r)  s.t.  sum_i |w_i| <= E_budget

  其中互信息近似为:
    I(s; r) ≈ 0.5 * log det( I + SNR * C_s )
    C_s 为信号协方差矩阵, SNR = sigma_s^2 / sigma_n^2

  有理背包松弛：
    将每个突触权重视为"物品"，信息增益为"利润"，能量消耗为"重量"。
    贪心策略：按 利润密度 = delta_I_i / |w_i| 降序排列，依次选取。

  离散卷积 (polynomial_multiply 思想):
    给定两个脉冲序列 a = {a_k}, b = {b_k}，其突触后电位叠加为离散卷积:
      c_n = sum_k a_k b_{n-k}
    这等价于多项式乘法:
      A(z) = sum a_k z^k,  B(z) = sum b_k z^k
      C(z) = A(z) B(z)
"""

import numpy as np


class AlphaSynapse:
    """
    Alpha 突触核函数模型。
    """

    def __init__(self, tau_s=2.0):
        if tau_s <= 0:
            raise ValueError("tau_s must be positive.")
        self.tau_s = tau_s

    def kernel(self, t):
        """
        计算 alpha 核函数 K(t)。
        t: 时间差 (ms), 可以为数组
        """
        t = np.atleast_1d(t)
        K = np.zeros_like(t, dtype=float)
        mask = t > 0
        K[mask] = (t[mask] / self.tau_s) * np.exp(-t[mask] / self.tau_s)
        return K

    def convolve_spikes(self, spike_times, weights, t_grid):
        """
        将脉冲序列与 alpha 核做离散卷积。
        spike_times: 脉冲时刻列表
        weights: 对应权重
        t_grid: 输出时间网格
        """
        s = np.zeros_like(t_grid, dtype=float)
        for ti, wi in zip(spike_times, weights):
            dt = t_grid - ti
            s += wi * self.kernel(dt)
        return s


def polynomial_multiply_convolution(a, b):
    """
    使用多项式乘法思想计算离散卷积。
    融合 polynomial_multiply 的核心算法。

    输入序列 a, b 视为多项式系数:
      A(z) = a[0] + a[1] z + a[2] z^2 + ...
      B(z) = b[0] + b[1] z + b[2] z^2 + ...
    输出 c 为乘积多项式系数:
      C(z) = A(z) B(z) = c[0] + c[1] z + ...
      c[k] = sum_{i+j=k} a[i] b[j]
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    # 去除尾部零
    pn = len(a)
    while pn > 1 and np.isclose(a[pn - 1], 0.0):
        pn -= 1
    qn = len(b)
    while qn > 1 and np.isclose(b[qn - 1], 0.0):
        qn -= 1

    rn = pn + qn - 1
    c = np.zeros(rn)
    for i in range(pn):
        for j in range(qn):
            k = i + j
            if k < rn:
                c[k] += a[i] * b[j]
    return c


def rational_knapsack_encoding(profits, weights, budget):
    """
    有理背包问题求解突触权重最优分配。
    融合 knapsack_rational 的贪心策略。

    参数:
      profits: 每个权重 w_i 带来的信息增益 (长度 N)
      weights: 每个权重消耗的能量 |w_i| (长度 N, 非负)
      budget: 总能量预算

    返回:
      x: 分配比例 (0 <= x_i <= 1)
      total_mass: 实际消耗能量
      total_profit: 实际信息增益

    算法:
      1. 按 profit_density = profits / weights 降序排序
      2. 依次取完整权重，直到预算不足
      3. 最后一个取分数部分
    """
    profits = np.asarray(profits, dtype=float)
    weights = np.asarray(weights, dtype=float)
    N = len(profits)
    if N != len(weights):
        raise ValueError("profits and weights must have same length.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    if np.any(profits < 0):
        raise ValueError("profits must be non-negative.")

    # 处理零权重项
    safe_weights = np.where(weights == 0, 1e-12, weights)
    density = profits / safe_weights

    # 按密度降序排序 (保留原始索引)
    order = np.argsort(-density)

    x = np.zeros(N)
    mass = 0.0
    profit = 0.0

    for idx in order:
        wi = weights[idx]
        pi = profits[idx]
        if mass >= budget - 1e-12:
            x[idx] = 0.0
            continue
        if mass + wi <= budget:
            x[idx] = 1.0
            mass += wi
            profit += pi
        else:
            remaining = budget - mass
            if wi > 0:
                frac = remaining / wi
            else:
                frac = 1.0
            frac = np.clip(frac, 0.0, 1.0)
            x[idx] = frac
            mass = budget
            profit += pi * frac

    return x, mass, profit


def optimal_synaptic_weights(spike_times, signal_target, t_grid, tau_s=2.0, E_budget=10.0, sigma_noise=0.5):
    """
    基于有理背包问题求解最优突触权重。

    参数:
      spike_times: 候选脉冲时刻 (ms)
      signal_target: 目标信号 (在 t_grid 上)
      t_grid: 时间网格
      tau_s: 突触时间常数
      E_budget: 能量预算 (权重绝对值之和上限)
      sigma_noise: 观测噪声标准差

    返回:
      weights: 最优权重向量
      encoded_signal: 编码后的信号
      mutual_info: 近似互信息 (bits)
    """
    synapse = AlphaSynapse(tau_s)
    n_spikes = len(spike_times)
    if n_spikes == 0:
        return np.array([]), np.zeros_like(t_grid), 0.0

    # 构建每个脉冲的基函数响应
    basis = np.zeros((len(t_grid), n_spikes))
    for j, tj in enumerate(spike_times):
        basis[:, j] = synapse.kernel(t_grid - tj)

    # 计算每个候选权重的信息增益 (贪心近似)
    # 使用最小二乘残差减少作为利润代理
    profits = np.zeros(n_spikes)
    for j in range(n_spikes):
        # 该基函数对目标信号的最大投影
        proj = np.dot(basis[:, j], signal_target) / (np.dot(basis[:, j], basis[:, j]) + 1e-12)
        approx = proj * basis[:, j]
        profits[j] = np.linalg.norm(approx) ** 2

    # 归一化利润，避免过大
    max_profit = np.max(profits)
    if max_profit > 0:
        profits = profits / max_profit

    # 能量消耗近似为 |w_j|，先统一设为 1.0 (后续按背包结果缩放)
    candidate_weights = np.ones(n_spikes)
    x, _, _ = rational_knapsack_encoding(profits, candidate_weights, E_budget)

    # 根据背包分配结果，用最小二乘精细调整非零权重
    active = x > 0.01
    if not np.any(active):
        active[0] = True  # 至少保留一个

    basis_active = basis[:, active]
    # 正则化最小二乘
    lam = 0.01
    A = basis_active.T @ basis_active + lam * np.eye(basis_active.shape[1])
    b_vec = basis_active.T @ signal_target
    try:
        w_active = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        w_active = np.linalg.lstsq(basis_active, signal_target, rcond=None)[0]

    # 施加总能量约束 (L1 范数)
    l1_norm = np.sum(np.abs(w_active))
    if l1_norm > E_budget and l1_norm > 0:
        w_active = w_active * (E_budget / l1_norm)

    weights = np.zeros(n_spikes)
    weights[active] = w_active

    encoded_signal = basis @ weights

    # 近似互信息 (高斯近似)
    var_signal = np.var(encoded_signal)
    var_noise = sigma_noise ** 2
    if var_signal > 0 and var_noise > 0:
        snr = var_signal / var_noise
        mutual_info = 0.5 * np.log2(1.0 + snr)
    else:
        mutual_info = 0.0

    return weights, encoded_signal, mutual_info


def demo_encoding():
    """突触编码 demo。"""
    t_grid = np.linspace(0, 100, 1000)
    # 目标信号：低频正弦 + 噪声
    signal_target = 5.0 * np.sin(2.0 * np.pi * 0.05 * t_grid) + 0.5 * np.random.randn(len(t_grid))
    # 候选脉冲时刻 (泊松过程)
    np.random.seed(7)
    rate = 0.1  # per ms
    spike_times = []
    t = 0.0
    while t < 100.0:
        dt_spike = np.random.exponential(1.0 / rate)
        t += dt_spike
        if t < 100.0:
            spike_times.append(t)
    weights, encoded, mi = optimal_synaptic_weights(
        spike_times, signal_target, t_grid, tau_s=3.0, E_budget=8.0, sigma_noise=0.5
    )
    return weights, encoded, mi, spike_times
