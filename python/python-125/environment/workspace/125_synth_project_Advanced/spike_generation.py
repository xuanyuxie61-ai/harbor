"""
spike_generation.py
视网膜神经节细胞随机脉冲发放模拟

基于以下种子项目合成：
- 320_duel_simulation: 蒙特卡洛模拟
- 349_faure: Faure准随机序列生成

科学背景：
神经节细胞的脉冲发放具有随机性，通常用非齐次泊松过程建模：
    P(N(t) = k) = [∫₀ᵗ λ(s)ds]^k / k! * exp(-∫₀ᵗ λ(s)ds)

其中λ(t)为时变发放率，由双极细胞输入决定。

本模块实现：
1. 非齐次泊松过程的蒙特卡洛模拟（基于duel_simulation的随机方法）
2. Faure准随机序列用于高维神经参数空间采样
3. 脉冲序列的统计验证
"""

import numpy as np
from typing import Tuple


# =============================================================================
# Faure准随机序列（基于349_faure）
# =============================================================================

def _is_prime(n: int) -> bool:
    """判断整数n是否为素数。"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _next_prime_ge(n: int) -> int:
    """返回大于等于n的最小素数。"""
    while not _is_prime(n):
        n += 1
    return n


def _binomial_table_mod_p(max_n: int, p: int) -> np.ndarray:
    """
    生成模p的二项式系数表（Pascal三角形）。
    
    递推关系：
        C(i,j) = [C(i-1,j) + C(i-1,j-1)] mod p
    
    参数:
        max_n: 最大行数
        p: 素数模数
    
    返回:
        table: (max_n, max_n) 模p二项式系数矩阵
    """
    table = np.zeros((max_n, max_n), dtype=np.int64)
    table[0, 0] = 1 % p
    
    for i in range(1, max_n):
        table[i, 0] = 1 % p
        for j in range(1, i + 1):
            table[i, j] = (table[i - 1, j] + table[i - 1, j - 1]) % p
    
    return table


def faure_sequence_1d(key: int, p: int, max_digits: int = 20) -> float:
    """
    生成一维Faure准随机序列的第key个点。
    
    Faure序列基于p进制展开：
        key = Σ_{j=0}^{m} y_j * p^j
        quasi = Σ_{j=0}^{m} y_j / p^{j+1}
    
    参数:
        key: 序列索引（非负整数）
        p: 素数基
        max_digits: 最大p进制位数
    
    返回:
        quasi: [0,1]区间内的准随机数
    """
    quasi = 0.0
    p_power = 1.0 / p
    
    k = key
    for _ in range(max_digits):
        digit = k % p
        quasi += digit * p_power
        k //= p
        p_power /= p
        if k == 0:
            break
    
    return quasi


def faure_generate(
    dim_num: int,
    n: int,
    skip: int = 0
) -> np.ndarray:
    """
    批量生成高维Faure准随机序列点。
    
    第k维通过对y向量进行Pascal三角形（模p）变换得到：
        y_j^{(new)} = Σ_{i=j}^{m} y_i^{(old)} * C(i,j) mod p
    
    参数:
        dim_num: 空间维度
        n: 生成点数
        skip: 跳过的初始点数
    
    返回:
        r: (dim_num, n) 准随机点矩阵
    """
    p = _next_prime_ge(dim_num)
    max_digits = 20
    
    # 生成二项式系数表
    binom_table = _binomial_table_mod_p(max_digits + 1, p)
    
    r = np.zeros((dim_num, n), dtype=np.float64)
    
    for point_idx in range(n):
        key = skip + point_idx
        
        # p进制展开
        y = np.zeros(max_digits, dtype=np.int64)
        k = key
        pos = 0
        while k > 0 and pos < max_digits:
            y[pos] = k % p
            k //= p
            pos += 1
        
        # 第一维
        r[0, point_idx] = faure_sequence_1d(key, p, max_digits)
        
        # 高维：Pascal变换
        for dim in range(1, dim_num):
            y_new = np.zeros(max_digits, dtype=np.int64)
            for j in range(max_digits):
                for i in range(j, max_digits):
                    if i < binom_table.shape[0] and j < binom_table.shape[1]:
                        y_new[j] = (y_new[j] + y[i] * binom_table[i, j]) % p
            
            quasi = 0.0
            p_power = 1.0 / p
            for j in range(max_digits):
                quasi += y_new[j] * p_power
                p_power /= p
            
            r[dim, point_idx] = quasi
    
    return r


# =============================================================================
# 非齐次泊松过程脉冲发放（基于320_duel_simulation的蒙特卡洛方法）
# =============================================================================

def generate_inhomogeneous_poisson_spikes(
    rate_func: callable,
    t_start: float,
    t_end: float,
    dt: float = 0.001,
    max_rate: float = None,
    seed: int = 42
) -> np.ndarray:
    """
    使用 thinning 方法生成非齐次泊松过程的脉冲序列。
    
    Thinning算法：
    1. 生成速率为λ_max的齐次泊松过程
    2. 对每个候选脉冲，以概率 λ(t)/λ_max 接受
    
    数学原理：
    对于非齐次泊松过程，强度函数为λ(t)，
    若在[t, t+dt]内以概率λ_max*dt生成候选事件，
    然后以概率λ(t)/λ_max保留，则保留的事件服从强度λ(t)的泊松过程。
    
    参数:
        rate_func: 时变发放率函数 λ(t)
        t_start: 起始时间
        t_end: 结束时间
        dt: 时间步长
        max_rate: 最大发放率（若None则自动估计）
        seed: 随机种子
    
    返回:
        spike_times: 脉冲时间数组
    """
    np.random.seed(seed)
    
    # 估计最大发放率
    if max_rate is None:
        t_samples = np.linspace(t_start, t_end, 1000)
        rate_samples = np.array([rate_func(t) for t in t_samples])
        max_rate = np.max(rate_samples) * 1.2  # 留20%余量
        max_rate = max(max_rate, 1e-6)
    
    # 生成齐次泊松过程的候选脉冲
    n_steps = int((t_end - t_start) / dt)
    candidate_spikes = []
    
    for i in range(n_steps):
        t = t_start + i * dt
        # 在[t, t+dt]内以概率λ_max*dt生成脉冲
        if np.random.random() < max_rate * dt:
            candidate_spikes.append(t + np.random.random() * dt)
    
    # Thinning：以概率 λ(t)/λ_max 接受
    spike_times = []
    for t in candidate_spikes:
        rate_t = rate_func(t)
        if rate_t > max_rate:
            rate_t = max_rate
        if np.random.random() < rate_t / max_rate:
            spike_times.append(t)
    
    return np.array(spike_times, dtype=np.float64)


def simulate_rgc_spike_train(
    bipolar_response: np.ndarray,
    time_array: np.ndarray,
    baseline_rate: float = 5.0,
    gain: float = 20.0,
    refractory_period: float = 0.002,
    seed: int = 42
) -> np.ndarray:
    """
    模拟视网膜神经节细胞的脉冲发放。
    
    发放率模型：
        λ(t) = baseline + gain * max(0, R_bipolar(t))
    
    其中R_bipolar(t)为双极细胞响应（已归一化）。
    
    考虑不应期：脉冲发生后refractory_period内不发放。
    
    参数:
        bipolar_response: (N,) 双极细胞响应时间序列
        time_array: (N,) 对应的时间点
        baseline_rate: 基线发放率 (Hz)
        gain: 响应增益
        refractory_period: 不应期 (s)
        seed: 随机种子
    
    返回:
        spike_times: 脉冲时间数组
    """
    np.random.seed(seed)
    
    dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.001
    N = len(bipolar_response)
    
    # 发放率
    rate = baseline_rate + gain * np.maximum(0.0, bipolar_response)
    rate = np.maximum(rate, 0.0)
    
    max_rate = np.max(rate) * 1.2
    max_rate = max(max_rate, 1e-6)
    
    spike_times = []
    last_spike = -refractory_period - 1.0
    
    for i in range(N):
        t = time_array[i]
        if t - last_spike < refractory_period:
            continue
        
        # 当前时刻的发放概率
        p_spike = rate[i] * dt
        p_spike = min(p_spike, 1.0)
        
        if np.random.random() < p_spike:
            spike_times.append(t)
            last_spike = t
    
    return np.array(spike_times, dtype=np.float64)


# =============================================================================
# 高维神经参数空间采样（基于Faure序列）
# =============================================================================

def sample_neural_parameter_space(
    n_samples: int,
    param_ranges: dict,
    skip: int = 100
) -> dict:
    """
    使用Faure准随机序列在神经参数空间中进行均匀采样。
    
    参数:
        n_samples: 采样点数
        param_ranges: 参数字典，每个键对应(min, max)范围
        skip: 跳过的初始点数（消除初始偏差）
    
    返回:
        samples: 每个参数对应的采样值数组
    """
    dim_num = len(param_ranges)
    param_names = list(param_ranges.keys())
    
    # 生成Faure序列
    faure_points = faure_generate(dim_num, n_samples, skip)
    
    samples = {}
    for i, name in enumerate(param_names):
        pmin, pmax = param_ranges[name]
        samples[name] = pmin + faure_points[i, :] * (pmax - pmin)
    
    return samples
