"""
neural_encoding.py
神经节细胞脉冲发放模式分析与编码

基于以下种子项目合成：
- 1357_trig_interp_basis: 三角插值基函数
- 200_collocation: Horner多项式求值
- 597_iplot: 字符串表达式解析

科学背景：
视网膜神经节细胞（RGC）将光感受器和双极细胞处理后的模拟信号
转换为脉冲序列（spike trains）。这种编码过程涉及：
1. 脉冲发放的周期性模式分析
2. 调谐曲线的高效计算
3. 时空编码的数学描述

关键公式：
- 发放率函数：r(t) = r_0 + Σ_k a_k * φ_k(t)
- 三角插值基：τ_k(x) = sin(kπx/2) / [k * sin(πx/2)]  （k为奇数）
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 三角插值基函数（基于1357_trig_interp_basis）
# =============================================================================

def trig_interp_basis(x: np.ndarray, k: int) -> np.ndarray:
    """
    计算三角插值的基函数 τ_k(x)。
    
    在周期等距节点上，三角插值多项式的Lagrange型基函数为：
    
    当k为奇数时：
        τ_k(x) = sin(k * π * x / 2) / [k * sin(π * x / 2)]
    
    当k为偶数时：
        τ_k(x) = sin(k * π * x / 2) / [k * tan(π * x / 2)]
    
    在x = 0处，通过极限可得 τ_k(0) = 1。
    
    这些基函数用于在周期区间上构造三角多项式：
        T(x) = Σ_{j=0}^{N-1} f_j * τ_N(x - x_j)
    其中x_j = 2j/N 为等距节点。
    
    参数:
        x: (M,) 评估点，x∈[-1,1]或任意实数（周期为2）
        k: 基函数阶数（正整数）
    
    返回:
        value: (M,) 基函数值
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    
    # 处理x=0处的奇点
    eps = 1e-14
    
    for i in range(len(x)):
        xi = x[i]
        if abs(xi) < eps:
            result[i] = 1.0
            continue
        
        denom = np.sin(np.pi * xi / 2.0)
        if abs(denom) < eps:
            # sin(Nπ/2) ≈ 0 的情况，需要特殊处理
            # 当 x ≈ 2m 时，sin(kπx/2) 和 sin(πx/2) 同时趋于0
            # 使用L'Hopital法则或泰勒展开
            result[i] = 1.0 if k % 2 == 1 else 0.0
            continue
        
        numer = np.sin(k * np.pi * xi / 2.0)
        
        if k % 2 == 1:
            result[i] = numer / (k * denom)
        else:
            tan_val = np.tan(np.pi * xi / 2.0)
            if abs(tan_val) < eps:
                result[i] = 0.0
            else:
                result[i] = numer / (k * tan_val)
    
    return result


def trig_interpolate_spike_pattern(
    spike_times: np.ndarray,
    spike_values: np.ndarray,
    t_eval: np.ndarray,
    period: float = 2.0
) -> np.ndarray:
    """
    使用三角插值重构周期性脉冲发放模式。
    
    对于周期为T的脉冲序列，在等距节点t_j = j*T/N上进行插值：
        f(t) = Σ_{j=0}^{N-1} f_j * τ_N(2t/T - 2j/N)
    
    参数:
        spike_times: (N,) 脉冲时间点（已归一化到[0,T]）
        spike_values: (N,) 脉冲幅度
        t_eval: (M,) 评估时间点
        period: 周期T
    
    返回:
        interpolated: (M,) 插值后的发放模式
    """
    N = len(spike_times)
    if N == 0:
        return np.zeros_like(t_eval)
    
    # 归一化到[-1,1]区间
    x_nodes = 2.0 * spike_times / period - 1.0
    x_eval = 2.0 * t_eval / period - 1.0
    
    result = np.zeros_like(t_eval)
    for j in range(N):
        # 每个节点贡献 τ_N(x - x_j)
        shift = x_eval - x_nodes[j]
        # 利用周期性将shift映射到[-1,1]附近
        shift = ((shift + 1.0) % 2.0) - 1.0
        result += spike_values[j] * trig_interp_basis(shift, N)
    
    return result


# =============================================================================
# Horner多项式求值（基于200_collocation）
# =============================================================================

def horner_polynomial_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    使用Horner方法（嵌套乘法）高效计算多项式值。
    
    对于多项式：
        p(x) = c_0 + c_1*x + c_2*x² + ... + c_m*x^m
    
    Horner形式：
        p(x) = (...((c_m * x + c_{m-1}) * x + c_{m-2}) * x + ... + c_1) * x + c_0
    
    时间复杂度：O(m*n)，其中m为多项式次数，n为评估点数量。
    
    参数:
        coeffs: (m+1,) 系数向量，c[i]为x^i的系数
        x: (n,) 评估点
    
    返回:
        p: (n,) 多项式值
    """
    x = np.asarray(x, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    m = len(coeffs) - 1
    
    if m < 0:
        return np.zeros_like(x)
    
    # Horner迭代
    p = np.full_like(x, coeffs[m])
    for k in range(m - 1, -1, -1):
        p = p * x + coeffs[k]
    
    return p


def compute_tuning_curve(
    stimulus_values: np.ndarray,
    coeffs: np.ndarray
) -> np.ndarray:
    """
    计算神经节细胞的刺激-响应调谐曲线。
    
    调谐曲线描述神经节细胞对不同刺激强度的响应：
        R(s) = p(s) = c_0 + c_1*s + c_2*s² + ... + c_m*s^m
    
    参数:
        stimulus_values: (n,) 刺激强度值
        coeffs: (m+1,) 多项式系数
    
    返回:
        response: (n,) 神经节细胞响应
    """
    return horner_polynomial_eval(coeffs, stimulus_values)


# =============================================================================
# 脉冲序列分析与编码
# =============================================================================

def analyze_spike_train(
    spike_times: np.ndarray,
    t_max: float,
    n_bins: int = 100
) -> dict:
    """
    分析神经节细胞脉冲序列的统计特性。
    
    计算指标：
    1. 平均发放率：r = N_spikes / t_max
    2. 发放率变异系数：CV = σ_ISI / μ_ISI
    3. 局部发放率（分箱直方图）
    4. 自相关函数（脉冲间隔分布）
    
    参数:
        spike_times: 脉冲时间数组（已排序）
        t_max: 记录总时长
        n_bins: 直方图分箱数
    
    返回:
        metrics: 包含各种统计指标的字典
    """
    n_spikes = len(spike_times)
    if n_spikes < 2:
        return {
            'mean_rate': n_spikes / t_max if t_max > 0 else 0.0,
            'cv_isi': 0.0,
            'isi_mean': 0.0,
            'isi_std': 0.0,
            'rate_histogram': np.zeros(n_bins),
            'bin_edges': np.linspace(0, t_max, n_bins + 1),
        }
    
    # 平均发放率
    mean_rate = n_spikes / t_max
    
    # 脉冲间隔（ISI）
    isi = np.diff(spike_times)
    isi_mean = float(np.mean(isi))
    isi_std = float(np.std(isi))
    cv_isi = isi_std / isi_mean if isi_mean > 1e-14 else 0.0
    
    # 分箱发放率
    hist, bin_edges = np.histogram(spike_times, bins=n_bins, range=(0, t_max))
    bin_width = t_max / n_bins
    rate_histogram = hist / bin_width  # 发放率 = 脉冲数 / 时间
    
    return {
        'mean_rate': mean_rate,
        'cv_isi': cv_isi,
        'isi_mean': isi_mean,
        'isi_std': isi_std,
        'rate_histogram': rate_histogram,
        'bin_edges': bin_edges,
        'n_spikes': n_spikes,
    }


def encoding_efficiency(
    spike_train: np.ndarray,
    stimulus: np.ndarray,
    dt: float
) -> dict:
    """
    计算神经编码效率（基于信息论）。
    
    使用脉冲计数编码模型，计算互信息下界：
        I(R;S) ≥ 0.5 * log₂(1 + SNR)
    
    其中SNR为信噪比，通过线性回归估计。
    
    参数:
        spike_train: (T,) 二进制脉冲序列
        stimulus: (T,) 刺激时间序列
        dt: 时间步长
    
    返回:
        metrics: 包含信息率、信噪比等指标
    """
    T = len(spike_train)
    if T == 0:
        return {'info_rate': 0.0, 'snr': 0.0, 'correlation': 0.0}
    
    # 计算发放率与刺激的互相关
    stimulus_mean = np.mean(stimulus)
    spike_mean = np.mean(spike_train)
    
    cov = np.mean((stimulus - stimulus_mean) * (spike_train - spike_mean))
    var_s = np.var(stimulus)
    var_r = np.var(spike_train)
    
    if var_s < 1e-14 or var_r < 1e-14:
        correlation = 0.0
        snr = 0.0
    else:
        correlation = cov / np.sqrt(var_s * var_r)
        snr = correlation ** 2 / (1.0 - correlation ** 2 + 1e-14)
    
    # 互信息下界（bits per spike）
    info_rate = 0.5 * np.log2(1.0 + snr) if snr > 0 else 0.0
    
    return {
        'info_rate_bits': float(info_rate),
        'snr': float(snr),
        'correlation': float(correlation),
        'mean_stimulus': float(stimulus_mean),
        'mean_spike_rate': float(spike_mean / dt),
    }
