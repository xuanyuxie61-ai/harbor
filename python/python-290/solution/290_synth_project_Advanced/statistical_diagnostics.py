"""
statistical_diagnostics.py - 统计分析与时序诊断模块

本模块对阿尔芬波-EP模拟结果进行统计分析，包括:
  - 增长率/阻尼率估计 (从时间序列)
  - 频谱分析 (FFT)
  - 统计检验 (Fisher 精确检验)
  - 差分分析 (DiD)

核心算法融合了以下种子项目:
  - rPMDD (1162): 时间序列分析、DiD 分析、Fisher 精确检验
  - ellipse_distance (329): Monte Carlo 统计量
  - circle_positive_distance (182): 曲线几何距离统计

物理背景:
  阿尔芬波不稳定性的增长率可以从场的振幅时间序列提取:
    |ψ(t)| ∝ exp(γ t)
    → γ = d/dt ln|ψ(t)|

  频谱分析给出模的频率和带宽:
    ψ̃(ω) = ∫ ψ(t) exp(-iωt) dt
    |ψ̃(ω)|² 的峰值对应本征模频率

  不同运行之间的统计比较:
    Fisher 精确检验: 比较两组模态分布是否显著不同
    DiD 分析: 比较 EP 加入前后的增长率变化

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np
from scipy import stats as scipy_stats


def estimate_growth_rate(time_array, amplitude_array, fit_start=None, fit_end=None):
    """
    从振幅时间序列估计指数增长率。

    对于不稳定性: |ψ(t)| = A₀ exp(γ t)
    → ln|ψ(t)| = ln A₀ + γ t

    线性拟合 ln|ψ| vs t 得到增长率 γ。

    参数:
      time_array: ndarray, 时间序列
      amplitude_array: ndarray, 振幅序列 (正值)
      fit_start: int or None, 拟合起始索引
      fit_end: int or None, 拟合结束索引

    返回:
      result: dict, 包含增长率、R²、误差等
    """
    time_array = np.asarray(time_array, dtype=float)
    amplitude_array = np.asarray(amplitude_array, dtype=float)

    # 过滤零值和负值
    valid = amplitude_array > 1e-30
    if np.sum(valid) < 3:
        return {'growth_rate': 0.0, 'r_squared': 0.0, 'error': float('inf')}

    t_fit = time_array[valid]
    a_fit = amplitude_array[valid]
    ln_a = np.log(a_fit)

    if fit_start is not None or fit_end is not None:
        n = len(t_fit)
        i_start = fit_start if fit_start is not None else 0
        i_end = fit_end if fit_end is not None else n
        t_fit = t_fit[i_start:i_end]
        ln_a = ln_a[i_start:i_end]

    if len(t_fit) < 2:
        return {'growth_rate': 0.0, 'r_squared': 0.0, 'error': float('inf')}

    # 线性最小二乘: ln|A| = γ t + ln A₀
    n_pts = len(t_fit)
    S_t = np.sum(t_fit)
    S_lnA = np.sum(ln_a)
    S_tt = np.sum(t_fit ** 2)
    S_t_lnA = np.sum(t_fit * ln_a)

    denom = n_pts * S_tt - S_t ** 2
    if abs(denom) < 1e-30:
        return {'growth_rate': 0.0, 'r_squared': 0.0, 'error': float('inf')}

    gamma = (n_pts * S_t_lnA - S_t * S_lnA) / denom
    ln_A0 = (S_lnA - gamma * S_t) / n_pts

    # R² 计算
    ln_a_pred = gamma * t_fit + ln_A0
    ss_res = np.sum((ln_a - ln_a_pred) ** 2)
    ss_tot = np.sum((ln_a - np.mean(ln_a)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    # 标准误差
    if n_pts > 2:
        se = np.sqrt(ss_res / (n_pts - 2))
        gamma_error = se * np.sqrt(n_pts / (n_pts * S_tt - S_t ** 2 + 1e-30))
    else:
        gamma_error = float('inf')

    return {
        'growth_rate': gamma,
        'ln_amplitude_0': ln_A0,
        'r_squared': r_squared,
        'standard_error': gamma_error,
        'is_growing': gamma > 0,
        'e_folding_time': 1.0 / abs(gamma) if abs(gamma) > 1e-30 else float('inf'),
    }


def spectral_analysis(time_array, signal_array, n_fft=None):
    """
    信号的频谱分析 (FFT)。

    提取阿尔芬波模的频率成分:
      ψ̃(ω) = Σ ψ(t_n) exp(-i ω t_n) Δt

    参数:
      time_array: ndarray, 时间序列
      signal_array: ndarray, 信号序列
      n_fft: int or None, FFT 点数

    返回:
      result: dict, 包含频率、功率谱、主要频率等
    """
    time_array = np.asarray(time_array, dtype=float)
    signal_array = np.asarray(signal_array, dtype=float)

    n = len(signal_array)
    if n < 4:
        return {'frequencies': np.array([]), 'power': np.array([]),
                'dominant_frequency': 0.0}

    dt = np.mean(np.diff(time_array))
    if dt <= 0:
        dt = 1.0

    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n)))

    # 加窗 (Hann 窗减少频谱泄漏)
    window = np.hanning(n)
    signal_windowed = signal_array * window

    # FFT
    fft_vals = np.fft.rfft(signal_windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    power = np.abs(fft_vals) ** 2 / n_fft

    # 排除直流分量
    if len(freqs) > 1:
        freqs = freqs[1:]
        power = power[1:]

    # 主要频率
    if len(power) > 0:
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx]
    else:
        dominant_freq = 0.0

    # 频谱重心 (加权平均频率)
    total_power = np.sum(power)
    if total_power > 0:
        mean_freq = np.sum(freqs * power) / total_power
        spectral_width = np.sqrt(np.sum((freqs - mean_freq) ** 2 * power) / total_power)
    else:
        mean_freq = 0.0
        spectral_width = 0.0

    return {
        'frequencies': freqs,
        'power': power,
        'dominant_frequency': dominant_freq,
        'mean_frequency': mean_freq,
        'spectral_width': spectral_width,
        'total_power': total_power,
    }


def fisher_exact_test_2x2(a, b, c, d):
    """
    2×2 列联表的 Fisher 精确检验。

    改编自 rPMDD (1162) 的 co-occurrence 分析。
    用于检验两组阿尔芬模态分布是否独立。

    列联表:
           | 特征A | 非A  |
      特征B|   a   |  b   |
      非B  |   c   |  d   |

    H₀: 两个特征独立

    参数:
      a, b, c, d: int, 列联表元素

    返回:
      result: dict, 包含 p 值、比值比等
    """
    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = scipy_stats.fisher_exact(table)

    return {
        'odds_ratio': odds_ratio,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'table': table,
    }


def difference_in_differences_analysis(before_control, before_treatment,
                                        after_control, after_treatment):
    """
    差分-差分 (DiD) 分析。

    改编自 rPMDD (1162) 的 DiD 方法。
    用于评估高能粒子对阿尔芬波增长率的因果效应。

    DiD = (After_treatment - Before_treatment) - (After_control - Before_control)

    参数:
      before_control: float, 对照组 (无EP) 加入前基线
      before_treatment: float, 实验组 (有EP) 加入前基线
      after_control: float, 对照组加入后
      after_treatment: float, 实验组加入后

    返回:
      result: dict, 包含 DiD 估计量等
    """
    treatment_change = after_treatment - before_treatment
    control_change = after_control - before_control
    did_estimate = treatment_change - control_change

    # 标准误差 (简化)
    se = np.sqrt(
        abs(after_treatment) + abs(before_treatment) +
        abs(after_control) + abs(before_control) + 1e-30
    ) * 0.1

    # t 统计量
    if se > 0:
        t_stat = did_estimate / se
    else:
        t_stat = 0.0

    return {
        'did_estimate': did_estimate,
        'treatment_change': treatment_change,
        'control_change': control_change,
        'standard_error': se,
        't_statistic': t_stat,
        'is_significant': abs(t_stat) > 1.96,
    }


def convergence_order_analysis(dx_values, error_values):
    """
    从不同网格间距的误差数据估计收敛阶。

    对于 p 阶方法: error ≈ C h^p
    → ln(error) ≈ ln(C) + p ln(h)

    线性拟合得到收敛阶 p。

    参数:
      dx_values: array_like, 网格间距序列
      error_values: array_like, 对应误差序列

    返回:
      order: float, 估计的收敛阶
      r_squared: float, 拟合优度
    """
    dx_values = np.asarray(dx_values, dtype=float)
    error_values = np.asarray(error_values, dtype=float)

    valid = (dx_values > 0) & (error_values > 0)
    if np.sum(valid) < 2:
        return 0.0, 0.0

    ln_h = np.log(dx_values[valid])
    ln_e = np.log(error_values[valid])

    n = len(ln_h)
    S_x = np.sum(ln_h)
    S_y = np.sum(ln_e)
    S_xx = np.sum(ln_h ** 2)
    S_xy = np.sum(ln_h * ln_e)

    denom = n * S_xx - S_x ** 2
    if abs(denom) < 1e-30:
        return 0.0, 0.0

    order = (n * S_xy - S_x * S_y) / denom
    ln_C = (S_y - order * S_x) / n

    # R²
    y_pred = order * ln_h + ln_C
    ss_res = np.sum((ln_e - y_pred) ** 2)
    ss_tot = np.sum((ln_e - np.mean(ln_e)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    return order, r_squared


def mode_decomposition(field_2d, n_theta):
    """
    将二维场分解为极向模分量。

    ψ(r, θ) = Σ_m ψ_m(r) exp(i m θ)

    通过 FFT 在极向方向分解:
      ψ_m(r) = (1/N_θ) Σ_k ψ(r, θ_k) exp(-i m θ_k)

    参数:
      field_2d: ndarray, shape (n_r, n_theta)
      n_theta: int, 极向点数

    返回:
      mode_amplitudes: ndarray, shape (n_r, n_theta//2+1), |ψ_m(r)|
      mode_numbers: ndarray, 极向模数
    """
    n_r = field_2d.shape[0]

    # 对每个径向位置做 FFT
    fft_result = np.fft.rfft(field_2d, axis=1)
    mode_amplitudes = np.abs(fft_result) / n_theta
    mode_numbers = np.arange(fft_result.shape[1])

    return mode_amplitudes, mode_numbers
