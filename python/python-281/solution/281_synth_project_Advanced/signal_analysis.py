"""
signal_analysis.py
==================
扩散信号分析与弛豫时间提取模块。
融合项目: 1064_ajakef_Earthquake_Infrasound_Paper (波形分析与阵列响应),
         1127_nschawor_eeg-mu-alpha-development (频谱分析与特征提取)

核心内容:
  1. 浓度时间序列的频谱分析 (DFT/PSD)
  2. 弛豫时间提取 (指数衰减拟合)
  3. 扩散系数的频率域反演
  4. 阵列响应函数 (类似地震学)
  5. 小波型多尺度分析

应用:
  从浓度-时间曲线中提取:
  - 特征扩散时间 τ_D = L²/D
  - 弛豫模式 (对应矩阵特征值)
  - 频率依赖的有效扩散系数
"""

import math
import numpy as np


def discrete_fourier_transform(signal):
    """
    离散 Fourier 变换 (DFT)。
    融合项目 1127: PSD 频谱分析。

    X_k = Σ_{n=0}^{N-1} x_n * exp(-2πi*k*n/N)

    Parameters
    ----------
    signal : ndarray, shape (N,)
        时域信号

    Returns
    -------
    spectrum : ndarray, shape (N,)
        频域信号 (复数)
    frequencies : ndarray
        归一化频率
    """
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * math.pi * k * n / N)
    spectrum = M @ signal
    frequencies = n / N
    return spectrum, frequencies


def power_spectral_density(signal, dt=1.0):
    """
    功率谱密度 (PSD) 估计。
    融合项目 1127: fig1_sensor_psd.py。

    PSD(f) = |X(f)|² / (N * dt)

    使用 Welch 方法的简化版本。

    Parameters
    ----------
    signal : ndarray
        时域信号
    dt : float
        采样间隔

    Returns
    -------
    frequencies : ndarray
        频率 [Hz]
    psd : ndarray
        功率谱密度
    """
    N = len(signal)
    # 去均值
    signal_centered = signal - np.mean(signal)

    # DFT
    spectrum, _ = discrete_fourier_transform(signal_centered)

    # 单边 PSD
    psd_full = np.abs(spectrum) ** 2 / (N * dt)
    # 取正频率部分
    n_pos = N // 2
    frequencies = np.arange(n_pos) / (N * dt)
    psd = psd_full[:n_pos] * 2.0  # 补偿负频率能量

    return frequencies, psd


def extract_relaxation_times(time_series, time_points, n_modes=5):
    """
    从时间序列中提取弛豫时间。
    融合项目 1064: 波形特征提取。

    方法: Prony 分析 / 指数拟合
    信号模型: c(t) = c_∞ + Σ_k A_k * exp(-t/τ_k)

    使用对数递减法估计初始 τ, 然后最小二乘优化。

    Parameters
    ----------
    time_series : ndarray
        浓度时间序列
    time_points : ndarray
        对应时间点
    n_modes : int
        要提取的弛豫模式数

    Returns
    -------
    relaxation_times : ndarray
        弛豫时间 τ_k
    amplitudes : ndarray
        对应振幅 A_k
    residual : float
        拟合残差
    """
    N = len(time_series)
    if N < 2 * n_modes + 1:
        return np.array([0.0]), np.array([0.0]), float('inf')

    # 平衡值估计 (取最后 10% 的均值)
    n_tail = max(1, N // 10)
    c_inf = np.mean(time_series[-n_tail:])

    # 去趋势信号
    signal = time_series - c_inf

    # 对数递减法估计主弛豫时间
    # 找信号过零点或半衰期
    abs_signal = np.abs(signal)
    if abs_signal[0] > 1e-14:
        half_level = abs_signal[0] / 2.0
        # 找首次衰减到半值的时间
        tau_est = time_points[-1]  # 默认
        for i in range(1, N):
            if abs_signal[i] < half_level:
                # 线性插值
                if abs_signal[i-1] > abs_signal[i]:
                    frac = (half_level - abs_signal[i]) / max(abs_signal[i-1] - abs_signal[i], 1e-30)
                    tau_est = time_points[i-1] + frac * (time_points[i] - time_points[i-1])
                break
        tau_est = max(tau_est, time_points[1] - time_points[0])
    else:
        tau_est = time_points[-1] / 5.0

    # 生成初始弛豫时间猜测 (对数均匀分布)
    tau_min = (time_points[-1] - time_points[0]) / N
    tau_max = time_points[-1] - time_points[0]
    relaxation_times = np.logspace(
        math.log10(max(tau_min, 1e-10)),
        math.log10(max(tau_max, 1e-10)),
        n_modes
    )

    # 最小二乘拟合振幅
    # 构建 Vandermonde 型矩阵: V[i,k] = exp(-t_i/τ_k)
    V = np.zeros((N, n_modes))
    for k in range(n_modes):
        V[:, k] = np.exp(-time_points / relaxation_times[k])

    # 最小二乘: V * A = signal
    try:
        amplitudes, _, _, _ = np.linalg.lstsq(V, signal, rcond=None)
        fitted = V @ amplitudes + c_inf
        residual = float(np.linalg.norm(fitted - time_series) / max(np.linalg.norm(time_series), 1e-14))
    except np.linalg.LinAlgError:
        amplitudes = np.zeros(n_modes)
        residual = float('inf')

    # 按弛豫时间排序
    order = np.argsort(relaxation_times)
    relaxation_times = relaxation_times[order]
    amplitudes = amplitudes[order]

    return relaxation_times, amplitudes, residual


def diffusion_coefficient_from_relaxation(tau, L_char, mode_number=1):
    """
    从弛豫时间反演扩散系数。

    对于球形颗粒中的扩散, 第 n 个弛豫模式:
    τ_n = R² / (n² * π² * D)

    反演: D = R² / (n² * π² * τ)

    Parameters
    ----------
    tau : float
        弛豫时间 [s]
    L_char : float
        特征长度 [m]
    mode_number : int
        模式编号

    Returns
    -------
    float
        扩散系数 [m²/s]
    """
    if tau <= 0 or L_char <= 0:
        return 0.0
    return L_char ** 2 / (mode_number ** 2 * math.pi ** 2 * tau)


def array_response_function(sensor_positions, source_position, wave_speed=1.0):
    """
    传感器阵列响应函数。
    融合项目 1064: Fig3_array_response.py。

    用于分析多位置测量的相干性和延迟。
    在电池诊断中, 可用于分析不同深度位置浓度波传播的延迟。

    beam(θ) = |Σ_i exp(i*ω*(τ_i(θ)))|² / N²

    Parameters
    ----------
    sensor_positions : ndarray, shape (N, d)
        传感器位置 (此处为径向坐标)
    source_position : ndarray, shape (d,)
        源位置
    wave_speed : float
        波速 (此处为扩散波速 ≈ √(D/τ))

    Returns
    -------
    delays : ndarray
        各传感器相对于源的传播延迟
    """
    distances = np.sqrt(np.sum((sensor_positions - source_position) ** 2, axis=1))
    delays = distances / max(wave_speed, 1e-14)
    return delays


def multiscale_decomposition(signal, n_levels=3):
    """
    多尺度分解 (简化小波分析)。
    融合项目 1127: 信号多尺度特征分析。

    使用 Haar 型分层平均:
    - 粗尺度: 滑动平均
    - 细节: 信号 - 粗尺度

    Parameters
    ----------
    signal : ndarray
        输入信号
    n_levels : int
        分解层数

    Returns
    -------
    approximations : list of ndarray
        各层近似 (粗尺度)
    details : list of ndarray
        各层细节
    """
    approximations = [signal.copy()]
    details = []

    current = signal.copy()
    for level in range(n_levels):
        N = len(current)
        if N < 4:
            break

        # 滑动平均 (窗口 = 2)
        approx = np.zeros(N // 2)
        detail = np.zeros(N // 2)
        for i in range(N // 2):
            approx[i] = 0.5 * (current[2*i] + current[2*i + 1])
            detail[i] = 0.5 * (current[2*i] - current[2*i + 1])

        approximations.append(approx)
        details.append(detail)
        current = approx

    return approximations, details


def concentration_autocorrelation(time_series, max_lag=None):
    """
    浓度时间序列的自相关函数。

    C(τ) = <c(t)*c(t+τ)> / <c²>

    衰减时间 = 积分时间尺度 T_int = ∫₀^∞ C(τ) dτ

    Parameters
    ----------
    time_series : ndarray
        浓度时间序列
    max_lag : int, optional
        最大滞后步数

    Returns
    -------
    lags : ndarray
        滞后步数
    autocorr : ndarray
        归一化自相关
    integral_timescale : float
        积分时间尺度 (以步数计)
    """
    N = len(time_series)
    if max_lag is None:
        max_lag = N // 2

    signal = time_series - np.mean(time_series)
    var = np.var(time_series)

    if var < 1e-30:
        return np.arange(max_lag), np.ones(max_lag), 0.0

    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag >= N:
            break
        autocorr[lag] = np.mean(signal[:N-lag] * signal[lag:]) / var

    # 积分时间尺度
    dt_lag = 1.0  # 步长
    T_int = 0.0
    for lag in range(1, max_lag):
        if autocorr[lag] < 0:
            break
        T_int += autocorr[lag] * dt_lag

    return np.arange(max_lag), autocorr, T_int


def effective_diffusion_frequency(omega, L_char):
    """
    频率依赖的有效扩散系数。

    对于振荡边界条件 c(R,t) = c₀ + δc*exp(iωt),
    有效扩散渗透深度: δ = √(2D/ω)

    当 δ << R 时, 只有表层参与扩散 (高频极限)。
    当 δ >> R 时, 整个颗粒准静态 (低频极限)。

    Parameters
    ----------
    omega : float or ndarray
        角频率 [rad/s]
    L_char : float
        特征长度 [m]

    Returns
    -------
    penetration_depth : float or ndarray
        扩散渗透深度 [m]
    """
    if isinstance(omega, np.ndarray):
        omega_safe = np.maximum(omega, 1e-20)
        # 这里返回的是相对值, 实际深度需要 D
        return L_char / np.sqrt(omega_safe)
    else:
        omega_safe = max(omega, 1e-20)
        return L_char / math.sqrt(omega_safe)
