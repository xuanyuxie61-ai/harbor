"""
spectrum_analysis.py
超连续谱特征提取与定量分析

科学背景:
  超连续谱（Supercontinuum）的定量表征涉及多个物理量:
    1. -20 dB 带宽（Bandwidth）
    2. 光谱平坦度（Spectral Flatness）
    3. 相干度（Coherence）
    4. 平均色散长度（Dispersion Length）
    5. 非线性长度（Nonlinear Length）
    6. 孤子阶数（Soliton Order）

  对于高相干超连续谱（如飞秒泵浦），相干度接近1；
  对于噪声种子放大（如皮秒泵浦），相干度下降，出现"相干塌陷"。
  本模块提供上述参数的稳健计算工具。
"""

import numpy as np
from typing import Tuple


def spectral_bandwidth(omega: np.ndarray, power_spectrum: np.ndarray,
                       method: str = "fwhm") -> float:
    """
    计算光谱带宽。

    支持方法:
        "fwhm": 半高全宽（-3 dB 带宽）
        "twenty_db": -20 dB 带宽
        "rms": 均方根带宽 sigma_omega = sqrt(<omega^2> - <omega>^2)

    公式（RMS 带宽）:
        <omega> = integral omega * P(omega) domega / integral P(omega) domega
        <omega^2> = integral omega^2 * P(omega) domega / integral P(omega) domega
        sigma_omega = sqrt(<omega^2> - <omega>^2)

    Parameters
    ----------
    omega : np.ndarray
        角频率数组（rad/s）。
    power_spectrum : np.ndarray
        功率谱密度 P(omega)（线性尺度）。
    method : str
        带宽计算方法。

    Returns
    -------
    float
        带宽（rad/s）。
    """
    if len(omega) != len(power_spectrum):
        raise ValueError("spectral_bandwidth: length mismatch")
    p = np.maximum(power_spectrum, 0.0)
    p_max = np.max(p)
    if p_max <= 0.0:
        return 0.0
    if method == "fwhm":
        threshold = p_max * 0.5
        indices = np.where(p > threshold)[0]
        if len(indices) == 0:
            return 0.0
        return float(omega[indices[-1]] - omega[indices[0]])
    elif method == "twenty_db":
        threshold = p_max * 0.01
        indices = np.where(p > threshold)[0]
        if len(indices) == 0:
            return 0.0
        return float(omega[indices[-1]] - omega[indices[0]])
    elif method == "rms":
        p_norm = p / (np.trapz(p, omega) + 1e-20)
        mean_o = np.trapz(omega * p_norm, omega)
        mean_o2 = np.trapz(omega ** 2 * p_norm, omega)
        return float(np.sqrt(max(mean_o2 - mean_o ** 2, 0.0)))
    else:
        raise ValueError(f"spectral_bandwidth: unknown method {method}")


def spectral_flatness(power_spectrum: np.ndarray) -> float:
    """
    计算光谱平坦度（几何平均/算术平均）。

    公式:
        F = exp( (1/N) * sum log(P_i) ) / ( (1/N) * sum P_i )

    平坦度越接近 1，光谱越平坦；接近 0 则光谱起伏剧烈。

    Parameters
    ----------
    power_spectrum : np.ndarray
        功率谱密度。

    Returns
    -------
    float
        平坦度因子 [0, 1]。
    """
    p = np.maximum(power_spectrum, 1e-20)
    geo_mean = np.exp(np.mean(np.log(p)))
    arith_mean = np.mean(p)
    if arith_mean <= 0.0:
        return 0.0
    return float(geo_mean / arith_mean)


def coherence_degree(spectrum_ensemble: np.ndarray) -> np.ndarray:
    """
    计算超连续谱的模态相干度。

    对于 N 次独立噪声实现，相干度定义为:
        g_12(omega) = | <E_i(omega)> |_i / sqrt( < |E_i(omega)|^2 >_i )

    其中 <.>_i 表示对噪声系综平均。相干度 g_12 在 [0, 1] 之间，
    接近 1 表示高相干，接近 0 表示完全非相干。

    公式:
        g(omega) = | sum_{i=1}^{N} E_i(omega) | / sqrt( N * sum_{i=1}^{N} |E_i(omega)|^2 )

    Parameters
    ----------
    spectrum_ensemble : np.ndarray
        频域振幅系综，形状 (n_realizations, n_omega)。

    Returns
    -------
    np.ndarray
        相干度数组，形状 (n_omega,)。
    """
    if spectrum_ensemble.ndim != 2:
        raise ValueError("coherence_degree: expected 2D array")
    mean_field = np.mean(spectrum_ensemble, axis=0)
    mean_power = np.mean(np.abs(spectrum_ensemble) ** 2, axis=0)
    coherence = np.abs(mean_field) / np.sqrt(mean_power + 1e-20)
    coherence = np.clip(coherence, 0.0, 1.0)
    return coherence


def dispersion_length(T0: float, beta2: float) -> float:
    """
    计算色散长度。

    公式:
        L_D = T0^2 / |beta2|

    其中 T0 为脉冲特征宽度，beta2 为群速度色散（GVD）。
    当 L_D >> L_NL 时，非线性效应主导；反之色散主导。

    Parameters
    ----------
    T0 : float
        特征脉宽（s），对于双曲正割脉冲为 1/e 半宽。
    beta2 : float
        GVD 系数（s^2/m），beta2 = d^2 beta / d omega^2。

    Returns
    -------
    float
        色散长度（m）。
    """
    if abs(beta2) < 1e-30:
        return 1e20
    return float(T0 ** 2 / abs(beta2))


def nonlinear_length(gamma: float, P0: float) -> float:
    """
    计算非线性长度。

    公式:
        L_NL = 1 / (gamma * P0)

    其中 gamma 为非线性系数，P0 为峰值功率。

    Parameters
    ----------
    gamma : float
        非线性系数（1/(W*m)）。
    P0 : float
        峰值功率（W）。

    Returns
    -------
    float
        非线性长度（m）。
    """
    if gamma <= 0.0 or P0 <= 0.0:
        return 1e20
    return float(1.0 / (gamma * P0))


def soliton_order(beta2: float, gamma: float, T0: float, P0: float) -> float:
    """
    计算基态孤子阶数。

    公式:
        N = sqrt( L_D / L_NL ) = sqrt( gamma * P0 * T0^2 / |beta2| )

    当 N = 1 时，脉冲为基态孤子，在传播中保持形状不变。
    当 N > 1 时，高阶孤子经历周期性压缩与分裂（孤子裂变），
    是超连续谱产生的主要机制之一。

    Parameters
    ----------
    beta2 : float
        GVD 系数。
    gamma : float
        非线性系数。
    T0 : float
        脉宽。
    P0 : float
        峰值功率。

    Returns
    -------
    float
        孤子阶数。
    """
    ld = dispersion_length(T0, beta2)
    lnl = nonlinear_length(gamma, P0)
    return float(np.sqrt(ld / lnl))


def fourier_limit_duration(bandwidth_hz: float, pulse_shape: str = "sech") -> float:
    """
    计算傅里叶变换极限脉宽。

    公式:
        对于 sech^2 脉冲: T_FWHM = 0.315 / Delta_f
        对于 Gaussian 脉冲: T_FWHM = 0.441 / Delta_f

    Parameters
    ----------
    bandwidth_hz : float
        光谱带宽（Hz）。
    pulse_shape : str
        脉冲形状。

    Returns
    -------
    float
        傅里叶极限 FWHM 脉宽（s）。
    """
    if bandwidth_hz <= 0.0:
        return 0.0
    if pulse_shape == "sech":
        return 0.315 / bandwidth_hz
    elif pulse_shape == "gaussian":
        return 0.441 / bandwidth_hz
    else:
        return 0.4 / bandwidth_hz


def spectral_snr(power_spectrum: np.ndarray, signal_band: Tuple[int, int]) -> float:
    """
    估算光谱信噪比（SNR）。

    公式:
        SNR = 10 * log10( P_signal_avg / P_noise_avg )

    Parameters
    ----------
    power_spectrum : np.ndarray
        功率谱。
    signal_band : tuple
        (start_idx, end_idx) 信号带索引。

    Returns
    -------
    float
        SNR（dB）。
    """
    start, end = signal_band
    p = np.maximum(power_spectrum, 1e-20)
    signal_power = np.mean(p[start:end])
    # 噪声功率取信号带外的平均
    noise_indices = list(range(0, start)) + list(range(end, len(p)))
    if len(noise_indices) == 0:
        noise_power = 1e-20
    else:
        noise_power = np.mean(p[noise_indices])
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_db)
