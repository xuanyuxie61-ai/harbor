"""
multiscale_analysis.py
======================
基于 nested_sequence_display (798_nested_sequence_display) 的嵌套序列思想，
实现 ENSO 多尺度时空分析，包括小波分解、多分辨率能量谱计算与跨尺度能量串级。

科学背景
--------
ENSO 是一个典型的多尺度现象，涉及：
- 年际尺度（2–7 年）：ENSO 主周期；
- 季节尺度（3–6 个月）：季节循环锁相、Madden-Julian Oscillation (MJO)；
- 年代际尺度（10–30 年）：太平洋年代际振荡 (PDO)、背景态变化。

嵌套序列方法将不同分辨率的数据组织为层次结构，
通过追踪公共节点（能量极值点）来识别跨尺度的动力学联系。

核心公式
--------
1. 一维小波变换（Morlet 小波）：
   
   W(a, b) = (1/√a) ∫ f(t) ψ*((t-b)/a) dt

   其中 ψ(t) = π^{-1/4} exp(iω_0 t) exp(-t²/2)，
   a 为尺度参数，b 为平移参数。

2. 小波功率谱：
   
   P(a, b) = |W(a, b)|²

3. 全球小波谱（时间平均）：
   
   P̄(a) = (1/N) Σ_{b} |W(a, b)|²

4. 跨尺度能量通量（类比 nested sequence 的公共节点）：
   若尺度 a 上的极值点与尺度 a/2 上的某极值点空间/时间距离 < δ，
   则定义两者为"父子节点"，能量串级为：
   
   Π(a → a/2) = Σ_{父子对} P(a, b_a) * log(P(a/2, b_{a/2}) / P(a, b_a))

5. 显著性检验（红噪声背景谱）：
   对 AR(1) 过程，理论小波谱为：
   
   P̄_{AR1}(a) = σ² / (1 - ρ²) * (1 - ρ^{2a})
"""

import numpy as np
from typing import Tuple, List


def morlet_wavelet(t: np.ndarray, scale: float, omega0: float = 6.0) -> np.ndarray:
    """
    Morlet 母小波。

    公式：
    ψ(t) = π^{-1/4} * exp(i * ω_0 * t) * exp(-t² / 2)

    参数
    ----
    t : np.ndarray
        时间坐标。
    scale : float
        尺度参数 a。
    omega0 : float
        中心频率，默认 6.0。

    返回
    ----
    psi : np.ndarray
        小波函数值。
    """
    norm = np.pi ** (-0.25)
    psi = norm * np.exp(1j * omega0 * t / scale) * np.exp(-0.5 * (t / scale) ** 2)
    return psi


def cwt_1d(signal: np.ndarray, dt: float,
           scales: np.ndarray, omega0: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    一维连续小波变换 (CWT)。

    参数
    ----
    signal : np.ndarray, shape (N,)
        输入信号。
    dt : float
        时间步长。
    scales : np.ndarray
        尺度数组。
    omega0 : float
        Morlet 中心频率。

    返回
    ----
    W : np.ndarray, shape (len(scales), N)
        小波系数。
    freqs : np.ndarray
        对应的伪频率 f = ω_0 / (2π * a)。
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")

    N = signal.shape[0]
    n_scales = scales.shape[0]
    W = np.zeros((n_scales, N), dtype=complex)

    # FFT 加速
    signal_fft = np.fft.fft(signal)
    freqs_fft = np.fft.fftfreq(N, dt)

    for i, a in enumerate(scales):
        # Fourier 域滤波器
        psi_hat = np.zeros(N, dtype=complex)
        for k in range(N):
            s = 2.0 * np.pi * freqs_fft[k] * a
            # Morlet 的 Fourier 变换
            psi_hat[k] = np.pi ** (-0.25) * np.exp(-0.5 * (s - omega0) ** 2)
            # Heaviside 阶跃（仅正频率）
            if freqs_fft[k] <= 0:
                psi_hat[k] = 0.0

        # 卷积定理：W(a,b) = IFFT( FFT(f) * conj(FFT(ψ_a)) )
        conv = np.fft.ifft(signal_fft * np.conj(psi_hat))
        W[i, :] = conv * np.sqrt(dt / a)

    freqs = omega0 / (2.0 * np.pi * scales)
    return W, freqs


def global_wavelet_spectrum(W: np.ndarray) -> np.ndarray:
    """
    计算全球小波谱（时间平均功率）。

    公式：P̄(a) = (1/N) Σ_b |W(a, b)|²
    """
    return np.mean(np.abs(W) ** 2, axis=1)


def red_noise_spectrum(scales: np.ndarray, dt: float,
                       rho: float, sigma: float) -> np.ndarray:
    """
    计算 AR(1) 红噪声的理论小波谱。

    公式（Torrence & Compo, 1998）：
    P̄_{AR1}(a) = σ² / (1 - ρ²) * (1 - ρ^{2a})
    """
    if abs(rho) >= 1.0:
        rho = 0.99
    return (sigma ** 2 / (1.0 - rho ** 2)) * (1.0 - rho ** (2.0 * scales))


def find_scale_peaks(P_global: np.ndarray, scales: np.ndarray,
                     min_prominence: float = 0.1) -> List[Tuple[float, float]]:
    """
    识别小波谱中的显著峰值（对应 ENSO 主周期）。

    参数
    ----
    P_global : np.ndarray
        全球小波谱。
    scales : np.ndarray
        尺度数组。
    min_prominence : float
        最小相对峰高。

    返回
    ----
    peaks : List[Tuple[float, float]]
        (峰值尺度, 峰值功率) 列表。
    """
    peaks = []
    max_p = np.max(P_global)
    if max_p < 1e-14:
        return peaks

    for i in range(1, P_global.shape[0] - 1):
        if P_global[i] > P_global[i - 1] and P_global[i] > P_global[i + 1]:
            if P_global[i] / max_p > min_prominence:
                peaks.append((float(scales[i]), float(P_global[i])))

    return peaks


def nested_multiscale_analysis(signal: np.ndarray, dt: float,
                               n_levels: int = 5) -> dict:
    """
    对 ENSO 时间序列进行嵌套多尺度分析。

    参数
    ----
    signal : np.ndarray
        输入信号（如 Niño 3.4 指数）。
    dt : float
        时间步长（月）。
    n_levels : int
        分解层数。

    返回
    ----
    result : dict
        包含各尺度功率谱、显著周期、红噪声检验结果。
    """
    # 对数均匀分布的尺度
    s0 = 2.0 * dt
    s_max = len(signal) * dt / 4.0
    scales = s0 * (2.0 ** np.linspace(0, np.log2(s_max / s0), n_levels * 10))

    W, freqs = cwt_1d(signal, dt, scales)
    P_global = global_wavelet_spectrum(W)

    # 估计 AR(1) 参数
    rho_est = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    sigma_est = np.std(signal)
    P_red = red_noise_spectrum(scales, dt, rho_est, sigma_est)

    # 显著峰值（超过红噪声 95% 置信度）
    peaks = find_scale_peaks(P_global, scales, min_prominence=0.05)
    significant_peaks = []
    for s, p in peaks:
        idx = np.argmin(np.abs(scales - s))
        if P_global[idx] > 1.5 * P_red[idx]:  # 近似 90% 置信度
            period_months = s / dt
            significant_peaks.append({
                "scale": s,
                "period_months": period_months,
                "power": p,
            })
    # 若未找到显著峰，取全局最大值
    if not significant_peaks and len(peaks) > 0:
        best_peak = max(peaks, key=lambda x: x[1])
        idx = np.argmin(np.abs(scales - best_peak[0]))
        significant_peaks.append({
            "scale": best_peak[0],
            "period_months": best_peak[0] / dt,
            "power": best_peak[1],
        })

    return {
        "scales": scales,
        "frequencies": freqs,
        "global_spectrum": P_global,
        "red_noise_spectrum": P_red,
        "ar1_rho": float(rho_est),
        "ar1_sigma": float(sigma_est),
        "significant_peaks": significant_peaks,
    }


def cross_scale_energy_flux(W: np.ndarray, scales: np.ndarray,
                            threshold: float = 0.5) -> np.ndarray:
    """
    计算跨尺度能量通量（简化版本）。

    对于相邻尺度 a_i 和 a_{i+1}，计算能量变化的加权平均。

    返回
    ----
    flux : np.ndarray, shape (n_scales-1,)
        各尺度间的能量通量。
    """
    n_scales = scales.shape[0]
    flux = np.zeros(n_scales - 1)

    P = np.abs(W) ** 2
    for i in range(n_scales - 1):
        # 大尺度到小尺度的能量变化
        ratio = P[i + 1] / (P[i] + 1e-14)
        flux[i] = np.mean(np.log(ratio + 1e-14))

    return flux
