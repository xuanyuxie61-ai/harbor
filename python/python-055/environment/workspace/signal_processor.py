"""
signal_processor.py
基于种子项目 1268_toms243（复数对数计算）与
1071_shepard_interp_1d（Shepard 插值），
构建声纳回波信号的复包络处理与频谱分析模块。

科学背景：多波束声纳发射的调频（FM）脉冲信号经海底反射后，
接收信号可表示为复包络形式：

    r(t) = A(t) · exp(j·φ(t)) + n(t)

其中 A(t) 为振幅包络，φ(t) 为相位，n(t) 为加性高斯白噪声。
为提取传播时间信息，需对信号进行匹配滤波（matched filtering）：

    y(t) = ∫ r(τ) · s*(τ - t) dτ

其中 s(t) 为发射信号的复包络，* 表示复共轭。

此外，混响与噪声的统计特性常通过对数变换进行处理：
    z = log(w) = log|w| + j·arg(w)
这里复数对数算法（TOMS243）用于稳定计算信号频谱的对数幅度。

Shepard 插值则用于从离散频谱采样点重建连续频谱包络，
以便精确检测回波到达时间。
"""

import numpy as np


def complex_logarithm_toms243(z: complex) -> complex:
    """
    计算复数的自然对数（源自 toms243.m）。

    算法：通过缩放避免溢出/下溢，精确计算：
        log(z) = log|z| + j·arg(z)

    参数:
        z: 复数输入
    返回:
        复数对数值
    """
    a = float(z.real)
    b = float(z.imag)

    if abs(a) < 1e-300 and abs(b) < 1e-300:
        return complex(np.nan, np.nan)

    e = a / 2.0
    f = b / 2.0

    if abs(e) < 0.5 and abs(f) < 0.5:
        c = abs(2.0 * a) + abs(2.0 * b)
        if c < 1e-300:
            return complex(-700.0, 0.0)
        d = 8.0 * (a / c) * a + 8.0 * (b / c) * b
        c_val = 0.5 * (np.log(c) + np.log(d)) - np.log(np.sqrt(8.0))
    else:
        c = abs(e / 2.0) + abs(f / 2.0)
        if c < 1e-300:
            return complex(-700.0, 0.0)
        d = 0.5 * (e / c) * e + 0.5 * (f / c) * f
        c_val = 0.5 * (np.log(c) + np.log(d)) + np.log(np.sqrt(8.0))

    # 计算相位 arg(z)
    if a != 0.0 and abs(f) <= abs(e):
        if np.sign(a) >= 0:
            d_val = np.arctan(b / a)
        elif np.sign(b) >= 0:
            d_val = np.arctan(b / a) + np.pi
        else:
            d_val = np.arctan(b / a) - np.pi
    else:
        if b > 0:
            d_val = -np.arctan(a / b) + np.pi / 2.0
        elif b < 0:
            d_val = -np.arctan(a / b) - np.pi / 2.0
        else:
            d_val = 0.0 if a > 0 else np.pi

    return complex(c_val, d_val)


def shepard_interp_1d(xd: np.ndarray, yd: np.ndarray, p: float, xi: np.ndarray) -> np.ndarray:
    """
    一维 Shepard 插值（源自 shepard_interp_1d.m）。

    权重公式:
        w_i = 1 / |xi - x_i|^p
        若 xi 恰好等于某 x_i，则 w_i = 1，其余为 0。
        yi = Σ w_i · y_i / Σ w_i

    参数:
        xd: 数据点横坐标，形状 (nd,)
        yd: 数据点纵坐标，形状 (nd,)
        p:  距离幂次（通常 p=2）
        xi: 插值点横坐标，形状 (ni,)
    返回:
        yi: 插值结果，形状 (ni,)
    """
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    nd = len(xd)
    ni = len(xi)
    yi = np.zeros(ni, dtype=np.float64)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd) / nd
        else:
            w = np.abs(xi[i] - xd)
            z = np.where(w < 1e-15)[0]
            if len(z) > 0:
                w = np.zeros(nd)
                w[z[0]] = 1.0
            else:
                w = 1.0 / (w ** p)
                s = np.sum(w)
                if s > 0:
                    w = w / s
                else:
                    w = np.ones(nd) / nd
        yi[i] = np.dot(w, yd)

    return yi


class SonarSignalProcessor:
    """
    声纳回波信号处理器。
    """

    def __init__(self, fs: float = 48000.0, f0: float = 12000.0, bandwidth: float = 4000.0):
        """
        参数:
            fs:        采样率 (Hz)
            f0:        中心频率 (Hz)
            bandwidth: 信号带宽 (Hz)
        """
        self.fs = float(fs)
        self.f0 = float(f0)
        self.bandwidth = float(bandwidth)

    def generate_chirp_pulse(self, duration: float = 0.01) -> tuple:
        """
        生成线性调频（LFM）脉冲信号。

        瞬时频率:
            f(t) = f0 - B/2 + (B/T) · t,  t ∈ [0, T]
        相位:
            φ(t) = 2π · ∫ f(t) dt = 2π · [(f0 - B/2)·t + (B/(2T))·t²]
        """
        T = duration
        t = np.arange(0.0, T, 1.0 / self.fs)
        f_start = self.f0 - self.bandwidth / 2.0
        k = self.bandwidth / T
        phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t ** 2)
        s = np.exp(1j * phase)
        return t, s

    def matched_filter(self, received: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        匹配滤波器。

        公式:
            y[n] = Σ_k r[k] · s*[k - n]
        在频域实现：Y(f) = R(f) · S*(f)
        """
        n_recv = len(received)
        n_temp = len(template)
        n_fft = 1
        while n_fft < n_recv + n_temp - 1:
            n_fft *= 2

        R = np.fft.fft(received, n_fft)
        S = np.fft.fft(template, n_fft)
        Y = R * np.conj(S)
        y = np.fft.ifft(Y)
        return y[:n_recv]

    def compute_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        计算解析信号的包络（Hilbert 变换幅度）。

        公式:
            envelope = |s + j·H{s}|
        其中 H 为 Hilbert 变换，在频域实现为符号函数滤波。
        """
        n = len(signal)
        n_fft = 1
        while n_fft < n:
            n_fft *= 2
        S = np.fft.fft(signal, n_fft)
        # Hilbert 变换频域滤波器
        h = np.zeros(n_fft)
        h[0] = 1.0
        h[1:n_fft // 2] = 2.0
        if n_fft % 2 == 0:
            h[n_fft // 2] = 1.0
        S_h = S * h
        s_analytic = np.fft.ifft(S_h)
        envelope = np.abs(s_analytic[:n])
        return envelope

    def detect_peak_time(
        self,
        t: np.ndarray,
        envelope: np.ndarray,
        threshold_ratio: float = 0.3
    ) -> float:
        """
        检测回波包络峰值对应的传播时间。

        算法：
            1. 找到超过阈值的最大峰值；
            2. 在峰值附近使用 Shepard 插值精化时间估计。
        """
        threshold = threshold_ratio * np.max(envelope)
        peaks = np.where(envelope > threshold)[0]
        if len(peaks) == 0:
            return -1.0

        peak_idx = peaks[np.argmax(envelope[peaks])]
        # 在峰值附近 ±5 个采样点用 Shepard 插值精化
        half_win = min(5, peak_idx, len(t) - peak_idx - 1)
        if half_win < 2:
            return float(t[peak_idx])

        local_idx = np.arange(peak_idx - half_win, peak_idx + half_win + 1)
        local_t = t[local_idx]
        local_env = envelope[local_idx]

        # 在峰值位置附近进行高密度插值
        ti_fine = np.linspace(local_t[0], local_t[-1], 200)
        env_fine = shepard_interp_1d(local_t, local_env, p=2.0, xi=ti_fine)
        max_fine_idx = np.argmax(env_fine)
        return float(ti_fine[max_fine_idx])

    def compute_log_spectrum(self, signal: np.ndarray) -> tuple:
        """
        计算信号的复对数频谱。

        公式:
            Z[k] = log(FFT(s)[k])
        使用 TOMS243 复数对数算法保证数值稳定性。
        """
        n = len(signal)
        n_fft = 1
        while n_fft < n:
            n_fft *= 2
        S = np.fft.fft(signal, n_fft)
        # 使用 TOMS243 计算复对数
        log_spectrum = np.array([complex_logarithm_toms243(z) for z in S], dtype=complex)
        freqs = np.fft.fftfreq(n_fft, d=1.0 / self.fs)
        return freqs, log_spectrum

    def process_single_ping(
        self,
        received_signal: np.ndarray,
        pulse_duration: float = 0.01,
        noise_std: float = 0.01
    ) -> dict:
        """
        处理单 ping 回波信号的完整流程。

        返回字典包含：
            - envelope: 包络
            - peak_time: 峰值时间
            - log_spectrum: 复对数频谱
            - snr_db: 信噪比 (dB)
        """
        # 生成参考模板
        _, template = self.generate_chirp_pulse(pulse_duration)

        # 匹配滤波
        mf_output = self.matched_filter(received_signal, template)

        # 包络检测
        envelope = self.compute_envelope(mf_output)

        # 时间轴
        t = np.arange(len(received_signal)) / self.fs

        # 峰值检测
        peak_time = self.detect_peak_time(t, envelope)

        # 复对数频谱
        freqs, log_spec = self.compute_log_spectrum(received_signal)

        # 信噪比估算
        signal_power = np.mean(np.abs(mf_output) ** 2)
        noise_power = noise_std ** 2
        snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-15) + 1e-15)

        return {
            't': t,
            'envelope': envelope,
            'peak_time': peak_time,
            'freqs': freqs,
            'log_spectrum': log_spec,
            'snr_db': float(snr_db),
        }
