# -*- coding: utf-8 -*-
"""
seismic_analysis.py
基于 426_fft_serial 合成
恒星星震学振动模式分析：利用 FFT 分析脉动频谱。
"""

import numpy as np
from numerical_utils import cooley_tukey_fft, inverse_fft
from typing import Tuple, Optional


class SeismicAnalysis:
    """
    恒星星震学分析器。
    
    恒星脉动方程（Cowling 近似下径向模式）：
      d²ξ/dr² + (4/r - 1/H_P) dξ/dr + (σ²/c_s² - l(l+1)/r²) ξ = 0
    
    其中：
      ξ : 径向位移
      H_P = P / (ρ g) : 压强标高
      c_s = sqrt(Γ1 P / ρ) : 声速
      σ = 2π ν : 角频率
      l : 球谐度
    
    本征频率近似：
      ν_nl ≈ Δν (n + l/2 + ε)  (大频率分离)
      Δν ≈ (2 ∫_0^R dr/c_s)^{-1}
    """

    def __init__(self, n_modes: int = 50):
        self.n_modes = n_modes

    @staticmethod
    def acoustic_cutoff_frequency(radius: np.ndarray, sound_speed: np.ndarray) -> float:
        """
        声学截断频率（Lamb 频率 l=1）：
          ω_ac = c_s / (2 H_P) ≈ c_s √(g ρ / (2 Γ1 P))
        简化：ω_ac ≈ c_s / R
        """
        c_s = np.asarray(sound_speed, dtype=np.float64)
        R = np.max(radius) if len(radius) > 0 else 7e10
        return float(np.mean(c_s)) / R if R > 0 else 1e-6

    @staticmethod
    def large_frequency_separation(radius: np.ndarray, sound_speed: np.ndarray) -> float:
        """
        大频率分离：
          Δν = [2 ∫_0^R dr/c_s]^{-1}
        单位：μHz
        """
        r = np.asarray(radius, dtype=np.float64)
        cs = np.asarray(sound_speed, dtype=np.float64)
        cs = np.maximum(cs, 1e3)
        integrand = 1.0 / cs
        # 数值积分
        integral = np.trapz(integrand, r)
        if integral <= 0:
            return 100.0
        dnu = 1.0e6 / (2.0 * integral)  # 转换为 μHz
        return float(dnu)

    @staticmethod
    def small_frequency_separation(radius: np.ndarray, density: np.ndarray,
                                   sound_speed: np.ndarray) -> float:
        """
        小频率分离（对 l=0 和 l=2）：
          δν_{02} ≈ Δν / (6 + 12 dlnΔν/dlnν)
        简化：δν_{02} ≈ -Δν² / (2ν) * ∫_0^R (1/c_s)(dlnρ/dr - dlnρ_c/dr) dr
        """
        dnu = SeismicAnalysis.large_frequency_separation(radius, sound_speed)
        # 简化近似
        return dnu / 6.0

    def compute_p_mode_frequencies(self, n_max: int, l_max: int,
                                   dnu: float, epsilon: float = 1.5) -> np.ndarray:
        """
        计算 p-模式本征频率（渐近公式）：
          ν_{n,l} ≈ Δν (n + l/2 + ε)
        
        参数：
          n : 径向阶数 (0,1,...,n_max)
          l : 球谐度 (0,1,...,l_max)
          Δν : 大频率分离 [μHz]
          ε : 相移常数
        """
        freqs = []
        for l in range(l_max + 1):
            for n in range(n_max + 1):
                nu = dnu * (n + l / 2.0 + epsilon)
                freqs.append((n, l, nu))
        return np.array(freqs, dtype=[('n', int), ('l', int), ('nu', float)])

    def compute_g_mode_frequencies(self, n_g: int, l: int,
                                   N_brunt: float, R: float) -> np.ndarray:
        """
        g-模式频率近似（均匀恒星模型）：
          ν_{n,g} ≈ (n π / (l+1)) * N_brunt / (2π)
        其中 N_brunt 是 Brunt-Väisälä 频率 [Hz]。
        """
        freqs = []
        for n in range(1, n_g + 1):
            nu = (n * np.pi / (l + 1)) * N_brunt / (2.0 * np.pi) * 1e6  # μHz
            freqs.append((n, l, nu))
        return np.array(freqs, dtype=[('n', int), ('l', int), ('nu', float)])

    def frequency_spectrum_fft(self, time_series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        对光变曲线时间序列做 FFT，提取功率谱。
        
        输入：
          time_series : 等间隔时间序列（如光度变化）
          dt : 采样间隔 [s]
        输出：
          freqs : 频率数组 [μHz]
          power : 功率谱密度
        """
        ts = np.asarray(time_series, dtype=np.float64)
        n = len(ts)
        if n < 4:
            return np.array([]), np.array([])

        # 减去均值
        ts = ts - np.mean(ts)
        # 汉宁窗
        window = np.hanning(n)
        ts_win = ts * window

        # FFT (使用 Cooley-Tukey)
        spectrum = cooley_tukey_fft(ts_win)
        power = np.abs(spectrum) ** 2

        # 频率轴
        freqs = np.fft.fftfreq(n, d=dt) * 1e6  # 转换为 μHz
        # 只取正频率
        pos_mask = freqs >= 0
        return freqs[pos_mask], power[pos_mask]

    def echelle_diagram_data(self, freqs: np.ndarray, dnu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Échelle 图数据：将频率对 Δν 取模。
        x = ν mod Δν
        y = ν
        用于识别模式序列。
        """
        nu = np.asarray(freqs, dtype=np.float64)
        x = np.mod(nu, dnu)
        return x, nu

    def mode_inertia(self, radius: np.ndarray, density: np.ndarray,
                     eigenfunction: np.ndarray) -> float:
        """
        模式惯性（归一化）：
          E = ∫_0^M |ξ|² dm / M
        """
        r = np.asarray(radius, dtype=np.float64)
        rho = np.asarray(density, dtype=np.float64)
        xi = np.asarray(eigenfunction, dtype=np.float64)
        # 近似 dm = 4π r² ρ dr
        dr = np.diff(r)
        dr = np.append(dr, dr[-1])
        dm = 4.0 * np.pi * r ** 2 * rho * dr
        E = np.sum(xi ** 2 * dm)
        M = np.sum(dm)
        return float(E / M) if M > 0 else 0.0
