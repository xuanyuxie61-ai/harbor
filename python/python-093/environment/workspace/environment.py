#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
environment.py
水声传播抛物方程模型 — 海洋环境参数与声速剖面

本模块定义抛物方程（PE）求解所需的海洋声学环境，包括：
- Munk 标准声速剖面（深海典型 SSP）
- 吸收系数模型（Thorp 公式 + 硼酸盐/硫酸镁弛豫）
- 海底地形（bathymetry）参数化模型
- 海水密度与压缩率剖面

核心物理公式：
1. Munk 声速剖面：
   c(z) = c₀ · [1 + ε · (η + e^{−η} − 1)]
   其中：
     z 为深度（m），向下为正；
     η = 2(z − z_a)/B  为无量纲深度偏移；
     z_a 为声道轴深度（m）；
     B 为声道尺度参数（m）；
     ε = Δc / c₀ 为无量纲声速扰动；
     c₀ 为声道轴处声速（m/s）。
   该剖面在 z = z_a 处取得最小值 c₀，向下和向上均递增，
   形成典型的 SOFAR 声道。

2. 海水体积吸收系数（Thorp 公式，低频扩展）：
   α(f) = 0.11·f² / (1 + f²) + 44·f² / (4100 + f²) + 2.75×10⁻⁴·f² + 0.0033
   其中 f 为频率（kHz），α 单位为 dB/km。
   转换为 Np/m：α_Np = α_dB / (8685.889638)

3. 复数波数：
   k(z) = ω/c(z) + i·α(z)
   其中 ω = 2πf 为角频率，α(z) 为吸收引起的衰减系数。

4. 参考波数 k₀ 与折射率 n(z)：
   k₀ = ω / c₀
   n(z) = c₀ / c(z)
   抛物方程中的折射项：n²(z) − 1。

5. 海水密度状态方程（线性近似）：
   ρ(z) ≈ ρ₀ · [1 + κ_T · (P(z) − P₀)]
   其中 κ_T 为等温压缩率，P(z) = P₀ + ρ₀·g·z 为静水压。

6. 海底地形参数化（高斯山丘 + 斜坡）：
   h_b(r) = H₀ + H₁·tanh(β·(r − r₀)/L) + H₂·exp[−((r−r_c)/σ_r)² − ((z−z_c)/σ_z)²]
   用于模拟海山、大陆坡等复杂地形。
"""

import numpy as np


class OceanEnvironment:
    """
    海洋环境参数容器，提供声速剖面、吸收系数、海底地形等查询。
    """

    def __init__(self,
                 c0=1500.0,          # 声道轴声速 (m/s)
                 z_axis=1000.0,      # 声道轴深度 (m)
                 B=1000.0,           # Munk 尺度 (m)
                 epsilon=0.0057,     # Munk 扰动参数
                 rho0=1024.0,        # 参考密度 (kg/m³)
                 kappa_T=4.6e-10,    # 等温压缩率 (Pa⁻¹)
                 g=9.80665,          # 重力加速度 (m/s²)
                 P0=1.01325e5,       # 参考压强 (Pa)
                 seabed_type="clay", # 海底类型
                 seabed_cp=1700.0,   # 海底纵波声速 (m/s)
                 seabed_cs=800.0,    # 海底横波声速 (m/s)
                 seabed_rho=1900.0,  # 海底密度 (kg/m³)
                 seabed_loss=0.5,    # 海底损耗参数 (dB/λ)
                 depth_max=4000.0,   # 最大水深 (m)
                 frequency=100.0):   # 声源频率 (Hz)
        self.c0 = float(c0)
        self.z_axis = float(z_axis)
        self.B = float(B)
        self.epsilon = float(epsilon)
        self.rho0 = float(rho0)
        self.kappa_T = float(kappa_T)
        self.g = float(g)
        self.P0 = float(P0)
        self.seabed_type = seabed_type
        self.seabed_cp = float(seabed_cp)
        self.seabed_cs = float(seabed_cs)
        self.seabed_rho = float(seabed_rho)
        self.seabed_loss = float(seabed_loss)
        self.depth_max = float(depth_max)
        self.frequency = float(frequency)
        self.omega = 2.0 * np.pi * self.frequency
        self.k0 = self.omega / self.c0

        # 海底地形参数（默认平坦 + 一个高斯山丘）
        self.bathymetry_params = {
            'H0': depth_max,
            'H1': 0.0,
            'beta': 1.0,
            'r0': 0.0,
            'L': 1.0,
            'H2': 500.0,
            'r_c': 20000.0,
            'sigma_r': 5000.0,
        }

    def sound_speed(self, z):
        """
        Munk 声速剖面 c(z)。
        参数 z 可以是标量或 numpy 数组（深度，m，向下为正）。
        """
        z = np.asarray(z, dtype=np.float64)
        eta = 2.0 * (z - self.z_axis) / self.B
        c = self.c0 * (1.0 + self.epsilon * (eta + np.exp(-eta) - 1.0))
        # 边界保护：浅水处不应低于物理极限
        c_min = 1400.0
        return np.maximum(c, c_min)

    def absorption_db_per_km(self, f_khz=None):
        """
        Thorp 吸收公式（dB/km）。
        若 f_khz 为 None，则使用当前环境频率。
        """
        if f_khz is None:
            f_khz = self.frequency / 1000.0
        f = float(f_khz)
        alpha = (0.11 * f * f / (1.0 + f * f)
                 + 44.0 * f * f / (4100.0 + f * f)
                 + 2.75e-4 * f * f
                 + 0.0033)
        return alpha

    def absorption_np_per_m(self, f_khz=None):
        """
        将 dB/km 转换为 Np/m：
        α(Np/m) = α(dB/km) / (20·log₁₀(e)·1000) = α(dB/km) / 8685.889638
        """
        alpha_db = self.absorption_db_per_km(f_khz)
        return alpha_db / 8685.889638

    def wavenumber(self, z):
        """
        复数波数 k(z) = ω/c(z) + i·α。
        其中 α 为体积吸收系数 (Np/m)。
        """
        z = np.asarray(z, dtype=np.float64)
        c = self.sound_speed(z)
        alpha = self.absorption_np_per_m()
        k = self.omega / c + 1j * alpha
        return k

    def refractive_index(self, z):
        """折射率 n(z) = c₀ / c(z)。"""
        z = np.asarray(z, dtype=np.float64)
        return self.c0 / self.sound_speed(z)

    def refractive_index_squared_deviation(self, z):
        """
        抛物方程中的折射偏差项：n²(z) − 1。
        用于宽角 PE 的折射算子。
        """
        n = self.refractive_index(z)
        return n * n - 1.0

    def density(self, z):
        """
        海水密度随深度的线性近似：
        ρ(z) ≈ ρ₀ · [1 + κ_T · ρ₀ · g · z]
        """
        z = np.asarray(z, dtype=np.float64)
        return self.rho0 * (1.0 + self.kappa_T * self.rho0 * self.g * z)

    def impedance(self, z):
        """声阻抗 Z = ρ·c"""
        return self.density(z) * self.sound_speed(z)

    def bathymetry(self, r):
        """
        海底地形深度 h_b(r)（m），表示从海面到海底的垂直距离。
        参数 r 为水平距离（m）。
        """
        r = np.asarray(r, dtype=np.float64)
        p = self.bathymetry_params
        h = p['H0'] + p['H1'] * np.tanh(p['beta'] * (r - p['r0']) / p['L'])
        h += p['H2'] * np.exp(-((r - p['r_c']) / p['sigma_r']) ** 2)
        # 保证地形在物理范围内
        h = np.clip(h, 100.0, self.depth_max * 1.5)
        return h

    def seabed_reflection_coefficient(self, theta):
        """
        海底反射系数（Rayleigh 反射近似，用于 PE 边界）。
        对于流体海底：
        R(θ) = [ρ_b·c_b·cos(θ) − ρ_w·c_w·cos(θ_b)] /
               [ρ_b·c_b·cos(θ) + ρ_w·c_w·cos(θ_b)]
        其中 cos(θ_b) = √(1 − (c_b/c_w)² sin²θ)。
        加入吸收损耗：R_eff = R · exp(−2·δ·sin(θ))，δ 为损耗参数。
        """
        theta = np.asarray(theta, dtype=np.float64)
        cw = self.c0
        cb = self.seabed_cp
        rw = self.rho0
        rb = self.seabed_rho
        sin_theta = np.sin(theta)
        # Snell 定律给出海底折射角
        sin_thetab = (cw / cb) * sin_theta
        # 全反射临界角处理
        cos_thetab = np.sqrt(np.maximum(0.0, 1.0 - sin_thetab ** 2))
        cos_theta = np.cos(theta)
        num = rb * cb * cos_theta - rw * cw * cos_thetab
        den = rb * cb * cos_theta + rw * cw * cos_thetab
        R = safe_divide(num, den, fill_value=-1.0)
        # 加入衰减
        loss_factor = np.exp(-2.0 * self.seabed_loss * sin_theta)
        return R * loss_factor


def safe_divide(a, b, fill_value=0.0):
    """安全除法。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(a, fill_value, dtype=np.float64)
    mask = np.abs(b) > np.finfo(np.float64).eps * 100
    result[mask] = a[mask] / b[mask]
    return result
