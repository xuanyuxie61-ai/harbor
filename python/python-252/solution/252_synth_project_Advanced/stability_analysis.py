#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_analysis.py  ——  MHD 线性稳定性分析 & 色散关系求解

融合种子项目:
  - 1374_unstable_ode  : 非稳定 ODE 的精确解 & 增长率提取
  - 095_bisection_integer : 整数二分 → 离散波数搜索临界值
  - 1008_C15StabilityDataW : Arrhenius 拟合 → 增长率指数拟合

核心问题:
  线性化 MHD 方程  d U' / dt = L U'
  其中 L 是线性算子 (含背景流/磁场/度规).
  本征值问题:  L U' = omega U'
  omega = omega_r + i omega_i
  omega_i > 0: 不稳定模 (增长率), omega_i < 0: 衰减模.

色散关系 (理想 MHD, 均匀背景):
  (omega - k.v)^2 = k^2 vA^2    (Alfvén 波)
  (omega - k.v)^4 - (cs^2 + vA^2) k^2 (omega - k.v)^2 + cs^2 vA^2 k_z^2 k^2 = 0
  (快/慢磁声波)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List
from mhd_constants import MHDConfig, PlasmaParameters


# ============================================================
# 色散关系求解器 (均匀背景, 解析)
# ============================================================
class DispersionSolver:
    """
    求解理想 MHD 色散关系:
      rho (omega - k.v)^2 = (k.B)^2 / mu0    (Alfvén)
      ... 快/慢磁声 ...
    """

    def __init__(self, plasma: PlasmaParameters) -> None:
        self.plasma = plasma

    def alfven_frequency(self, k: np.ndarray, B: np.ndarray,
                          rho: float, v: np.ndarray) -> np.ndarray:
        """
        Alfvén 波频率:  omega_A = |k . B| / sqrt(rho)
        (几何单位制 mu0=1).
        """
        kdotB = np.sum(k * B, axis=-1) if k.ndim > 1 else np.dot(k, B)
        return np.abs(kdotB) / math.sqrt(max(rho, 1e-30))

    def magnetosonic_frequencies(self, k_mag: float, theta_kB: float,
                                   cs: float, vA: float
                                   ) -> Tuple[float, float]:
        """
        快/慢磁声波频率:
        omega_{f,s}^2 = 0.5 k^2 [ cs^2 + vA^2 ± sqrt((cs^2+vA^2)^2 - 4 cs^2 vA^2 cos^2 theta) ]
        """
        a = cs * cs + vA * vA
        disc = a * a - 4.0 * cs * cs * vA * vA * math.cos(theta_kB) ** 2
        disc = max(disc, 0.0)
        sqrt_disc = math.sqrt(disc)
        omega_f2 = 0.5 * k_mag * k_mag * (a + sqrt_disc)
        omega_s2 = 0.5 * k_mag * k_mag * (a - sqrt_disc)
        return math.sqrt(max(omega_f2, 0.0)), math.sqrt(max(omega_s2, 0.0))


# ============================================================
# 增长率提取 (指数拟合, 融合 1008 Arrhenius 思想)
# ============================================================
def extract_growth_rate(time: np.ndarray, amplitude: np.ndarray
                         ) -> Tuple[float, float, float]:
    """
    从时间序列 amplitude(t) 提取指数增长率 gamma.
    amplitude ~ A exp(gamma t).
    用最小二乘拟合 log(amplitude) = log A + gamma t.
    返回 (gamma, A, R^2).
    """
    # 过滤掉非正值
    mask = amplitude > 1e-30
    if np.sum(mask) < 3:
        return 0.0, 0.0, 0.0
    t = time[mask]
    y = np.log(amplitude[mask])
    # 线性拟合
    n = len(t)
    Sx = np.sum(t)
    Sy = np.sum(y)
    Sxx = np.sum(t * t)
    Sxy = np.sum(t * y)
    denom = n * Sxx - Sx * Sx
    if abs(denom) < 1e-30:
        return 0.0, 0.0, 0.0
    gamma = (n * Sxy - Sx * Sy) / denom
    logA = (Sy - gamma * Sx) / n
    A = math.exp(logA)
    # R^2
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    y_pred = logA + gamma * t
    ss_res = np.sum((y - y_pred) ** 2)
    R2 = 1.0 - ss_res / (ss_tot + 1e-30)
    return gamma, A, R2


# ============================================================
# 临界波数搜索 (融合 095_bisection_integer)
# ============================================================
def find_critical_wavenumber(omega_i_func, k_min: float, k_max: float,
                              tol: float = 1e-6) -> float:
    """
    二分搜索临界波数 k_c, 使得 omega_i(k_c) = 0.
    omega_i_func: 函数 k -> omega_i (增长率).
    假设 omega_i(k_min) > 0 (不稳定), omega_i(k_max) < 0 (稳定).
    参考 095_bisection_integer 的二分结构.
    """
    a, b = k_min, k_max
    fa = omega_i_func(a)
    fb = omega_i_func(b)
    if fa * fb > 0:
        raise ValueError(f"区间 [{a}, {b}] 不是变号区间: f(a)={fa}, f(b)={fb}")
    while abs(b - a) > tol:
        c = 0.5 * (a + b)
        fc = omega_i_func(c)
        if abs(fc) < 1e-12:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


# ============================================================
# MRI 增长率 (磁旋转不稳定性, 黑洞吸积盘核心)
# ============================================================
def mri_growth_rate(k_z: float, Omega: float, vA_z: float
                     ) -> float:
    """
    理想 MRI 增长率 (Balbus & Hawley 1991):
      gamma^2 = -k_z^2 vA_z^2 + Omega^2 (k_z vA_z / Omega)^2 / ...
    简化形式 (轴对称, incompressible):
      gamma = sqrt( (k.vA)^2 + Omega^2 - sqrt(...) )
    最大增长率 ~ 0.75 Omega (在 k vA ~ Omega).
    """
    kvA2 = k_z * k_z * vA_z * vA_z
    # 色散关系: omega^4 + (2 k^2 vA^2 - 4 Omega^2) omega^2 + k^2 vA^2 (k^2 vA^2 - 2 kappa^2) = 0
    # 其中 kappa^2 = 4 Omega^2 (开普勒)
    kappa2 = 4.0 * Omega * Omega
    a_coeff = 2.0 * kvA2 - 4.0 * Omega * Omega
    b_coeff = kvA2 * (kvA2 - 2.0 * kappa2)
    disc = a_coeff * a_coeff - 4.0 * b_coeff
    if disc < 0:
        return 0.0
    omega2_minus = 0.5 * (-a_coeff - math.sqrt(disc))
    if omega2_minus > 0:
        return math.sqrt(omega2_minus)
    return 0.0


# ============================================================
# 模态分析聚合
# ============================================================
class StabilityAnalyzer:
    """
    对 MHD 背景场做线性稳定性分析.
    返回最不稳定模态的增长率 & 频率.
    """

    def __init__(self, cfg: MHDConfig) -> None:
        self.cfg = cfg
        self.disp = DispersionSolver(cfg.plasma)

    def scan_wavenumbers(self, rho: float, p: float, Bz: float,
                          Omega: float, k_range: Tuple[float, float] = (0.01, 10.0)
                          ) -> List[Tuple[float, float, str]]:
        """
        扫描波数 k, 返回 [(k, gamma, mode_type), ...].
        mode_type: "MRI", "Alfvén", "fast/slow magnetosonic".
        """
        cs = self.cfg.plasma.sound_speed(p, rho)
        vA = abs(Bz) / math.sqrt(max(rho, 1e-30))
        results = []
        n_scan = self.cfg.num.omega_scan_pts
        k_arr = np.linspace(k_range[0], k_range[1], n_scan)
        for k in k_arr:
            # MRI
            gamma_mri = mri_growth_rate(k, Omega, vA)
            if gamma_mri > 0:
                results.append((k, gamma_mri, "MRI"))
            # Alfvén
            omega_A = self.disp.alfven_frequency(
                np.array([0, 0, k]), np.array([0, 0, Bz]), rho, np.array([0, 0, 0]))
            results.append((k, -1e-6, "Alfvén"))  # 理想 Alfvén 无增长
        # 排序 (按增长率降序)
        results.sort(key=lambda x: -x[1])
        return results[:self.cfg.num.n_modes_kept]
