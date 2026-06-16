# -*- coding: utf-8 -*-
"""
detector_response.py
====================

探测器响应函数与效率建模

对应种子项目 1260_Energy-Sufficient-Safe-Centralized-Control:
  - PDF (概率密度函数) 建模
  - 基于网格的响应函数

科学公式
--------
观测反冲谱:
  dN/dE_obs = ε(E_obs) * ∫ R(E_true) * G(E_obs, E_true; σ_E) dE_true

其中:
  ε(E)     探测效率
  G        探测器能量分辨率函数 (高斯核)
  σ_E(E)   = √(E * E_noise² + σ_Fano²) 能量分辨率

能量分辨率:
  σ_E/E = 2.355 * √(F ε_pair / E)  (半导体探测器)
  F       = Fano 因子
  ε_pair  = 电子-空穴对产生能

效率函数:
  ε(E) = ε_0 * (1 - exp(-(E/E_th)^k))  *  exp(-E/E_sat)

探测阈值:
  E_th ~ 0.1-1 keV (Xe), ~ 0.05 keV (Ge)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 第一部分: 能量分辨率
# ---------------------------------------------------------------------------

class EnergyResolution:
    """
    探测器能量分辨率模型。

    半导体探测器:
      σ_E(E) = √(2.355² * F * ε_pair * E + σ_noise²)

    闪烁体探测器:
      σ_E/E = a / √E ⊕ b  (统计项 ⊕ 常数项)

    Fano 因子:
      Xe: F ≈ 0.13, ε_pair ≈ 13.7 eV
      Ge: F ≈ 0.11, ε_pair ≈ 2.96 eV
      Ar: F ≈ 0.18, ε_pair ≈ 23.6 eV
    """

    # 探测器参数数据库
    DETECTORS = {
        'Xe_lz': {
            'F': 0.13, 'eps_pair_eV': 13.7, 'sigma_noise_eV': 100.0,
            'type': 'semiconductor', 'name': 'XENON-nT LZ (LXe)'
        },
        'Ge_pp': {
            'F': 0.11, 'eps_pair_eV': 2.96, 'sigma_noise_eV': 50.0,
            'type': 'semiconductor', 'name': 'SuperCDMS Ge PP'
        },
        'Ar_darkside': {
            'F': 0.18, 'eps_pair_eV': 23.6, 'sigma_noise_eV': 200.0,
            'type': 'scintillator', 'name': 'DarkSide-20k (LAr)'
        },
    }

    def __init__(self, detector: str = 'Xe_lz'):
        if detector not in self.DETECTORS:
            raise ValueError(
                f"未知探测器: {detector}. "
                f"可选: {list(self.DETECTORS.keys())}"
            )
        self.params = self.DETECTORS[detector].copy()
        self.detector_name = detector

    def sigma_E(self, E_keV) -> np.ndarray:
        """
        计算能量分辨率 σ_E (keV)。

        σ_E = √(2.355² * F * ε_pair * E_keV * 1000 + σ_noise²) / 1000
        支持数组输入。
        """
        E_arr = np.atleast_1d(np.asarray(E_keV, dtype=np.float64))
        E_arr = np.where(E_arr < 0, 0.0, E_arr)
        F = self.params['F']
        eps = self.params['eps_pair_eV']
        sigma_noise = self.params['sigma_noise_eV'] / 1000.0  # keV
        variance = (2.355**2) * F * eps * E_arr * 1000.0 + sigma_noise**2
        sigma = np.sqrt(np.maximum(variance, 0.0))
        if np.isscalar(E_keV):
            return float(sigma[0])
        return sigma

    def gaussian_response(
        self,
        E_obs,
        E_true,
    ) -> np.ndarray:
        """
        高斯响应函数:
          G(E_obs; E_true) = (1/√(2π σ_E²))
                              * exp(-(E_obs - E_true)² / (2 σ_E²))
        E_obs: array of observation energies
        E_true: array or scalar of true energies
        返回与 E_obs 同形状的数组 (当 E_true 为 scalar),
        或与 E_true 同形状 (当 E_obs 为 scalar)
        """
        E_obs = np.atleast_1d(np.asarray(E_obs, dtype=np.float64))
        E_true = np.atleast_1d(np.asarray(E_true, dtype=np.float64))
        # 广播: E_true 为标量情况
        if E_true.size == 1:
            sigma = self.sigma_E(float(E_true[0]))
            if sigma <= 0:
                return np.where(np.abs(E_obs - E_true[0]) < 1e-10, 1.0, 0.0)
            return (
                np.exp(-0.5 * ((E_obs - E_true[0]) / sigma)**2)
                / (sigma * np.sqrt(2.0 * np.pi))
            )
        # 两者都是数组: 逐点计算
        result = np.zeros_like(E_obs)
        for i in range(len(E_obs)):
            E_t = E_true[i] if i < len(E_true) else E_true[-1]
            sigma = self.sigma_E(float(E_t))
            if sigma > 0:
                result[i] = (
                    np.exp(-0.5 * ((E_obs[i] - E_t) / sigma)**2)
                    / (sigma * np.sqrt(2.0 * np.pi))
                )
        return result


# ---------------------------------------------------------------------------
# 第二部分: 探测效率 (源自 1260 PDF 建模)
# ---------------------------------------------------------------------------

class DetectionEfficiency:
    """
    探测效率函数。

    ε(E) = ε_0 * (1 - exp(-(E/E_th)^k)) * exp(-E/E_sat)

    参数:
      ε_0    最大效率 (平台值)
      E_th   阈值能量 (效率达 63% 的能量)
      k      阈值陡度
      E_sat  饱和能量 (高能衰减)
    """

    def __init__(
        self,
        eps_0: float = 1.0,
        E_th: float = 1.0,    # keV
        k: float = 2.0,
        E_sat: float = 1000.0,  # keV
    ):
        if eps_0 < 0 or eps_0 > 1:
            raise ValueError(f"ε_0 必须在 [0, 1], 得到 {eps_0}")
        if E_th <= 0:
            raise ValueError(f"E_th 必须 > 0, 得到 {E_th}")
        self.eps_0 = eps_0
        self.E_th = E_th
        self.k = k
        self.E_sat = E_sat

    def efficiency(self, E_keV: np.ndarray) -> np.ndarray:
        """计算效率 ε(E)。"""
        E = np.maximum(E_keV, 0.0)
        rise = 1.0 - np.exp(-(E / self.E_th)**self.k)
        fall = np.exp(-E / self.E_sat)
        return self.eps_0 * rise * fall

    def efficiency_derivative(self, E_keV: np.ndarray) -> np.ndarray:
        """效率对能量的导数 dε/dE (用于误差传播)。"""
        E = np.maximum(E_keV, 1e-10)
        x = E / self.E_th
        rise = 1.0 - np.exp(-x**self.k)
        drise = self.k * x**(self.k - 1) / self.E_th * np.exp(-x**self.k)
        fall = np.exp(-E / self.E_sat)
        dfall = -fall / self.E_sat
        return self.eps_0 * (drise * fall + rise * dfall)


# ---------------------------------------------------------------------------
# 第三部分: 完整探测器响应
# ---------------------------------------------------------------------------

class DetectorResponse:
    """
    完整探测器响应: 效率 × 能量分辨率卷积。

    dN/dE_obs = ∫ ε(E_true) * (dR/dE_true) * G(E_obs; E_true) dE_true

    使用 Gauss-Legendre 求积实现卷积。
    """

    def __init__(
        self,
        detector: str = 'Xe_lz',
        eps_0: float = 1.0,
        E_th: float = 1.0,
        exposure_kg_day: float = 1000.0,
    ):
        self.resolution = EnergyResolution(detector)
        self.efficiency = DetectionEfficiency(eps_0=eps_0, E_th=E_th)
        self.exposure = exposure_kg_day

    def observed_rate(
        self,
        E_obs_keV: np.ndarray,
        true_rate_func,
        E_true_grid: np.ndarray,
    ) -> np.ndarray:
        """
        计算观测到的反冲率 (真谱与响应函数的卷积)。

        使用复合梯形法则 (避免求积节点依赖):
          rate_obs(E_obs) = Σ_j w_j * ε(E_j) * rate_true(E_j)
                                * G(E_obs; E_j)
        """
        dE = E_true_grid[1] - E_true_grid[0] if len(E_true_grid) > 1 else 1.0
        rate_true = np.array([true_rate_func(E) for E in E_true_grid])

        rate_obs = np.zeros(len(E_obs_keV))
        for i, E_obs in enumerate(E_obs_keV):
            G_vals = self.resolution.gaussian_response(E_obs, E_true_grid)
            eps_vals = self.efficiency.efficiency(E_true_grid)
            integrand = eps_vals * rate_true * G_vals
            rate_obs[i] = np.trapz(integrand, E_true_grid)

        return rate_obs * self.exposure

    def expected_events(
        self,
        E_obs_keV: np.ndarray,
        true_rate_func,
        E_true_grid: np.ndarray,
    ) -> np.ndarray:
        """期望事件数 = 观测率 × 曝光量。"""
        return self.observed_rate(E_obs_keV, true_rate_func, E_true_grid)

    def significance(
        self,
        signal_rate: np.ndarray,
        background_rate: np.ndarray,
    ) -> float:
        """
        信号显著性 (简单 Poisson):
          S = N_signal / √(N_signal + N_background)
        """
        N_s = np.sum(signal_rate)
        N_b = np.sum(background_rate)
        denom = np.sqrt(N_s + N_b)
        if denom <= 0:
            return 0.0
        return N_s / denom


# ---------------------------------------------------------------------------
# 第四部分: 基于网格的 PDF 响应 (源自 1260 Controller_sim)
# ---------------------------------------------------------------------------

def build_response_matrix(
    E_grid: np.ndarray,
    detector: DetectorResponse,
) -> np.ndarray:
    """
    构建响应矩阵 R[i, j]: 真能在 E_j 的事件被观测到 E_i 的概率。

    R[i, j] = ε(E_j) * G(E_i; E_j) * ΔE_j

    用于反卷积 (unfolding) 问题
    """
    n_obs = len(E_grid)
    n_true = len(E_grid)
    R = np.zeros((n_obs, n_true))
    dE = E_grid[1] - E_grid[0] if n_true > 1 else 1.0
    for j in range(n_true):
        G_col = detector.resolution.gaussian_response(E_grid, E_grid[j])
        eps_j = detector.efficiency.efficiency(np.array([E_grid[j]]))[0]
        R[:, j] = eps_j * G_col * dE
    return R
