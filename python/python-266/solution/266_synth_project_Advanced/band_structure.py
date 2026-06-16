"""
band_structure.py — 能带结构计算与分析
========================================

本模块实现能带结构的计算、分析和后处理。
融合种子项目:
  - 596_interp_trig: 三角插值, 用于能带的平滑插值
  - 225_cpr: Chebyshev 代理方法, 用于能带交叉点定位
  - 350_fd_predator_prey: 有限差分 ODE, 用于能带曲率分析

核心物理:
  能带结构 ε_n(k):
    在 Brillouin 区 [-π/a, π/a] 上求解 KS 方程得到的本征值

  有效质量:
    1/m* = (1/ℏ²) · d²ε/dk²

  能带速度:
    v_n(k) = (1/ℏ) · dε_n/dk

  能隙:
    在 BZ 边界 k=π/a 处, ε_{n+1}(π/a) - ε_n(π/a) = Δ_n

  群速度:
    ℏ·v_g = ∇_k ε_n(k)

  对于 1D: v_g = (1/ℏ) dε/dk
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from physical_constants import PI, TWO_PI
from lattice import monkhorst_pack_grid


# ============================================================
# 能带结构计算
# ============================================================

class BandStructure:
    """
    能带结构数据容器与分析工具。

    存储:
      ε_{n,k}: shape (n_kpoints, n_bands)
      k: shape (n_kpoints,)
      ψ_{n,k}(x): shape (n_kpoints, n_grid, n_bands)

    提供:
      - 能带插值 (三角插值)
      - 有效质量计算
      - 能隙提取
      - van Hove 奇点检测
    """

    def __init__(self, kpoints: np.ndarray,
                 eigenvalues: np.ndarray,
                 eigenvectors: Optional[np.ndarray] = None,
                 lattice_constant: float = 1.0):
        self.kpoints = kpoints
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.a = lattice_constant
        self.n_kpoints = len(kpoints)
        self.n_bands = eigenvalues.shape[1] if eigenvalues.ndim > 1 else 1

    def band_energy(self, band_index: int,
                     k_index: int) -> float:
        """返回第 band_index 能带在 k_index 处的能量"""
        return self.eigenvalues[k_index, band_index]

    def band_gap(self, band_index: int,
                  location: str = 'X') -> float:
        """
        计算能隙 (band gap)。

        在 BZ 边界 (X 点, k=π/a) 处:
          Δ_n = ε_{n+1}(π/a) - ε_n(π/a)

        在 Γ 点 (k=0):
          Δ_n = ε_{n+1}(0) - ε_n(0)

        Parameters
        ----------
        band_index : int
            能带指标 (计算 band_index 与 band_index+1 之间的能隙)
        location : str
            'X' 或 'Gamma'

        Returns
        -------
        gap : float
            能隙大小 [Ha]
        """
        if band_index >= self.n_bands - 1:
            return 0.0

        if location == 'X':
            # BZ 边界
            ik = np.argmin(np.abs(self.kpoints - PI / self.a))
        elif location == 'Gamma':
            ik = np.argmin(np.abs(self.kpoints))
        else:
            raise ValueError(f"未知高对称点: {location}")

        gap = self.eigenvalues[ik, band_index + 1] - \
              self.eigenvalues[ik, band_index]
        return gap

    def effective_mass(self, band_index: int,
                        k_center: float = 0.0) -> float:
        """
        计算能带有效质量。

        1/m* = d²ε/dk²  (原子单位 ℏ=1)

        使用中心差分:
          d²ε/dk² ≈ [ε(k+dk) - 2ε(k) + ε(k-dk)] / dk²

        其中 dk 为 k 点间距。

        Parameters
        ----------
        band_index : int
            能带指标
        k_center : float
            计算有效质量的 k 点

        Returns
        -------
        m_star : float
            有效质量 (原子单位 m_e = 1)
        """
        ik = np.argmin(np.abs(self.kpoints - k_center))

        if ik == 0 or ik == self.n_kpoints - 1:
            # 端点, 使用前向/后向差分
            if ik == 0:
                dk = self.kpoints[1] - self.kpoints[0]
                d2e = (self.eigenvalues[2, band_index] -
                       2.0 * self.eigenvalues[1, band_index] +
                       self.eigenvalues[0, band_index]) / (dk ** 2)
            else:
                dk = self.kpoints[-1] - self.kpoints[-2]
                d2e = (self.eigenvalues[-1, band_index] -
                       2.0 * self.eigenvalues[-2, band_index] +
                       self.eigenvalues[-3, band_index]) / (dk ** 2)
        else:
            dk = self.kpoints[ik + 1] - self.kpoints[ik - 1]
            dk /= 2.0
            d2e = (self.eigenvalues[ik + 1, band_index] -
                   2.0 * self.eigenvalues[ik, band_index] +
                   self.eigenvalues[ik - 1, band_index]) / (dk ** 2)

        if abs(d2e) < 1e-15:
            return float('inf')

        m_star = 1.0 / d2e
        return m_star

    def group_velocity(self, band_index: int) -> np.ndarray:
        """
        计算群速度 v_g(k) = dε/dk (原子单位)。

        使用中心差分:
          v_g(k_i) = [ε(k_{i+1}) - ε(k_{i-1})] / (k_{i+1} - k_{i-1})

        Parameters
        ----------
        band_index : int
            能带指标

        Returns
        -------
        velocities : np.ndarray, shape (n_kpoints,)
            群速度 [Bohr·Ha/ℏ]
        """
        e_band = self.eigenvalues[:, band_index]
        velocities = np.gradient(e_band, self.kpoints)
        return velocities

    def bandwidth(self, band_index: int) -> float:
        """
        能带宽度:
          W_n = max_k ε_n(k) - min_k ε_n(k)
        """
        e_band = self.eigenvalues[:, band_index]
        return np.max(e_band) - np.min(e_band)


# ============================================================
# 能带三角插值 (源自 596_interp_trig)
# ============================================================

def interpolate_band_trig(kpoints: np.ndarray,
                           energies: np.ndarray,
                           k_fine: np.ndarray,
                           lattice_constant: float) -> np.ndarray:
    """
    使用三角插值加密能带 (源自 596_interp_trig)。

    能带 ε(k) 是 k 的周期函数 (在倒格子空间中):
      ε(k + G) = ε(k)

    因此可以使用三角插值:
      ε_interp(k) = Σ_j ε(k_j) · L_j(k)

    其中 L_j 为三角基数函数 (与 596_interp_trig 相同)。

    优势: 对于光滑能带, 指数收敛; 无 Runge 现象。

    Parameters
    ----------
    kpoints : np.ndarray
        原始 k 点
    energies : np.ndarray
        原始能带能量
    k_fine : np.ndarray
        加密 k 点
    lattice_constant : float
        晶格常数

    Returns
    -------
    energies_fine : np.ndarray
        加密后的能带能量
    """
    N = len(kpoints)
    h = kpoints[1] - kpoints[0] if N > 1 else 1.0

    energies_fine = np.zeros_like(k_fine)

    for j in range(N):
        dx = k_fine - kpoints[j]
        arg = PI * dx / h

        if N % 2 == 1:
            denom = N * np.sin(arg / N)
            safe = np.abs(denom) > 1e-14
            L = np.zeros_like(k_fine)
            L[safe] = np.sin(arg[safe]) / denom[safe]
            L[~safe] = 1.0
        else:
            denom = N * np.tan(arg / N)
            safe = np.abs(denom) > 1e-14
            L = np.zeros_like(k_fine)
            L[safe] = np.sin(arg[safe]) / denom[safe]
            L[~safe] = 1.0

        L[np.abs(dx) < 1e-14] = 1.0
        energies_fine += energies[j] * L

    return energies_fine


# ============================================================
# 能带分析汇总
# ============================================================

def analyze_band_structure(band_struct: BandStructure
                            ) -> Dict[str, any]:
    """
    全面分析能带结构, 返回物理量汇总。

    包括:
    - 各能带的能隙 (Γ 和 X 点)
    - 各能带的宽度
    - 各能带在 Γ 点的有效质量
    - 直接/间接带隙

    Parameters
    ----------
    band_struct : BandStructure
        能带结构对象

    Returns
    -------
    analysis : dict
        分析结果汇总
    """
    results = {
        'n_bands': band_struct.n_bands,
        'n_kpoints': band_struct.n_kpoints,
        'band_gaps_X': [],
        'band_gaps_Gamma': [],
        'bandwidths': [],
        'effective_masses_Gamma': [],
    }

    for ib in range(band_struct.n_bands):
        results['bandwidths'].append(band_struct.bandwidth(ib))

        if ib < band_struct.n_bands - 1:
            results['band_gaps_X'].append(
                band_struct.band_gap(ib, 'X'))
            results['band_gaps_Gamma'].append(
                band_struct.band_gap(ib, 'Gamma'))

        try:
            m_star = band_struct.effective_mass(ib, 0.0)
            results['effective_masses_Gamma'].append(m_star)
        except (ValueError, ZeroDivisionError):
            results['effective_masses_Gamma'].append(float('inf'))

    # 间接带隙 (价带顶和导带底不在同一 k 点)
    if band_struct.n_bands >= 2:
        # 假设偶数电子填充前 n/2 条能带
        n_occ = band_struct.n_bands // 2
        vbm_k = band_struct.kpoints[
            np.argmax(band_struct.eigenvalues[:, n_occ - 1])]
        cbm_k = band_struct.kpoints[
            np.argmin(band_struct.eigenvalues[:, n_occ])]
        results['is_direct_gap'] = abs(vbm_k - cbm_k) < 0.01
        results['vbm_k'] = vbm_k
        results['cbm_k'] = cbm_k

    return results
