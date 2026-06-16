"""
dos.py — 态密度 (DOS) 计算
=============================

本模块实现态密度的各种计算方法。
融合种子项目:
  - 683_line_monte_carlo: Monte Carlo 积分 (随机/遍历采样)
  - 681_line_integrals: 精确单式积分 (参考值)
  - 559_hypercube_integrals: 高维 BZ 积分
  - 302_disk01_rule: Gauss 求积节点

核心物理:
  态密度 (DOS):
    g(ε) = (1/V_BZ) ∫_{BZ} δ(ε - ε_n(k)) dk

  积分态密度 (IDOS):
    N(ε) = ∫_{-∞}^{ε} g(ε') dε'

  在 1D 中, van Hove 奇点出现在 dε/dk = 0 处:
    g(ε) ~ 1/√|ε - ε_vH|  (可积奇点)

  计算方法:
  1. Gaussian 展宽: δ(ε) → (1/σ√π) exp(-ε²/σ²)
  2. 四面体方法 (1D 简化)
  3. Monte Carlo (源自 683_line_monte_carlo)
  4. 解析 (对简单能带模型)
"""

import numpy as np
from typing import Tuple, Optional
from physical_constants import PI, TWO_PI


# ============================================================
# Gaussian 展宽法
# ============================================================

def dos_gaussian_broadening(eigenvalues: np.ndarray,
                              weights: np.ndarray,
                              energy_grid: np.ndarray,
                              sigma: float = 0.05
                              ) -> np.ndarray:
    """
    使用 Gaussian 展宽计算态密度。

    δ(ε - ε_{nk}) → (1/σ√(2π)) exp(-(ε-ε_{nk})²/(2σ²))

    g(ε) = Σ_{n,k} w_k · (1/σ√(2π)) exp(-(ε-ε_{nk})²/(2σ²))

    展宽参数 σ 的选择:
    - σ 太小: 噪声大, 不光滑
    - σ 太大: 物理特征被抹平
    - 建议: σ ~ 几倍 k 点间距对应的能量尺度

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_kpoints, n_bands)
        本征值
    weights : np.ndarray, shape (n_kpoints,)
        k 点权重
    energy_grid : np.ndarray
        能量网格
    sigma : float
        展宽参数

    Returns
    -------
    dos : np.ndarray
        态密度 g(ε)
    """
    n_kpoints, n_bands = eigenvalues.shape
    dos = np.zeros_like(energy_grid)
    norm = 1.0 / (sigma * np.sqrt(2.0 * PI))

    for ik in range(n_kpoints):
        for ib in range(n_bands):
            gauss = norm * np.exp(
                -(energy_grid - eigenvalues[ik, ib]) ** 2 / (2.0 * sigma ** 2))
            dos += weights[ik] * gauss

    return dos


# ============================================================
# 积分态密度 (IDOS)
# ============================================================

def integrated_dos(dos: np.ndarray, energy_grid: np.ndarray
                    ) -> np.ndarray:
    """
    计算积分态密度 N(ε) = ∫_{-∞}^{ε} g(ε') dε'。

    使用梯形法则:
      N(ε_i) = Σ_{j≤i} (g(ε_j) + g(ε_{j-1})) / 2 · Δε

    N(ε) 的物理意义:
    - N(ε_F) = 电子数 / 2 (考虑自旋)
    - dN/dε = g(ε)

    Parameters
    ----------
    dos : np.ndarray
        态密度
    energy_grid : np.ndarray
        能量网格

    Returns
    -------
    idos : np.ndarray
        积分态密度
    """
    de = np.diff(energy_grid)
    idos = np.zeros_like(dos)
    for i in range(1, len(dos)):
        idos[i] = idos[i - 1] + 0.5 * (dos[i] + dos[i - 1]) * de[i - 1]
    return idos


# ============================================================
# 解析态密度 (1D 自由电子)
# ============================================================

def dos_free_electron_1d(energy: np.ndarray,
                           mass: float = 1.0,
                           a: float = 1.0) -> np.ndarray:
    """
    1D 自由电子气的解析态密度。

    色散关系: ε = ℏ²k²/(2m)
    在 BZ [-π/a, π/a] 中:

    g(ε) = (2/π) · (1/√(2ε/m))  for ε > 0
          = 0                      for ε < 0

    因子 2 来自自旋简并。
    在 1D 中, g(ε) ~ ε^{-1/2}, 在 ε=0 处有 van Hove 奇点。

    Parameters
    ----------
    energy : np.ndarray
        能量网格
    mass : float
        电子有效质量
    a : float
        晶格常数

    Returns
    -------
    dos : np.ndarray
        解析态密度
    """
    dos = np.zeros_like(energy)
    positive = energy > 1e-15
    dos[positive] = (2.0 / PI) * np.sqrt(mass / (2.0 * energy[positive])) / a
    return dos


# ============================================================
# Monte Carlo 态密度 (源自 683_line_monte_carlo)
# ============================================================

def dos_monte_carlo(eigenvalues_func: callable,
                     n_samples: int,
                     energy_grid: np.ndarray,
                     a: float,
                     sigma: float = 0.05,
                     method: str = 'random',
                     seed: int = 42
                     ) -> np.ndarray:
    """
    Monte Carlo 方法计算态密度 (源自 683_line_monte_carlo)。

    g(ε) = (1/N_samples) Σ_i δ(ε - ε(k_i))

    其中 k_i 为 BZ 中的采样点:
    - 随机采样: k_i = rand() · 2π/a - π/a
    - 遍历采样: k_i = (i · φ) mod (2π/a) - π/a, φ = 黄金比例

    遍历采样 (低差异序列) 的优势:
    - 误差 ~ O(1/N) vs 随机采样的 O(1/√N)
    - 更均匀覆盖 BZ

    Parameters
    ----------
    eigenvalues_func : callable
        k → eigenvalues(k) 的函数
    n_samples : int
        采样点数
    energy_grid : np.ndarray
        能量网格
    a : float
        晶格常数
    sigma : float
        Gaussian 展宽
    method : str
        'random' 或 'ergodic'
    seed : int
        随机种子

    Returns
    -------
    dos : np.ndarray
        Monte Carlo 态密度
    """
    np.random.seed(seed)
    bz_min = -PI / a
    bz_max = PI / a
    bz_length = bz_max - bz_min

    # 生成 k 点
    if method == 'random':
        kpoints = bz_min + np.random.rand(n_samples) * bz_length
    elif method == 'ergodic':
        # 黄金比例遍历序列
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        kpoints = np.zeros(n_samples)
        k_val = 0.0
        for i in range(n_samples):
            k_val = (k_val + phi) % 1.0
            kpoints[i] = bz_min + k_val * bz_length
    else:
        raise ValueError(f"未知采样方法: {method}")

    # 计算本征值
    norm = 1.0 / (sigma * np.sqrt(2.0 * PI))
    dos = np.zeros_like(energy_grid)

    for k in kpoints:
        evals = eigenvalues_func(k)
        for e in evals:
            dos += norm * np.exp(-(energy_grid - e) ** 2 / (2.0 * sigma ** 2))

    dos /= n_samples
    return dos


# ============================================================
# 精确 BZ 积分参考 (源自 681_line_integrals)
# ============================================================

def bz_integral_reference(exponent: int, a: float) -> float:
    """
    BZ 上 k^e 的精确积分 (源自 681_line_integrals)。

    ∫_{-π/a}^{π/a} k^e dk

    用于验证 Monte Carlo 积分的精度。
    """
    from lattice import line_bz_monomial_integral
    return line_bz_monomial_integral(exponent, a)


# ============================================================
# 部分态密度 (PDOS)
# ============================================================

def partial_dos(eigenvectors: np.ndarray,
                 eigenvalues: np.ndarray,
                 weights: np.ndarray,
                 region_mask: np.ndarray,
                 energy_grid: np.ndarray,
                 sigma: float = 0.05) -> np.ndarray:
    """
    计算部分态密度 (PDOS)。

    PDOS 将态密度按空间区域分解:
      g_A(ε) = Σ_{n,k} w_k |⟨A|ψ_{nk}⟩|² δ(ε - ε_{nk})

    其中 |A⟩ 为区域 A 的投影算子。
    在实空间 FD 表示中:
      |⟨A|ψ_{nk}⟩|² = Σ_{i∈A} |ψ_{nk}(x_i)|² · dx

    Parameters
    ----------
    eigenvectors : np.ndarray, shape (n_k, n_grid, n_bands)
    eigenvalues : np.ndarray, shape (n_k, n_bands)
    weights : np.ndarray, shape (n_k,)
    region_mask : np.ndarray, shape (n_grid,)
        区域掩码 (0 或 1)
    energy_grid : np.ndarray
        能量网格
    sigma : float
        展宽

    Returns
    -------
    pdos : np.ndarray
    """
    n_kpoints, n_grid, n_bands = eigenvectors.shape
    pdos = np.zeros_like(energy_grid)
    norm = 1.0 / (sigma * np.sqrt(2.0 * PI))

    for ik in range(n_kpoints):
        for ib in range(n_bands):
            # 区域投影权重
            proj = np.sum(np.abs(eigenvectors[ik, :, ib]) ** 2 *
                          region_mask)
            # Gaussian 展宽
            gauss = norm * np.exp(
                -(energy_grid - eigenvalues[ik, ib]) ** 2 /
                (2.0 * sigma ** 2))
            pdos += weights[ik] * proj * gauss

    return pdos
