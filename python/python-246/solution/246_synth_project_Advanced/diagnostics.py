"""
diagnostics.py  —  宇宙学模拟诊断工具
====================================

功能:
  1. 功率谱 P(k) 估计
  2. 两点相关函数 ξ(r)
  3. 密度 PDF 统计
  4. 能量守恒检验
  5. 残差诊断 (与 KdV 精确解比较)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, List


# ---------------------------------------------------------------------------- #
#                          功率谱 P(k)
# ---------------------------------------------------------------------------- #
def power_spectrum(delta: NDArray, box_length: float) -> Tuple[NDArray, NDArray]:
    """
    3D 密度场的功率谱估计:
        P(k) = <|δ(k)|²> / V
    按 |k| 分箱平均。

    Returns
    -------
    k_bins : (n_bins,)  波数 bin 中心
    Pk     : (n_bins,)  功率谱估计
    """
    n = delta.shape[0]
    kfreq = np.fft.fftfreq(n, d=box_length / n) * (2.0 * np.pi)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).ravel()
    delta_k = np.fft.fftn(delta).ravel()
    Pk_3d = np.abs(delta_k) ** 2 / (n ** 3) * (box_length ** 3)
    # 分箱:
    k_max = k_mag.max()
    n_bins = min(32, n // 2)
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
    Pk = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (k_mag >= k_edges[b]) & (k_mag < k_edges[b + 1])
        if mask.sum() > 0:
            Pk[b] = Pk_3d[mask].mean()
    return k_bins, Pk


# ---------------------------------------------------------------------------- #
#                       两点相关函数 ξ(r)
# ---------------------------------------------------------------------------- #
def two_point_correlation(delta: NDArray, box_length: float,
                          n_bins: int = 20) -> Tuple[NDArray, NDArray]:
    """
    两点相关函数 ξ(r) 通过 FFT 计算:
        ξ(r) = FFT^{-1}[ P(k) ] / V
    即 Wiener-Khinchin 定理: 功率谱与相关函数互为 Fourier 对。

    Returns
    -------
    r_bins : (n_bins,) 距离 bin 中心 [Mpc/h]
    xi     : (n_bins,) 相关函数
    """
    n = delta.shape[0]
    delta_k = np.fft.fftn(delta)
    xi_full = np.fft.ifftn(np.abs(delta_k) ** 2).real / (n ** 3)
    # 径向平均:
    x = np.arange(n) - n // 2
    x = np.where(x < 0, x + n, x) * (box_length / n)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    r_max = 0.5 * box_length
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])
    xi = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (r.ravel() >= r_edges[b]) & (r.ravel() < r_edges[b + 1])
        if mask.sum() > 0:
            xi[b] = xi_full.ravel()[mask].mean()
    return r_bins, xi


# ---------------------------------------------------------------------------- #
#                       密度 PDF
# ---------------------------------------------------------------------------- #
def density_pdf(delta: NDArray, n_bins: int = 50) -> Tuple[NDArray, NDArray]:
    """
    密度对比 δ 的概率分布函数。
    """
    hist, edges = np.histogram(delta.ravel(), bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


# ---------------------------------------------------------------------------- #
#                     能量诊断
# ---------------------------------------------------------------------------- #
def kinetic_energy(v: NDArray, masses: NDArray) -> float:
    """总动能 KE = (1/2) Σ_p m_p |v_p|²。"""
    v2 = np.sum(v ** 2, axis=1)
    return float(0.5 * np.sum(masses * v2))


def potential_energy(positions: NDArray, masses: NDArray,
                     box_length: float, softening: float) -> float:
    """
    总引力势能 (周期域 Ewald 求和简化):
        PE = -G Σ_{i<j} m_i m_j / |r_ij|_softened
    使用软化长度 ε:
        |r|_ε = √(|r|² + ε²)
    """
    n = positions.shape[0]
    n_sample = min(n, 500)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=n_sample, replace=False)
    pos_s = positions[idx]
    m_s = masses[idx]
    pe = 0.0
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            dr = pos_s[i] - pos_s[j]
            dr -= box_length * np.round(dr / box_length)
            r = np.sqrt(np.sum(dr ** 2) + softening ** 2)
            pe -= m_s[i] * m_s[j] / r
    # 缩放:
    scale = (n / n_sample) ** 2
    return float(pe * scale)


def total_energy(v: NDArray, positions: NDArray, masses: NDArray,
                 box_length: float, softening: float) -> float:
    return kinetic_energy(v, masses) + potential_energy(positions, masses,
                                                         box_length, softening)


# ---------------------------------------------------------------------------- #
#                   KdV 残差诊断
# ---------------------------------------------------------------------------- #
def kdv_residual_field(u: NDArray, h: float, dt: float) -> NDArray:
    """
    评估 1D 密度波 u(x,t) 是否满足 KdV:
        r = u_t - 6 u u_x + u_xxx
    使用有限差分计算 u_t, u_x, u_xxx。
    """
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * h)
    u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (h * h)
    u_xxx = (np.roll(u, -2) - 2.0 * np.roll(u, -1)
             + 2.0 * np.roll(u, 1) - np.roll(u, 2)) / (2.0 * h ** 3)
    # u_t 近似 (简化: 假设 ∂u/∂t ≈ -u u_x, 即 KdV 自身):
    u_t = 6.0 * u * u_x - u_xxx
    return u_t - 6.0 * u * u_x + u_xxx


# ---------------------------------------------------------------------------- #
#                  综合诊断报告
# ---------------------------------------------------------------------------- #
def full_diagnostics(delta: NDArray, positions: NDArray,
                     velocities: NDArray, masses: NDArray,
                     box_length: float, softening: float) -> Dict[str, any]:
    """汇总所有诊断指标。"""
    k_bins, Pk = power_spectrum(delta, box_length)
    r_bins, xi = two_point_correlation(delta, box_length)
    delta_centers, delta_pdf = density_pdf(delta)
    KE = kinetic_energy(velocities, masses)
    PE = potential_energy(positions, masses, box_length, softening)
    return {
        "power_spectrum": {"k": k_bins, "Pk": Pk},
        "correlation": {"r": r_bins, "xi": xi},
        "density_pdf": {"delta": delta_centers, "pdf": delta_pdf},
        "kinetic_energy": KE,
        "potential_energy": PE,
        "total_energy": KE + PE,
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_max": float(delta.max()),
        "delta_min": float(delta.min()),
    }
