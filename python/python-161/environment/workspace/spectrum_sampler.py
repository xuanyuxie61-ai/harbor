"""
spectrum_sampler.py
基于种子项目 538_histogram_data_2d_sample (discrete CDF to XY sampling)
改造为 AM1.5 太阳光谱的离散概率采样器。

在钙钛矿太阳能电池模拟中，光子能量 E 与波长 λ 的联合分布可通过
二维离散累积分布函数 (CDF) 进行采样。本模块实现从 AM1.5G 标准
光谱的二维直方图中进行反变换采样，生成入射光子的能量与位置样本。

核心公式：
  1. 光子能量与波长关系：E = hc/λ，其中 h = 6.626e-34 J·s，c = 3.0e8 m/s
  2. AM1.5G 光谱辐照度 I(λ) [W·m^{-2}·nm^{-1}]
  3. 归一化联合 PDF：p(λ, θ) = I(λ)/(λ_max · I_total)  (θ 为入射角)
  4. 离散 CDF：C_{ij} = Σ_{m≤i} Σ_{n≤j} p_{mn} / Σ p_{mn}
  5. 反变换采样：给定 U ~ Uniform(0,1)，找到最小 (i,j) 使 C_{ij} ≥ U
"""

import numpy as np
from typing import Tuple

# 物理常数
PLANCK_CONSTANT: float = 6.62607015e-34   # J·s
SPEED_OF_LIGHT: float = 2.99792458e8      # m/s
ELEMENTARY_CHARGE: float = 1.602176634e-19  # C


def build_am15_spectrum(
    lambda_min: float = 280.0,
    lambda_max: float = 1200.0,
    n_bins: int = 64,
    theta_max_deg: float = 15.0,
    n_theta: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 AM1.5G 标准光谱的二维离散 PDF 与 CDF。

    Parameters
    ----------
    lambda_min, lambda_max : float
        波长范围 [nm]
    n_bins : int
        波长方向离散格点数
    theta_max_deg : float
        最大入射偏角 [度]，模拟非垂直入射
    n_theta : int
        入射角方向离散格点数

    Returns
    -------
    lambdas : (n_bins,) array
        波长格点中心 [nm]
    thetas : (n_theta,) array
        入射角格点中心 [度]
    pdf : (n_bins, n_theta) array
        归一化二维概率密度
    cdf : (n_bins, n_theta) array
        二维累积分布函数（按行优先展开为单调序列后累积）
    """
    if lambda_min >= lambda_max or lambda_min <= 0:
        raise ValueError("波长范围必须满足 0 < lambda_min < lambda_max")
    if n_bins <= 0 or n_theta <= 0:
        raise ValueError("离散格点数必须为正整数")

    lambdas = np.linspace(lambda_min, lambda_max, n_bins)
    thetas = np.linspace(0.0, theta_max_deg, n_theta)

    # AM1.5G 光谱近似解析公式（W·m^{-2}·nm^{-1}）
    # 基于分段黑体辐射近似 + 大气吸收修正
    def am15_irradiance(lam: np.ndarray) -> np.ndarray:
        lam_m = lam * 1e-9  # nm -> m
        # 黑体辐射谱 (T ~ 5778 K)，经大气透过率近似
        T_sun = 5778.0
        h, c, k_B = PLANCK_CONSTANT, SPEED_OF_LIGHT, 1.380649e-23
        # 普朗克定律
        term1 = (2.0 * h * c ** 2) / (lam_m ** 5)
        term2 = 1.0 / (np.exp((h * c) / (lam_m * k_B * T_sun)) - 1.0)
        bb = term1 * term2
        # 归一化到 AM1.5G 峰值 ~ 1.5 W·m^{-2}·nm^{-1} 附近
        bb_peak = bb.max() if bb.max() > 0 else 1.0
        am15 = 1000.0 * bb / bb_peak  # 缩放至约 1000 W/m^2 积分
        # 加入简单的大气吸收谷（水蒸气 940nm, 氧气 760nm）
        am15 *= (1.0 - 0.15 * np.exp(-((lam - 940.0) / 60.0) ** 2))
        am15 *= (1.0 - 0.08 * np.exp(-((lam - 760.0) / 40.0) ** 2))
        return np.maximum(am15, 0.0)

    irr = am15_irradiance(lambdas)

    # 入射角修正：朗伯余弦定律 + 透镜聚焦因子近似
    theta_rad = np.deg2rad(thetas)
    cos_factor = np.cos(theta_rad) * (1.0 + 0.05 * np.cos(2.0 * theta_rad))

    # 二维联合分布：p(λ, θ) ∝ I(λ) * cos_factor(θ)
    pdf = np.outer(irr, cos_factor)  # (n_bins, n_theta)

    # 边界处理：若全为零，则均匀分布
    total = pdf.sum()
    if total <= 0.0 or not np.isfinite(total):
        pdf = np.ones_like(pdf) / (n_bins * n_theta)
    else:
        pdf /= total

    # 按行优先构建单调 CDF
    cdf_flat = np.cumsum(pdf.ravel(order='C'))
    cdf = cdf_flat.reshape(pdf.shape, order='C')
    # 数值鲁棒性：确保最后一个元素精确为 1
    if cdf.size > 0:
        cdf.flat[-1] = 1.0

    return lambdas, thetas, pdf, cdf


def discrete_cdf_to_xy(
    n1: int,
    n2: int,
    cdf: np.ndarray,
    n_samples: int,
    u: np.ndarray,
) -> np.ndarray:
    """
    基于二维离散 CDF 的反变换采样，将均匀随机数映射到 (λ, θ) 样本。

    对应原项目 538_histogram_data_2d_sample 中 discrete_cdf_to_xy 的核心思想。

    Parameters
    ----------
    n1, n2 : int
        两个维度的离散格点数
    cdf : (n1, n2) array
        二维累积分布函数（单调递增，行优先）
    n_samples : int
        采样点数
    u : (n_samples,) array
        [0,1] 均匀随机数

    Returns
    -------
    xy : (2, n_samples) array
        第0行为归一化 λ，第1行为归一化 θ
    """
    if cdf.shape != (n1, n2):
        raise ValueError(f"CDF 形状 {cdf.shape} 与 ({n1},{n2}) 不符")
    if u.size != n_samples:
        raise ValueError("随机数数组长度与 n_samples 不符")
    if n_samples <= 0:
        return np.zeros((2, 0))

    # 数值鲁棒性：裁剪到 [0,1]
    u = np.clip(u, 0.0, 1.0)

    xy = np.zeros((2, n_samples))
    cdf_flat = cdf.ravel(order='C')

    # 向量化搜索：对每个 u，找到 cdf_flat 中第一个 ≥ u 的索引
    for k in range(n_samples):
        idx = np.searchsorted(cdf_flat, u[k], side='left')
        idx = min(idx, n1 * n2 - 1)
        i = idx // n2
        j = idx % n2

        # 在子格内均匀撒点（假设 PDF 在子格内常数）
        r = np.random.rand(2)
        xy[0, k] = (i + r[0]) / n1
        xy[1, k] = (j + r[1]) / n2

    return xy


def sample_photons(
    n_photons: int = 10000,
    lambda_min: float = 280.0,
    lambda_max: float = 1200.0,
    theta_max_deg: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 AM1.5G 光谱采样指定数量的光子 (λ, θ)。

    Returns
    -------
    lambdas_sample : (n_photons,) array [nm]
    thetas_sample : (n_photons,) array [度]
    """
    if n_photons <= 0:
        return np.array([]), np.array([])

    n_bins, n_theta = 64, 16
    lambdas, thetas, pdf, cdf = build_am15_spectrum(
        lambda_min, lambda_max, n_bins, theta_max_deg, n_theta
    )

    u = np.random.rand(n_photons)
    xy = discrete_cdf_to_xy(n_bins, n_theta, cdf, n_photons, u)

    # 将归一化坐标映射回物理量
    lambdas_sample = lambda_min + xy[0, :] * (lambda_max - lambda_min)
    thetas_sample = 0.0 + xy[1, :] * theta_max_deg

    return lambdas_sample, thetas_sample


def photon_energy_ev(lambda_nm: np.ndarray) -> np.ndarray:
    """
    将波长 [nm] 转换为光子能量 [eV]。
    E [eV] = h*c / (e * λ[m]) = 1239.8 / λ[nm]
    """
    lambda_nm = np.asarray(lambda_nm)
    with np.errstate(divide='ignore', invalid='ignore'):
        energy = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (ELEMENTARY_CHARGE * lambda_nm * 1e-9)
    energy = np.where(lambda_nm > 0, energy, 0.0)
    return energy


if __name__ == "__main__":
    # 自测试
    lams, thetas = sample_photons(1000)
    print(f"采样光子数: {len(lams)}, λ 范围 [{lams.min():.1f}, {lams.max():.1f}] nm")
    print(f"θ 范围 [{thetas.min():.2f}, {thetas.max():.2f}] deg")
    E = photon_energy_ev(lams)
    print(f"光子能量范围 [{E.min():.3f}, {E.max():.3f}] eV")
