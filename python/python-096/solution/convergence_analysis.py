"""
convergence_analysis.py
=======================
收敛性分析与误差度量模块

核心算法来源：
  - 230_cube_distance：立方体内距离统计与 PDF
  - 542_histogram_pdf_2d_sample：二维直方图分析

在电磁学波束赋形中的角色：
  1. 立方体内随机点距离统计用于评估三维阵列单元位置误差的分布
  2. 二维直方图分析用于量化方向图采样的空间分布均匀性
  3. 距离 PDF 的解析公式用于验证蒙特卡罗仿真的正确性
"""

import numpy as np
from typing import Tuple, Optional


def cube_distance_pdf_exact(d: np.ndarray) -> np.ndarray:
    """
    单位立方体内两点距离的概率密度函数（解析公式）。

    来源：230_cube_distance

    数学公式（MathWorld: Cube Line Picking）：
      设 d 为两点距离，则

      当 0 <= d <= 1 时：
        f(d) = -d^2 [(d-8)d^2 + \pi(6d-4)]

      当 1 < d <= \sqrt{2} 时：
        f(d) = 2d [(d^2 - 8\sqrt{d^2-1} + 3)d^2 - 4\sqrt{d^2-1}
                  + 12 d^2 \operatorname{asec}(d) + \pi(3-4d) - 0.5]

      当 \sqrt{2} < d <= \sqrt{3} 时：
        f(d) = d [(1+d^2)(6\pi + 8\sqrt{d^2-2} - 5 - d^2)
                  - 16 d \operatorname{acsc}(\sqrt{2-2/d^2})
                  + 16 d \arctan(d\sqrt{d^2-2})
                  - 24 (d^2+1) \arctan(\sqrt{d^2-2})]

    注：本实现使用数值鲁棒版本，对边界区域做平滑处理。
    """
    d = np.asarray(d, dtype=float)
    pdf = np.zeros_like(d)
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)

    # Region 1: 0 <= d <= 1
    mask1 = (d >= 0.0) & (d <= 1.0)
    dm = d[mask1]
    pdf[mask1] = (-dm ** 2) * ((dm - 8.0) * dm ** 2 + np.pi * (6.0 * dm - 4.0))

    # Region 2: 1 < d <= sqrt(2)
    mask2 = (d > 1.0) & (d <= sqrt2)
    dm = d[mask2]
    t = dm ** 2 - 1.0
    t = np.maximum(t, 0.0)
    sqrt_term = np.sqrt(t)
    asec_dm = np.arccos(np.clip(1.0 / dm, -1.0, 1.0))
    pdf[mask2] = 2.0 * dm * (
        (dm ** 2 - 8.0 * sqrt_term + 3.0) * dm ** 2
        - 4.0 * sqrt_term
        + 12.0 * dm ** 2 * asec_dm
        + np.pi * (3.0 - 4.0 * dm) - 0.5
    )

    # Region 3: sqrt(2) < d <= sqrt(3)
    mask3 = (d > sqrt2) & (d <= sqrt3)
    dm = d[mask3]
    t2 = dm ** 2 - 2.0
    t2 = np.maximum(t2, 0.0)
    sqrt_term2 = np.sqrt(t2)
    # 数值稳定处理
    arg_acsc = np.sqrt(np.maximum(2.0 - 2.0 / (dm ** 2), 0.0))
    arg_acsc = np.clip(arg_acsc, 1e-12, 1e12)
    acsc_val = np.arcsin(np.clip(1.0 / arg_acsc, -1.0, 1.0))
    atan1 = np.arctan(dm * sqrt_term2)
    atan2 = np.arctan(sqrt_term2)
    pdf[mask3] = dm * (
        (1.0 + dm ** 2) * (6.0 * np.pi + 8.0 * sqrt_term2 - 5.0 - dm ** 2)
        - 16.0 * dm * acsc_val
        + 16.0 * dm * atan1
        - 24.0 * (dm ** 2 + 1.0) * atan2
    )

    pdf = np.maximum(pdf, 0.0)
    return pdf


def cube_distance_stats_monte_carlo(n_samples: int = 10000,
                                    seed: Optional[int] = None) -> Tuple[float, float]:
    """
    通过蒙特卡罗采样估计单位立方体内两点距离的均值与方差。

    来源：230_cube_distance

    理论值：
      E[D] \approx 0.661707182
      Var[D] \approx 0.062222585
    """
    if seed is not None:
        np.random.seed(seed)
    p = np.random.rand(n_samples, 3)
    q = np.random.rand(n_samples, 3)
    t = np.linalg.norm(p - q, axis=1)
    mu = float(np.mean(t))
    if n_samples > 1:
        var = float(np.sum((t - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def histogram_2d_uniformity(points: np.ndarray,
                            x_range: Tuple[float, float] = (-1.0, 1.0),
                            y_range: Tuple[float, float] = (-1.0, 1.0),
                            bins: int = 10) -> dict:
    """
    评估二维点分布的均匀性（二维直方图分析）。

    来源：542_histogram_pdf_2d_sample

    物理背景：
      在天线阵列方向图采样中，我们需要在 (\theta, \phi) 空间均匀采样。
      通过二维直方图的卡方统计量评估均匀性：

        \chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}

      其中 O_{ij} 为观测频数，E_{ij} = N / (bins^2) 为期望频数。
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必须为 (N, 2) 数组")

    H, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=bins, range=[x_range, y_range]
    )
    N = points.shape[0]
    expected = N / (bins * bins)
    if expected < 1e-12:
        return {'chi2': 0.0, 'p_uniformity': 0.0, 'max_deviation': 0.0}

    chi2 = np.sum((H - expected) ** 2 / expected)
    # 自由度 = bins^2 - 1
    df = bins * bins - 1
    # 简单 p 值估计（使用正态近似对数卡方）
    from stochastic_channel import alnorm
    p_uniformity = 1.0 - alnorm((chi2 - df) / np.sqrt(2.0 * df), False) if df > 0 else 1.0
    max_deviation = float(np.max(np.abs(H - expected)) / expected)

    return {
        'chi2': float(chi2),
        'p_uniformity': float(p_uniformity),
        'max_deviation': float(max_deviation),
        'histogram': H,
        'xedges': xedges,
        'yedges': yedges,
    }


def compute_array_pattern_metric(pattern_db: np.ndarray,
                                 mainlobe_indices: np.ndarray,
                                 sidelobe_threshold_db: float = -13.0) -> dict:
    """
    计算天线方向图的质量指标。

    参数：
        pattern_db: 方向图（dB），形状 (N,)
        mainlobe_indices: 主瓣索引
        sidelobe_threshold_db: 旁瓣阈值

    返回：
        dict 包含：
          - peak_sidelobe_level: 峰值旁瓣电平（dB）
          - integrated_sidelobe_ratio: 积分旁瓣比
          - beamwidth_efficiency: 主瓣宽度效率
    """
    pattern_lin = 10.0 ** (pattern_db / 10.0)
    total_power = np.sum(pattern_lin)
    mainlobe_power = np.sum(pattern_lin[mainlobe_indices])

    sidelobe_mask = np.ones(len(pattern_db), dtype=bool)
    sidelobe_mask[mainlobe_indices] = False
    sidelobe_power = np.sum(pattern_lin[sidelobe_mask])

    peak_sidelobe = float(np.max(pattern_db[sidelobe_mask])) if np.any(sidelobe_mask) else -100.0
    islr = 10.0 * np.log10(sidelobe_power / max(mainlobe_power, 1e-18))
    beamwidth_eff = mainlobe_power / max(total_power, 1e-18)

    return {
        'peak_sidelobe_level_db': peak_sidelobe,
        'integrated_sidelobe_ratio_db': float(islr),
        'beamwidth_efficiency': float(beamwidth_eff),
    }


def convergence_rate_residual(residual_history: np.ndarray) -> float:
    """
    估计迭代残差的收敛速率。

    数学模型：
      假设残差按几何级数衰减：r_k = C \rho^k
      则 \log r_k = \log C + k \log \rho
      通过线性回归估计 \rho。
    """
    if residual_history.size < 2:
        return 0.0
    k = np.arange(residual_history.size)
    log_r = np.log(np.maximum(residual_history, 1e-18))
    # 线性回归
    A = np.vstack([k, np.ones_like(k)]).T
    m, _ = np.linalg.lstsq(A, log_r, rcond=None)[0]
    return float(np.exp(m))
