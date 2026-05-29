"""
pdf_sampler.py
================================================================================
二维概率分布采样模块 (来源于 542_histogram_pdf_2d_sample 项目)
================================================================================
本模块提供基于离散累积分布函数 (CDF) 的二维随机采样方法，用于
海洋流速场的不确定性量化。在潮汐能功率预测中，流速的时空分布
存在显著不确定性，本模块通过二维直方图PDF进行蒙特卡洛采样，
评估功率输出的统计特性。

核心公式:
    离散CDF:
        CDF(i,j) = Σ_{p≤i, q≤j} PDF(p,q) / Σ PDF

    逆变换采样:
        给定均匀随机变量 U ~ Uniform(0,1)，找到最小 (i,j) 使得
        CDF(i,j) ≥ U，然后在对应网格内均匀采样。

    功率统计矩:
        <P> = 1/N Σ_k P(u_k)
        σ²_P = 1/(N-1) Σ_k (P(u_k) - <P>)²
"""

import numpy as np
from typing import Tuple


def set_discrete_cdf(pdf_mat: np.ndarray) -> np.ndarray:
    """
    从二维概率密度矩阵构造累积分布函数。

    参数:
        pdf_mat: 二维 PDF 矩阵 (m1, m2)，非负

    返回:
        cdf_mat: 对应的 CDF 矩阵
    """
    pdf = np.asarray(pdf_mat, dtype=float)
    if np.any(pdf < 0):
        raise ValueError("set_discrete_cdf: PDF 元素不能为负")
    total = np.sum(pdf)
    if total < 1e-14:
        raise ValueError("set_discrete_cdf: PDF 总和为零")
    pdf_norm = pdf / total
    cdf = np.cumsum(np.cumsum(pdf_norm, axis=0), axis=1)
    return cdf


def discrete_cdf_to_xy(
    m1: int,
    m2: int,
    cdf_mat: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    n: int,
    u: np.ndarray,
) -> np.ndarray:
    """
    根据离散CDF值找到对应的二维采样点。

    参数:
        m1, m2: PDF 网格维度
        cdf_mat: CDF 矩阵 (m1, m2)
        xb: x方向边界，长度 m1+1
        yb: y方向边界，长度 m2+1
        n: 采样点数
        u: 均匀随机数，长度 n

    返回:
        s: 采样点坐标 (2, n)
    """
    s = np.zeros((2, n))
    low = 0.0
    cdf = np.asarray(cdf_mat)
    for j in range(m2):
        for i in range(m1):
            high = cdf[i, j]
            mask = (low <= u) & (u <= high)
            count = np.count_nonzero(mask)
            if count > 0:
                r = np.random.rand(2, count)
                idx = np.where(mask)[0]
                s[0, idx] = (1.0 - r[0, :]) * xb[i] + r[0, :] * xb[i + 1]
                s[1, idx] = (1.0 - r[1, :]) * yb[j] + r[1, :] * yb[j + 1]
            low = high
    return s


def sample_velocity_2d(
    pdf_mat: np.ndarray,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_samples: int = 1000,
) -> np.ndarray:
    """
    从二维流速PDF中采样速度向量。

    参数:
        pdf_mat: 二维 PDF 矩阵 (u, v 方向的离散概率)
        u_range: u 速度范围 (min, max)
        v_range: v 速度范围 (min, max)
        n_samples: 采样点数

    返回:
        samples: 速度采样 (2, n_samples)，[u; v]
    """
    m1, m2 = pdf_mat.shape
    cdf_mat = set_discrete_cdf(pdf_mat)
    xb = np.linspace(u_range[0], u_range[1], m1 + 1)
    yb = np.linspace(v_range[0], v_range[1], m2 + 1)
    u_rand = np.random.rand(n_samples)
    samples = discrete_cdf_to_xy(m1, m2, cdf_mat, xb, yb, n_samples, u_rand)
    return samples


def estimate_power_statistics(
    pdf_mat: np.ndarray,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    turbine_area: float = 20.0,
    rho: float = 1025.0,
    n_samples: int = 5000,
) -> dict:
    """
    通过蒙特卡洛采样估计潮汐涡轮的功率统计特性。

    物理模型:
        功率系数 C_p = 4a(1-a)² ≈ 0.593 (Betz极限)
        瞬时功率: P = ½ ρ A C_p |V|³

    参数:
        pdf_mat: 二维流速 PDF
        u_range, v_range: 速度范围
        turbine_area: 涡轮扫掠面积 (m²)
        rho: 水密度
        n_samples: 采样数

    返回:
        包含 mean_power, std_power, max_power, capacity_factor 的字典
    """
    samples = sample_velocity_2d(pdf_mat, u_range, v_range, n_samples)
    u = samples[0, :]
    v = samples[1, :]
    speed = np.sqrt(u ** 2 + v ** 2)

    # Betz 极限功率系数
    cp = 16.0 / 27.0
    power = 0.5 * rho * turbine_area * cp * speed ** 3

    # 额定功率截断（典型潮汐涡轮额定流速约 2.5 m/s）
    rated_speed = 2.5
    rated_power = 0.5 * rho * turbine_area * cp * rated_speed ** 3
    power = np.minimum(power, rated_power)

    mean_p = float(np.mean(power))
    std_p = float(np.std(power))
    max_p = float(np.max(power))
    cap_factor = mean_p / rated_power if rated_power > 0 else 0.0

    return {
        "mean_power": mean_p,
        "std_power": std_p,
        "max_power": max_p,
        "rated_power": rated_power,
        "capacity_factor": cap_factor,
        "samples": samples,
        "speeds": speed,
    }
