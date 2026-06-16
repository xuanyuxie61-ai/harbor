# -*- coding: utf-8 -*-
"""
spectral_sampler.py
======================================================================
非均匀波长采样策略 —— CVT + Diaphony + Latin Hypercube

物理背景:
    系外行星大气光谱反演的观测数据通常分布在非均匀波长网格上,
    因为:
    (1) 强吸收线附近需要高分辨率采样 (光学厚度变化剧烈)
    (2) 连续谱区可低分辨率采样 (变化平缓)
    (3) 观测仪器噪声特性在不同波段差异显著

    为构建最优采样策略, 采用三层级方法:
    (a) CVT (Centroidal Voronoi Tessellation) 生成非均匀采样点
        (来自 253_cvt_circle_nonuniform)
    (b) Diaphony 检验采样点集的均匀性质量
        (来自 276_diaphony)
    (c) Latin Hypercube Sampling 对大气参数空间采样
        (来自 653_latinize)

数学公式:
    CVT 能量泛函:
        E({z_i}) = sum_i integral_{V_i} rho(x) |x - z_i|^2 dx
    其中 rho(x) 为密度函数 (此处取吸收系数梯度),
    V_i 为 Voronoi 区域。

    Diaphony:
        F_N^*(X) = (1/N^2) sum_{k != 0} |sum_{n=1}^N exp(2 pi i k.x_n)|^2
                       / prod_j max(1, |k_j|^(d+1)/2)
    衡量点集 X 在单位超立方体上的均匀程度。

依赖: numpy
======================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


# ============================================================
# Latin Hypercube Sampling (移植自 653_latinize)
# ============================================================
def heap_sort_index(a: np.ndarray) -> np.ndarray:
    """
    堆排序索引 (移植自 653_latinize 的 r8vec_sort_heap_index_a)。
    返回索引数组 indx, 使得 a[indx] 为升序排列。
    """
    n = len(a)
    if n < 1:
        return np.array([], dtype=int)

    indx = np.arange(n, dtype=int)

    if n == 1:
        return indx

    l = n // 2 + 1
    ir = n

    while True:
        if l > 1:
            l = l - 1
            ind = indx[l - 1]
            val = a[ind]
        else:
            ind = indx[ir - 1]
            val = a[ir - 1]
            indx[ir - 1] = indx[0]
            ir = ir - 1
            if ir == 1:
                indx[0] = ind
                break

        i = l
        j = 2 * l

        while j <= ir:
            if j < ir:
                if a[indx[j - 1]] < a[indx[j]]:
                    j = j + 1
            if val < a[indx[j - 1]]:
                indx[i - 1] = indx[j - 1]
                i = j
                j = 2 * i
            else:
                j = ir + 1

        indx[i - 1] = ind

    return indx


def latinize(m: int, n: int, table: np.ndarray) -> np.ndarray:
    """
    Latinize 一个 m x n 数据集 (移植自 653_latinize)。
    每行保持最小/最大值不变, 但使元素等间距分布。

    物理意义: 对大气参数 (T_eq, log g, C/O, Fe/H 等) 构建
    Latin Hypercube 样本, 确保参数空间覆盖均匀。
    """
    if m <= 2:
        return table.copy()

    result = table.copy()
    for j in range(n):
        col = result[:, j]
        v_min = np.min(col)
        v_max = np.max(col)
        indx = heap_sort_index(col)
        for i in range(m):
            result[indx[i], j] = ((m - i) * v_min + (i - 1) * v_max) / (m - 1)
    return result


def generate_lhs_samples(
    n_samples: int,
    parameter_ranges: Dict[str, Tuple[float, float]],
    rng_seed: int = 42,
) -> np.ndarray:
    """
    生成大气参数的 Latin Hypercube 样本。
    参数范围:
        T_eq   : [800, 2500] K
        log_g  : [2.5, 4.5] (cgs)
        C_O    : [0.3, 1.2]
        Fe_H   : [-1.0, 1.0] (dex)
        log_Kzz: [7, 12] (cm^2/s)
    """
    rng = np.random.default_rng(rng_seed)
    n_params = len(parameter_ranges)
    table = rng.random((n_samples, n_params))

    for j, (key, (vmin, vmax)) in enumerate(parameter_ranges.items()):
        table[:, j] = vmin + table[:, j] * (vmax - vmin)

    return latinize(n_samples, n_params, table)


# ============================================================
# CVT 非均匀采样 (移植自 253_cvt_circle_nonuniform)
# ============================================================
def compute_density_weights(
    wavelength_grid: np.ndarray,
    absorption_coeff: np.ndarray,
) -> np.ndarray:
    """
    计算波长网格上的采样密度权重。
    密度正比于吸收系数的梯度:
        rho(lambda) ~ |d kappa / d lambda| + epsilon
    使得吸收线中心附近采样更密。
    """
    dkappa = np.abs(np.gradient(absorption_coeff, wavelength_grid))
    rho = dkappa + 1.0e-6 * np.max(dkappa + 1.0e-30)
    rho = rho / np.sum(rho)
    return rho


def cvt_iteration(
    generators: np.ndarray,
    sample_points: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 20,
) -> np.ndarray:
    """
    Lloyd 算法进行 CVT 迭代。

    每次迭代:
    1. 将 sample_points 分配到最近的 generator (Voronoi)
    2. 计算每个 Voronoi 区域的加权质心
    3. 更新 generator 位置
    """
    gen = generators.copy()
    n_gen = len(gen)

    for it in range(n_iter):
        # 距离矩阵: (n_sample, n_gen)
        dist = np.abs(sample_points[:, None] - gen[None, :])
        labels = np.argmin(dist, axis=1)

        for k in range(n_gen):
            mask = labels == k
            if np.sum(mask) == 0:
                continue
            w_k = weights[mask]
            x_k = sample_points[mask]
            gen[k] = np.sum(w_k * x_k) / np.sum(w_k)

        gen = np.sort(gen)

    return gen


def build_cvt_wavelength_grid(
    wl_min_um: float,
    wl_max_um: float,
    n_points: int,
    absorption_profile: np.ndarray,
    reference_grid: np.ndarray,
    n_samples: int = 5000,
    n_iter: int = 30,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    基于 CVT 构建非均匀波长采样网格。

    输入:
        wl_min_um, wl_max_um : 波长范围 [微米]
        n_points             : 目标采样点数
        absorption_profile   : 参考网格上的吸收系数
        reference_grid       : 参考波长网格 [微米]
    """
    rng = np.random.default_rng(rng_seed)

    density = compute_density_weights(reference_grid, absorption_profile)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]

    u_samples = rng.random(n_samples)
    sample_wl = np.interp(u_samples, cdf, reference_grid)
    sample_weights = np.interp(sample_wl, reference_grid, density)

    gen_init = np.linspace(wl_min_um, wl_max_um, n_points)
    mask = (sample_wl >= wl_min_um) & (sample_wl <= wl_max_um)
    gen = cvt_iteration(
        gen_init, sample_wl[mask], sample_weights[mask], n_iter=n_iter
    )

    return np.sort(gen)


# ============================================================
# Diaphony 均匀性检验 (移植自 276_diaphony)
# ============================================================
def diaphony_compute(points: np.ndarray) -> float:
    """
    计算点集的 diaphony (来自 276_diaphony 的 diaphony_compute)。

    Diaphony 是 discrepancy 的类似物, 衡量点集在单位超立方体上的
    均匀程度。值越小, 均匀性越好。

    F_N^*(X)^2 = (1/N^2) sum_{k in Z^d \\ {0}}
                     |sum_{n=1}^N exp(2 pi i k.x_n)|^2
                     / prod_j max(1, |k_j|^2)

    对一维点集, 简化为:
        D^2 = (1/N^2) sum_{k=1}^{K_max}
                  (2/k^2) |sum_{n=1}^N exp(2 pi i k x_n)|^2
    """
    if points.ndim == 1:
        points = points[:, None]

    n_points, d_dim = points.shape

    if np.min(points) < -1.0e-10 or np.max(points) > 1.0 + 1.0e-10:
        points = points - np.min(points)
        rng = np.max(points) - np.min(points)
        if rng > 1.0e-10:
            points = points / rng

    k_max = min(200, max(20, 5 * n_points))
    d2 = 0.0

    for k in range(1, k_max + 1):
        weight = 1.0 / (k * k)
        for dim in range(d_dim):
            phases = 2.0 * np.pi * k * points[:, dim]
            s_real = np.sum(np.cos(phases))
            s_imag = np.sum(np.sin(phases))
            d2 += weight * (s_real ** 2 + s_imag ** 2) / (n_points ** 2)

    return np.sqrt(d2)


def assess_sampling_quality(
    wavelength_grid: np.ndarray,
    wl_min: float,
    wl_max: float,
) -> Dict[str, float]:
    """
    评估波长采样质量:
    - diaphony: 均匀性度量
    - min_spacing: 最小间距
    - max_spacing: 最大间距
    - spacing_ratio: 最大/最小间距比
    """
    normalized = (wavelength_grid - wl_min) / max(wl_max - wl_min, 1.0e-30)
    d = diaphony_compute(normalized)

    spacings = np.diff(wavelength_grid)
    return {
        "diaphony": float(d),
        "min_spacing": float(np.min(spacings)) if len(spacings) > 0 else 0.0,
        "max_spacing": float(np.max(spacings)) if len(spacings) > 0 else 0.0,
        "spacing_ratio": float(np.max(spacings) / max(np.min(spacings), 1.0e-30))
        if len(spacings) > 0
        else 1.0,
        "n_points": len(wavelength_grid),
    }
