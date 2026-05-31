#!/usr/bin/env python3
"""
optimal_sampling.py
最优水含量测点布置模块（源自 cvt_square_pdf_discrete 项目）

利用 Centroidal Voronoi Tessellation (CVT) 与 Lloyd 迭代，
在燃料电池膜平面内寻找最优传感器布置点，使得每个传感器位于
其监测区域的密度加权质心。

数学模型：
    给定密度函数 ρ(x,y)（反映膜内水含量变化剧烈程度），
    最小化能量泛函：
        F(z_1,...,z_N) = Σ_i ∫_{V_i} ρ(x) · ||x - z_i||² dx
    其中 V_i 为 Voronoi 区域。

    Lloyd 迭代：z_i^{k+1} = centroid(V_i^k)
"""

import numpy as np


def get_discrete_pdf(nx=20, ny=20):
    """
    定义膜平面上的离散概率密度函数（PDF），反映水含量监测优先级。
    对应原项目 get_discrete_pdf.m。

    中心区域（活性区）密度高，边缘密度低（模拟反应集中区）。
    """
    pdf = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) / nx
            y = (j + 0.5) / ny
            # 双峰密度：模拟流道下方与脊下方不同的水管理需求
            pdf[i, j] = (np.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.05) +
                         0.7 * np.exp(-((x - 0.7) ** 2 + (y - 0.5) ** 2) / 0.08))
    # 归一化
    pdf = pdf / np.sum(pdf)
    return pdf


def set_discrete_cdf(pdf):
    """
    由 PDF 计算累积分布函数 CDF，对应原项目 set_discrete_cdf.m。
    """
    cdf = np.cumsum(pdf.flatten())
    return cdf


def discrete_pdf_sample(n_samples, pdf, seed=42):
    """
    从离散 PDF 中采样 n_samples 个点，对应原项目 discrete_pdf_sample.m。
    方法：CDF 反演法 + 单元内均匀随机偏移。
    """
    rng = np.random.default_rng(seed)
    nx, ny = pdf.shape
    cdf = set_discrete_cdf(pdf)

    samples = np.zeros((n_samples, 2))
    for k in range(n_samples):
        u = rng.random()
        idx = np.searchsorted(cdf, u)
        idx = min(idx, nx * ny - 1)
        i = idx // ny
        j = idx % ny
        # 在单元内均匀偏移
        dx = rng.random() / nx
        dy = rng.random() / ny
        samples[k, 0] = (i / nx) + dx
        samples[k, 1] = (j / ny) + dy

    return samples


def lloyd_iteration(generators, pdf, n_iter=20):
    """
    Lloyd 算法迭代，对应原项目 cvt_square_pdf_discrete.m 的核心循环。

    对于每个生成点，计算其 Voronoi 区域在离散 PDF 下的质心，
    并将生成点移至质心。
    """
    nx, ny = pdf.shape
    gens = np.asarray(generators, dtype=float).copy()
    n_gens = gens.shape[0]

    # 构建网格坐标
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    for _ in range(n_iter):
        new_gens = np.zeros_like(gens)
        counts = np.zeros(n_gens)

        # 对每个网格点，找到最近的生成点
        for i in range(nx):
            for j in range(ny):
                px, py = X[i, j], Y[i, j]
                dists = np.sum((gens - [px, py]) ** 2, axis=1)
                k = np.argmin(dists)
                new_gens[k, 0] += px * pdf[i, j]
                new_gens[k, 1] += py * pdf[i, j]
                counts[k] += pdf[i, j]

        # 更新生成点为质心
        for k in range(n_gens):
            if counts[k] > 1e-12:
                new_gens[k] /= counts[k]
            else:
                # 若区域为空，随机重置
                new_gens[k] = np.random.rand(2)

        gens = new_gens

    return gens


def optimize_sensor_placement(params, n_sensors=None, n_iter=30):
    """
    在膜平面上优化水含量传感器布置。

    Parameters
    ----------
    params : dict
        物理参数（含 N_sensors）
    n_sensors : int, optional
        传感器数量，默认从 params 读取
    n_iter : int
        Lloyd 迭代次数

    Returns
    -------
    sensors : ndarray, shape (N, 2)
        最优传感器平面坐标 [m]
    """
    if n_sensors is None:
        n_sensors = params.get('N_sensors', 16)

    # 离散密度
    pdf = get_discrete_pdf(nx=40, ny=40)

    # 初始生成点：从 PDF 采样
    gens = discrete_pdf_sample(n_sensors, pdf, seed=42)

    # Lloyd 迭代优化
    gens_opt = lloyd_iteration(gens, pdf, n_iter=n_iter)

    # 缩放到实际膜尺寸（假设膜面 10cm × 10cm）
    L_mem = 0.1  # [m]
    sensors = gens_opt * L_mem

    return sensors


if __name__ == '__main__':
    p = {'N_sensors': 16}
    sensors = optimize_sensor_placement(p)
    print("Sensors shape:", sensors.shape)
    print("Sensor range:", sensors.min(), sensors.max())
