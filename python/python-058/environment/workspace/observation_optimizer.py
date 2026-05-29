"""
最优观测网络布局模块 (Optimal Observation Network Placement)

集成种子项目:
- 146_ccvt_reflect: 约束重心 Voronoi 镶嵌 (CCVT) 与 Lloyd 迭代

科学背景:
  中尺度对流系统 (MCS) 的数值预报精度严重依赖于观测网络的密度与空间分布.
  使用 Centroidal Voronoi Tessellation (CVT) 优化雷达/自动气象站的空间布局,
  使得观测点均匀覆盖预报区域, 同时最小化 CVT 能量:

    E = Σ_i ∫_{V_i} ρ(x) ||x - g_i||² dx

  其中 V_i 为 Voronoi 单元, g_i 为生成子 (generator).

核心公式:
  Lloyd 算法:
    1. 在区域内随机采样大量点
    2. 对每个样本, 找到最近的生成子
    3. 更新生成子为对应样本的质心:
         g_i^{new} = (Σ_{x∈V_i} x) / |V_i|
    4. 对超出边界的生成子进行反射/投影处理
    5. 重复直至收敛

  CVT 能量单调递减, 收敛到局部最优.
"""

import numpy as np
from typing import Tuple, List


def find_closest_generators(samples: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """
    为每个样本找到最近的生成子索引.
    """
    n_samples = len(samples)
    n_gen = len(generators)
    closest = np.zeros(n_samples, dtype=int)
    for s in range(n_samples):
        dists = np.sum((generators - samples[s])**2, axis=1)
        closest[s] = int(np.argmin(dists))
    return closest


def cvt_energy(samples: np.ndarray, generators: np.ndarray) -> float:
    """
    计算离散 CVT 能量 (基于 146_ccvt_reflect/cvt_energy).

    E = (1/S) Σ_s min_j ||r_j - s||²
    """
    n_samples = len(samples)
    total = 0.0
    for s in range(n_samples):
        dists = np.sum((generators - samples[s])**2, axis=1)
        total += np.min(dists)
    return total / n_samples


def ccvt_reflect_2d(n_generators: int, domain: Tuple[float, float, float, float],
                    max_iter: int = 100, sample_num: int = 10000,
                    tol: float = 1e-6) -> np.ndarray:
    """
    在二维矩形域上使用反射边界处理生成约束 CVT (基于 146_ccvt_reflect).

    参数:
      n_generators: 生成子数量
      domain: (xmin, xmax, ymin, ymax)
      max_iter: 最大 Lloyd 迭代次数
      sample_num: 每次迭代的 Monte Carlo 采样数
      tol: 收敛容差

    返回:
      generators: (n_generators, 2) 最优生成子位置
    """
    xmin, xmax, ymin, ymax = domain

    # 初始化生成子 (均匀随机)
    np.random.seed(42)
    generators = np.column_stack([
        np.random.uniform(xmin, xmax, n_generators),
        np.random.uniform(ymin, ymax, n_generators)
    ])

    energy_history = []

    for it in range(max_iter):
        # Monte Carlo 采样
        samples = np.column_stack([
            np.random.uniform(xmin, xmax, sample_num),
            np.random.uniform(ymin, ymax, sample_num)
        ])

        # 分配样本到 Voronoi 单元
        closest = find_closest_generators(samples, generators)

        # 计算新质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        for s in range(sample_num):
            g_idx = closest[s]
            new_generators[g_idx] += samples[s]
            counts[g_idx] += 1.0

        # 更新生成子, 反射边界处理
        for g in range(n_generators):
            if counts[g] > 0:
                new_generators[g] /= counts[g]
            else:
                # 空单元回退到原位置附近随机扰动
                new_generators[g] = generators[g] + np.random.randn(2) * 0.01 * (xmax - xmin)

            # 反射边界处理
            if new_generators[g, 0] < xmin:
                new_generators[g, 0] = 2.0 * xmin - new_generators[g, 0]
            if new_generators[g, 0] > xmax:
                new_generators[g, 0] = 2.0 * xmax - new_generators[g, 0]
            if new_generators[g, 1] < ymin:
                new_generators[g, 1] = 2.0 * ymin - new_generators[g, 1]
            if new_generators[g, 1] > ymax:
                new_generators[g, 1] = 2.0 * ymax - new_generators[g, 1]

            # 再次截断 (防止反射后仍越界)
            new_generators[g, 0] = max(xmin, min(xmax, new_generators[g, 0]))
            new_generators[g, 1] = max(ymin, min(ymax, new_generators[g, 1]))

        # 收敛判断
        shift = np.max(np.linalg.norm(new_generators - generators, axis=1))
        generators = new_generators
        energy = cvt_energy(samples, generators)
        energy_history.append(energy)

        if shift < tol * (xmax - xmin) and it > 10:
            break

    return generators


class ObservationNetworkOptimizer:
    """
    中尺度对流系统观测网络优化器.
    """

    def __init__(self, domain_km: Tuple[float, float, float, float] = (0.0, 200.0, 0.0, 200.0)):
        self.domain = domain_km  # (xmin, xmax, ymin, ymax) in km

    def optimize_radar_placement(self, n_radars: int = 5) -> np.ndarray:
        """
        优化雷达站点布局.
        """
        return ccvt_reflect_2d(n_radars, self.domain, max_iter=80, sample_num=5000)

    def optimize_station_network(self, n_stations: int = 20) -> np.ndarray:
        """
        优化自动气象站网络布局.
        """
        return ccvt_reflect_2d(n_stations, self.domain, max_iter=60, sample_num=8000)

    def coverage_score(self, generators: np.ndarray, n_test: int = 5000) -> float:
        """
        计算空间覆盖均匀度 (越接近 1 越均匀).
        """
        xmin, xmax, ymin, ymax = self.domain
        samples = np.column_stack([
            np.random.uniform(xmin, xmax, n_test),
            np.random.uniform(ymin, ymax, n_test)
        ])
        energy = cvt_energy(samples, generators)
        # 归一化: 理论最优能量 ~ 域面积 / (π * n)
        area = (xmax - xmin) * (ymax - ymin)
        n_gen = len(generators)
        optimal_energy = area / (np.pi * n_gen) * 0.5
        score = optimal_energy / (energy + 1e-10)
        return min(1.0, score)
