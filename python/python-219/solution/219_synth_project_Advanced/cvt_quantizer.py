"""
cvt_quantizer.py
================

CVT (Centroidal Voronoi Tessellation) 最优空间量化模块。

融合种子项目:
  - 244_cvt_1d_lumping    : 1D Lloyd CVT 算法 (带 lumping 加权)
  - 440_florida_cvt_pop   : 人口密度加权的 2D CVT

在最优控制中, CVT 用于:
  1. 最优传感器/采样点布置 (最小化量化误差)
  2. 空间自适应网格生成
  3. 概率密度加权的最优离散化

数学公式:
---------
1. CVT 定义:
     生成器集合 {z_i}_{i=1}^n 使得每个 z_i 是其 Voronoi 区域 V_i
     的质心:
       z_i = (integral_{V_i} x rho(x) dx) / (integral_{V_i} rho(x) dx)

2. Lloyd 算法 (迭代):
     a. 给定 {z_i}, 构造 Voronoi 划分 {V_i}
     b. 计算每个 V_i 的质心
     c. 更新 z_i <- 质心
     d. 重复直到收敛

3. 量化误差 (能量):
     E = sum_{i=1}^n integral_{V_i} |x - z_i|^2 rho(x) dx

4. 1D lumping 版本 (来自 244):
     均匀采样 + 密度加权, 而非直接从 rho 采样

5. 人口加权 CVT (来自 440):
     rho(x) 为人口密度, 生成器趋向人口密集区
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple


# ===========================================================================
# 1. 1D CVT Lloyd 算法 (来自 244_cvt_1d_lumping)
# ===========================================================================
def cvt_1d_lloyd(
    n_generators: int,
    n_iterations: int,
    n_samples: int,
    density_func: Callable[[List[float]], List[float]],
    init_mode: int = 1,
) -> Tuple[List[float], List[float]]:
    """1D CVT Lloyd 算法 (区间 [-1, 1]).

    Parameters
    ----------
    n_generators : int
        生成器数量 n.
    n_iterations : int
        Lloyd 迭代次数.
    n_samples : int
        用于估计 Voronoi 区域的采样点数.
    density_func : callable
        密度函数 rho(x), 输入 x 列表, 输出 rho 值列表.
    init_mode : int
        初始化方式:
            1 = 随机排序
            2 = Chebyshev 零点
            3 = 均匀交错

    Returns
    -------
    (generators, energies) : (List[float], List[float])
        最终生成器位置和各迭代的量化能量.
    """
    if n_generators <= 0:
        raise ValueError("生成器数必须为正")
    if n_samples < 2 * n_generators:
        raise ValueError("采样点数应 >= 2 * 生成器数")

    # 初始化生成器
    if init_mode == 1:
        # 随机排序 (简单确定性伪随机)
        gen = sorted([-1.0 + 2.0 * (i + 0.5) / n_generators for i in range(n_generators)])
    elif init_mode == 2:
        # Chebyshev 零点: x_k = cos((2k-1)pi / (2n))
        gen = [
            math.cos((2 * k + 1) * math.pi / (2 * n_generators))
            for k in range(n_generators)
        ]
        gen.sort()
    elif init_mode == 3:
        # 均匀交错
        gen = [-1.0 + 2.0 * (i + 0.5) / n_generators for i in range(n_generators)]
    else:
        raise ValueError(f"未知初始化模式: {init_mode}")

    # 采样点 (均匀)
    samples = [-1.0 + 2.0 * i / (n_samples - 1) for i in range(n_samples)]
    rho_vals = density_func(samples)

    energies: List[float] = []

    for _ in range(n_iterations):
        # Voronoi 分配 (1D: 相邻生成器的中点)
        boundaries = [-1.0]
        for i in range(n_generators - 1):
            boundaries.append(0.5 * (gen[i] + gen[i + 1]))
        boundaries.append(1.0)

        # 计算质心 (lumping 加权)
        new_gen = []
        total_energy = 0.0
        for i in range(n_generators):
            lo, hi = boundaries[i], boundaries[i + 1]
            # 找到 [lo, hi] 内的采样点
            weight_sum = 0.0
            weighted_sum = 0.0
            energy_i = 0.0
            for x, rho in zip(samples, rho_vals):
                if lo <= x < hi or (i == n_generators - 1 and x == hi):
                    weight_sum += rho
                    weighted_sum += x * rho
                    energy_i += (x - gen[i]) ** 2 * rho
            if weight_sum > 1e-15:
                new_gen.append(weighted_sum / weight_sum)
            else:
                new_gen.append(gen[i])
            total_energy += energy_i

        gen = new_gen
        energies.append(total_energy / n_samples)

    return gen, energies


# ===========================================================================
# 2. 2D CVT Lloyd 算法 (来自 440_florida_cvt_pop 思想)
# ===========================================================================
def cvt_2d_lloyd(
    n_generators: int,
    n_iterations: int,
    n_samples: int,
    density_func: Callable[[List[float], List[float]], List[float]],
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    seed: int = 12345,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """2D CVT Lloyd 算法 (矩形域).

    Parameters
    ----------
    n_generators : int
        生成器数量 n.
    n_iterations : int
        Lloyd 迭代次数.
    n_samples : int
        采样点数 (用于估计 Voronoi 区域).
    density_func : callable
        密度函数 rho(x, y).
    domain : (xmin, xmax, ymin, ymax)
        矩形域.
    seed : int
        伪随机种子.

    Returns
    -------
    (generators, energies) : (List[(float, float)], List[float])
    """
    if n_generators <= 0:
        raise ValueError("生成器数必须为正")

    xmin, xmax, ymin, ymax = domain

    # 确定性伪随机初始化 (简单 LCG)
    state = seed
    gen: List[Tuple[float, float]] = []
    for _ in range(n_generators):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        x = xmin + (xmax - xmin) * (state / 0x7FFFFFFF)
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        y = ymin + (ymax - ymin) * (state / 0x7FFFFFFF)
        gen.append((x, y))

    # 采样点 (网格)
    n_side = int(math.sqrt(n_samples))
    xs = [xmin + (xmax - xmin) * i / (n_side - 1) for i in range(n_side)]
    ys = [ymin + (ymax - ymin) * j / (n_side - 1) for j in range(n_side)]
    samples_x = []
    samples_y = []
    for x in xs:
        for y in ys:
            samples_x.append(x)
            samples_y.append(y)
    rho_vals = density_func(samples_x, samples_y)

    energies: List[float] = []

    for _ in range(n_iterations):
        # Voronoi 分配 (对每个采样点, 找最近生成器)
        assignments = []
        for sx, sy in zip(samples_x, samples_y):
            min_dist = math.inf
            min_idx = 0
            for k, (gx, gy) in enumerate(gen):
                d2 = (sx - gx) ** 2 + (sy - gy) ** 2
                if d2 < min_dist:
                    min_dist = d2
                    min_idx = k
            assignments.append(min_idx)

        # 计算质心
        new_gen = []
        total_energy = 0.0
        for k in range(n_generators):
            weight_sum = 0.0
            wx_sum = 0.0
            wy_sum = 0.0
            energy_k = 0.0
            for i, (sx, sy) in enumerate(zip(samples_x, samples_y)):
                if assignments[i] == k:
                    rho = rho_vals[i]
                    weight_sum += rho
                    wx_sum += sx * rho
                    wy_sum += sy * rho
                    energy_k += ((sx - gen[k][0]) ** 2 + (sy - gen[k][1]) ** 2) * rho
            if weight_sum > 1e-15:
                new_gen.append((wx_sum / weight_sum, wy_sum / weight_sum))
            else:
                new_gen.append(gen[k])
            total_energy += energy_k

        gen = new_gen
        energies.append(total_energy / len(samples_x))

    return gen, energies


# ===========================================================================
# 3. 辅助: 常用密度函数
# ===========================================================================
def uniform_density_1d(x_list: List[float]) -> List[float]:
    """1D 均匀密度 rho(x) = 1."""
    return [1.0] * len(x_list)


def gaussian_density_1d(x_list: List[float], mu: float = 0.0, sigma: float = 0.3) -> List[float]:
    """1D Gaussian 密度 rho(x) = exp(-(x-mu)^2 / (2 sigma^2))."""
    return [math.exp(-0.5 * ((x - mu) / sigma) ** 2) for x in x_list]


def bimodal_density_1d(x_list: List[float]) -> List[float]:
    """1D 双峰密度."""
    return [
        math.exp(-10 * (x - 0.3) ** 2) + math.exp(-10 * (x + 0.3) ** 2)
        for x in x_list
    ]


def gaussian_density_2d(
    xs: List[float], ys: List[float], cx: float = 0.5, cy: float = 0.5, sigma: float = 0.2
) -> List[float]:
    """2D Gaussian 密度."""
    return [
        math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        for x, y in zip(xs, ys)
    ]


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """CVT 模块自检."""
    print("[CVT 1D] Lloyd 算法 (均匀密度):")
    gen, energies = cvt_1d_lloyd(
        n_generators=5,
        n_iterations=20,
        n_samples=100,
        density_func=uniform_density_1d,
        init_mode=1,
    )
    print(f"  生成器: {[f'{g:.3f}' for g in gen]}")
    print(f"  最终能量: {energies[-1]:.6f}")

    print("[CVT 1D] Lloyd 算法 (Gaussian 密度):")
    gen2, energies2 = cvt_1d_lloyd(
        n_generators=5,
        n_iterations=20,
        n_samples=100,
        density_func=gaussian_density_1d,
        init_mode=2,
    )
    print(f"  生成器: {[f'{g:.3f}' for g in gen2]}")
    print(f"  最终能量: {energies2[-1]:.6f}")

    print("[CVT 2D] Lloyd 算法 (Gaussian 密度):")
    gen3, energies3 = cvt_2d_lloyd(
        n_generators=4,
        n_iterations=10,
        n_samples=400,
        density_func=gaussian_density_2d,
        domain=(0.0, 1.0, 0.0, 1.0),
    )
    print(f"  生成器: {[(f'{g[0]:.3f}', f'{g[1]:.3f}') for g in gen3]}")
    print(f"  最终能量: {energies3[-1]:.6f}")


if __name__ == "__main__":
    self_check()
