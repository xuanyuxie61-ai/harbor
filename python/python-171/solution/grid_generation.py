# -*- coding: utf-8 -*-
"""
grid_generation.py
==================
一维与高维自适应网格生成工具。

融合种子项目：
- 680_line_grid : 一维线段等距网格
- 244_cvt_1d_lumping : CVT (Centroidal Voronoi Tessellation) 自适应网格
- 578_image_double : 网格加倍（加密）思想
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# 一维线段网格（680_line_grid）
# ---------------------------------------------------------------------------

def line_grid(n, a, b, c=1):
    """
    在区间 [a, b] 上生成 n 个一维网格点，支持 5 种边界对齐方式。

    参数
    ----
    n : int
        网格点数，n >= 1。
    a, b : float
        区间端点。
    c : int, 1~5
        对齐方式：
        1: 包含两端点，均匀分布
        2: 不包含端点，内部 n 个均匀点
        3: 包含 a，不包含 b
        4: 不包含 a，包含 b
        5: 半偏移均匀点（中点型）

    返回
    ----
    x : ndarray, shape (n,)
        网格点坐标。
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if c < 1 or c > 5:
        raise ValueError("c must be in [1,5].")
    x = np.zeros(n, dtype=float)
    for j in range(1, n + 1):
        if c == 1:
            if n == 1:
                x[j - 1] = 0.5 * (a + b)
            else:
                x[j - 1] = ((n - j) * a + (j - 1) * b) / (n - 1)
        elif c == 2:
            x[j - 1] = ((n - j + 1) * a + j * b) / (n + 1)
        elif c == 3:
            x[j - 1] = ((n - j + 1) * a + (j - 1) * b) / n
        elif c == 4:
            x[j - 1] = ((n - j) * a + j * b) / n
        elif c == 5:
            x[j - 1] = ((2 * n - 2 * j + 1) * a + (2 * j - 1) * b) / (2 * n)
    return x


# ---------------------------------------------------------------------------
# CVT (Centroidal Voronoi Tessellation) 一维 Lloyd 算法（244_cvt_1d_lumping）
# ---------------------------------------------------------------------------

def cvt_1d_lloyd(n, it_num, s_num, density_func, init_type=1, seed=None):
    """
    一维 Lloyd 算法生成 CVT 网格点，带 lumping 加权。

    算法：
      给定密度函数 ρ(x)（本函数接收 μ(x)，内部取 ρ = μ^3），
      在 [-1,1] 上迭代计算生成点 g_j，使其为对应 Voronoi 区域的加权质心：
          g_j^{new} = Σ_{k∈V_j} ρ(s_k) s_k / Σ_{k∈V_j} ρ(s_k)

    参数
    ----
    n : int
        生成点数量。
    it_num : int
        Lloyd 迭代次数。
    s_num : int
        采样点数（均匀采样）。
    density_func : callable
        密度函数 μ(x)，接受 ndarray 返回 ndarray。
    init_type : int, 1/2/3
        1: 随机排序；2: Chebyshev 零点；3: 均匀交错。
    seed : int, optional

    返回
    ----
    g : ndarray, shape (n,)
        CVT 生成点。
    energy_history : ndarray, shape (it_num,)
        每次迭代的能量（二阶矩和）。
    motion_history : ndarray, shape (it_num,)
        每次迭代的平均生成点位移。
    """
    rng = np.random.default_rng(seed)
    # 初始化
    if init_type == 1:
        g = rng.random(n) * 2.0 - 1.0
        g.sort()
    elif init_type == 2:
        g = np.cos(math.pi * (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n))
    else:
        g = np.array([((n - i) * (-1.0) + i * 1.0) / (n + 1) for i in range(1, n + 1)], dtype=float)

    # 均匀采样点
    eps = 1e-12
    s = np.linspace(-1.0 + eps, 1.0 - eps, s_num)
    mu = density_func(s)
    mu = np.clip(mu, 1e-30, 1e30)
    rho = mu ** 3

    energy_history = np.zeros(it_num, dtype=float)
    motion_history = np.zeros(it_num, dtype=float)

    for it in range(it_num):
        # 计算 Voronoi 边界（中点）
        gb = np.zeros(n + 1, dtype=float)
        gb[0] = -1.0
        for j in range(1, n):
            gb[j] = 0.5 * (g[j - 1] + g[j])
        gb[n] = 1.0

        # 用二分查找定位每个采样点所属的 Voronoi 区域
        g_new = np.zeros(n, dtype=float)
        energy = 0.0

        # 简化：遍历采样点，累加到对应区域
        region_sums = np.zeros(n, dtype=float)
        region_weights = np.zeros(n, dtype=float)

        for k in range(s_num):
            sk = s[k]
            # 找到满足 gb[j] <= sk < gb[j+1] 的 j
            # 用 np.searchsorted
            j = np.searchsorted(gb, sk, side='right') - 1
            j = max(0, min(j, n - 1))
            region_sums[j] += rho[k] * sk
            region_weights[j] += rho[k]
            energy += rho[k] * (sk - g[j]) ** 2

        for j in range(n):
            if region_weights[j] > 1e-30:
                g_new[j] = region_sums[j] / region_weights[j]
            else:
                g_new[j] = g[j]

        # 若 n 为奇数，强制中心点对称
        if n % 2 == 1:
            g_new[n // 2] = 0.0

        energy_history[it] = energy / s_num
        motion_history[it] = np.mean((g_new - g) ** 2)
        g = g_new.copy()

    return g, energy_history, motion_history


def chebyshev_zero_density(s):
    """Chebyshev 零点型密度：μ(x) = 1 / sqrt(1 - x^2)。"""
    s = np.asarray(s, dtype=float)
    return 1.0 / np.sqrt(np.maximum(1e-30, 1.0 - s ** 2))


def polynomial_density(s, alpha=2.0):
    """多项式型密度：μ(x) = (1 + |x|)^alpha。"""
    s = np.asarray(s, dtype=float)
    return (1.0 + np.abs(s)) ** alpha


# ---------------------------------------------------------------------------
# 网格加密（image_double 思想）
# ---------------------------------------------------------------------------

def mesh_refinement_1d(x):
    """
    将一维网格 x 加密一倍：在每对相邻点之间插入中点。
    类似 image_double 的像素翻倍思想。

    输入 x: ndarray, shape (n,)
    返回 x_fine: ndarray, shape (2*n - 1,)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return x.copy()
    x_fine = np.zeros(2 * n - 1, dtype=float)
    x_fine[0::2] = x
    x_fine[1::2] = 0.5 * (x[:-1] + x[1:])
    return x_fine


def multi_level_grid(n_coarse, levels, a, b, c=1):
    """
    生成多级网格序列（用于多重网格预处理）。
    从粗网格 n_coarse 开始，每一级加密一倍。
    """
    grids = []
    n = n_coarse
    for _ in range(levels):
        grids.append(line_grid(n, a, b, c))
        n = 2 * n - 1
    return grids


# ---------------------------------------------------------------------------
# 高维张量积网格
# ---------------------------------------------------------------------------

def tensor_product_grid_1d(n_each, a, b, c=1):
    """
    生成 d 维张量积网格，每一维使用 line_grid。
    返回网格坐标列表 [x1, x2, ..., xd]。
    """
    d = len(n_each)
    grids = []
    for dim in range(d):
        grids.append(line_grid(n_each[dim], a[dim], b[dim], c))
    return grids
