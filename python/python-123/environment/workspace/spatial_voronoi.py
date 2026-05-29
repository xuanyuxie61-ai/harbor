"""
spatial_voronoi.py

肿瘤空间生长与 Voronoi 镶嵌模块

本模块融合以下种子项目的核心算法：
  - 256_cvt_corn_movie: 圆盘约束 CVT 与径向生长
  - 850_partition_greedy: 贪婪分区算法

科学背景：
  肿瘤内部的细胞竞争空间可用 Voronoi 镶嵌（Voronoi Tessellation）建模。
  每个细胞（或细胞簇）占据一个 Voronoi 区域，区域的大小反映该细胞的
  竞争优势。CVT（Centroidal Voronoi Tessellation）通过 Lloyd 算法
  将生成子迭代移动到其 Voronoi 区域的质心，实现空间分布的能量最小化：

    F(P) = sum_{i=1}^{N} int_{V_i} rho(x) * |x - p_i|^2 dx

  其中 V_i 是第 i 个 Voronoi 区域，p_i 是生成子，rho(x) 是密度函数。

  肿瘤径向生长（Radial Growth）遵循面积守恒：
    当新增 n_bud 个边界点时，半径扩展因子 factor = n_bud / n_boundary，
    所有点坐标同步缩放，保持弧长/点数比恒定。

  代谢异质性分区采用贪婪算法：
    给定细胞代谢标记集合 W = {w_1, ..., w_N}，
    将其划分为两个子集 S0 和 S1，使 |sum(S0) - sum(S1)| 最小。
"""

import numpy as np
from typing import Tuple


def disk_sample_uniform(num_samples: int, radius: float = 1.0,
                        seed: int = 42) -> np.ndarray:
    """
    在圆盘内均匀采样点。

    极坐标变换:
        r = R * sqrt(u),  theta = 2*pi*v
      其中 u, v ~ U(0,1)。
      面积元: dA = r dr dtheta => 需要 r ~ sqrt(u) 保证均匀。
    """
    if num_samples < 0:
        raise ValueError("disk_sample_uniform: num_samples >= 0")
    if radius <= 0:
        raise ValueError("disk_sample_uniform: radius > 0")

    rng = np.random.default_rng(seed=seed)
    u = rng.random(num_samples)
    v = rng.random(num_samples)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def find_closest(sample_points: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """
    将每个采样点分配到最近的生成子。

    参数:
        sample_points: (S, 2)
        generators:    (G, 2)

    返回:
        nearest: (S,) 整数数组，nearest[i] 为第 i 个采样点最近的生成子索引
    """
    S = sample_points.shape[0]
    G = generators.shape[0]
    nearest = np.zeros(S, dtype=int)

    for s in range(S):
        min_dist = np.inf
        min_idx = 0
        for g in range(G):
            dx = sample_points[s, 0] - generators[g, 0]
            dy = sample_points[s, 1] - generators[g, 1]
            d2 = dx * dx + dy * dy
            if d2 < min_dist:
                min_dist = d2
                min_idx = g
        nearest[s] = min_idx

    return nearest


def cvt_disk_iterate(
    radius: float, num_samples: int, generators: np.ndarray,
    p_type: np.ndarray, num_iterations: int = 30
) -> np.ndarray:
    """
    在圆盘约束下执行 CVT Lloyd 迭代。

    算法:
      for it = 1 .. num_iterations:
        1. 在圆盘内均匀采样 num_samples 个点
        2. 将每个采样点分配到最近的生成子
        3. 更新生成子位置为其 Voronoi 区域的质心
        4. 将边界生成子投影回圆周

    参数:
        radius: 圆盘半径
        num_samples: 采样点数
        generators: (N, 2) 当前生成子坐标
        p_type: (N,) 整数，1=边界约束，2=内部约束
        num_iterations: Lloyd 迭代次数

    返回:
        generators: 更新后的生成子坐标
    """
    N = generators.shape[0]
    if p_type.shape[0] != N:
        raise ValueError("cvt_disk_iterate: p_type 长度与生成子数不匹配")
    if num_samples < N:
        raise ValueError("cvt_disk_iterate: num_samples >= 生成子数")

    for _ in range(num_iterations):
        sample_points = disk_sample_uniform(num_samples, radius)
        nearest = find_closest(sample_points, generators)

        v_xy = generators.copy()
        counts = np.ones(N)

        for s in range(num_samples):
            g = nearest[s]
            v_xy[g, 0] += sample_points[s, 0]
            v_xy[g, 1] += sample_points[s, 1]
            counts[g] += 1.0

        # 质心 = 累加和 / 计数
        for g in range(N):
            if counts[g] > 1e-12:
                v_xy[g, 0] /= counts[g]
                v_xy[g, 1] /= counts[g]

        # 边界点投影回圆周
        for g in range(N):
            if p_type[g] == 1:  # 边界约束
                r2 = v_xy[g, 0] ** 2 + v_xy[g, 1] ** 2
                r = np.sqrt(max(r2, 1e-15))
                v_xy[g, 0] = radius * v_xy[g, 0] / r
                v_xy[g, 1] = radius * v_xy[g, 1] / r
            else:
                # 内部点不得超出圆盘
                r2 = v_xy[g, 0] ** 2 + v_xy[g, 1] ** 2
                if r2 > radius ** 2:
                    r = np.sqrt(r2)
                    v_xy[g, 0] = radius * v_xy[g, 0] / r * 0.999
                    v_xy[g, 1] = radius * v_xy[g, 1] / r * 0.999

        generators = v_xy

    return generators


def initialize_tumor_generators(
    n_boundary: int, n_interior: int, radius: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    初始化肿瘤 Voronoi 生成子。

    参数:
        n_boundary: 边界生成子数
        n_interior: 内部生成子数
        radius: 肿瘤半径

    返回:
        generators: (n_boundary+n_interior, 2)
        p_type:     (n_boundary+n_interior,)  1=boundary, 2=interior
    """
    if n_boundary < 3:
        raise ValueError("initialize_tumor_generators: n_boundary >= 3")
    if n_interior < 0:
        raise ValueError("initialize_tumor_generators: n_interior >= 0")
    if radius <= 0:
        raise ValueError("initialize_tumor_generators: radius > 0")

    np_total = n_boundary + n_interior
    rng = np.random.default_rng(seed=seed)

    # 内部点均匀采样
    interior = disk_sample_uniform(n_interior, radius, seed=seed)

    # 边界点在圆周上均匀分布
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    boundary = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta)
    ])

    generators = np.vstack([boundary, interior])
    p_type = np.zeros(np_total, dtype=int)
    p_type[:n_boundary] = 1
    p_type[n_boundary:] = 2

    return generators, p_type


def radial_growth_expand(
    generators: np.ndarray, p_type: np.ndarray,
    radius: float, new_boundary_count: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    模拟肿瘤径向扩展。

    扩展规则:
      新的边界点数 n_new = n_old + new_boundary_count
      扩展因子 factor = n_new / n_old
      新半径 R_new = factor * R_old
      所有点坐标乘以 factor

    参数:
        generators: 当前生成子
        p_type: 类型标记
        radius: 当前半径
        new_boundary_count: 新增的边界细胞数

    返回:
        new_radius, new_generators, new_p_type
    """
    n_old_boundary = int(np.sum(p_type == 1))
    if n_old_boundary <= 0:
        n_old_boundary = 1

    n_new_boundary = n_old_boundary + new_boundary_count
    factor = n_new_boundary / n_old_boundary

    new_radius = radius * factor
    new_generators = generators.copy() * factor

    # 更新 p_type: 所有变为正
    new_p_type = np.abs(p_type)

    return new_radius, new_generators, new_p_type


def add_boundary_generators(
    generators: np.ndarray, p_type: np.ndarray,
    radius: float, n_add: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在肿瘤边界上随机添加新的生成子（模拟肿瘤边缘细胞增殖）。

    参数:
        generators: 当前生成子
        p_type: 当前类型
        radius: 当前半径
        n_add: 新增边界点数

    返回:
        new_generators, new_p_type
    """
    rng = np.random.default_rng(seed=seed)
    theta_new = rng.uniform(0.0, 2.0 * np.pi, n_add)
    new_points = np.column_stack([
        radius * np.cos(theta_new),
        radius * np.sin(theta_new)
    ])

    new_generators = np.vstack([generators, new_points])
    new_p_type = np.concatenate([
        p_type,
        np.ones(n_add, dtype=int)  # 新增为边界点
    ])

    return new_generators, new_p_type


def partition_metabolic_activity(
    weights: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    使用贪婪算法对肿瘤细胞的代谢活性进行二分分区。

    算法（来自 partition_greedy）：
      1. 将权重按降序排序
      2. 依次将每个权重放入当前和较小的子集
      3. 返回分区标记和差异

    数学目标:
        minimize |sum(S0) - sum(S1)|

    参数:
        weights: (N,) 非负数组，细胞代谢标记权重

    返回:
        labels: (N,) 0/1 数组
        discrepancy: |sum(S0) - sum(S1)|
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("partition_metabolic_activity: 权重必须非负")

    n = weights.shape[0]
    if n == 0:
        return np.array([], dtype=int), 0.0

    # 降序索引
    idx_desc = np.argsort(-weights)
    labels = np.zeros(n, dtype=int)

    s0_sum = 0.0
    s1_sum = 0.0
    for i in range(n):
        j = idx_desc[i]
        if s0_sum < s1_sum:
            labels[j] = 0
            s0_sum += weights[j]
        else:
            labels[j] = 1
            s1_sum += weights[j]

    discrepancy = abs(s0_sum - s1_sum)
    return labels, float(discrepancy)


def compute_voronoi_energy(generators: np.ndarray,
                           sample_points: np.ndarray) -> float:
    """
    计算 CVT 能量泛函：

        F = sum_i sum_{x in V_i} |x - p_i|^2

    该能量衡量生成子与 Voronoi 区域质心的匹配程度。
    """
    nearest = find_closest(sample_points, generators)
    energy = 0.0
    for s in range(sample_points.shape[0]):
        g = nearest[s]
        dx = sample_points[s, 0] - generators[g, 0]
        dy = sample_points[s, 1] - generators[g, 1]
        energy += dx * dx + dy * dy
    return float(energy)
