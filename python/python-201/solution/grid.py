"""
grid.py — 物理空间网格与随机空间配置
======================================
生成二维物理空间网格 (Cahn-Hilliard 计算域)
并管理随机空间求积点配置。

映射种子项目:
  - 680_line_grid: 一维网格 → 张量积构造二维网格
  - 932_pyramid_grid: 三维网格生成 → 随机空间各向异性网格
  - 382_fem_to_xml: 有限元网格 I/O → 计算域离散化
"""

import numpy as np
from typing import Tuple, Optional
from config import CahnHilliardConfig, SparseGridConfig, MeasureConfig
from measure import gauss_quadrature, tensor_product_quadrature
from utils import line_grid


# ============================================================
#  二维物理空间网格 (映射 680_line_grid)
# ============================================================
def create_physical_mesh(config: CahnHilliardConfig
                         ) -> Tuple[np.ndarray, np.ndarray,
                                    float, float]:
    """
    生成二维矩形计算域上的均匀网格。

    映射 680_line_grid: 使用一维网格生成器的 centering=2
    (不含端点, 用于周期性边界) 构造二维张量积网格。

    参数:
        config: Cahn-Hilliard 配置

    返回:
        X: shape (ny, nx), x 坐标网格
        Y: shape (ny, nx), y 坐标网格
        dx: x 方向网格间距
        dy: y 方向网格间距
    """
    nx, ny = config.nx, config.ny
    Lx, Ly = config.Lx, config.Ly

    if config.bc_type == "periodic":
        # 周期性边界: 不含端点
        x = line_grid(nx, 0.0, Lx, centering=3)  # [0, Lx)
        y = line_grid(ny, 0.0, Ly, centering=3)
    else:
        # 无通量边界: 含端点
        x = line_grid(nx, 0.0, Lx, centering=1)
        y = line_grid(ny, 0.0, Ly, centering=1)

    dx = Lx / nx if config.bc_type == "periodic" else Lx / max(nx - 1, 1)
    dy = Ly / ny if config.bc_type == "periodic" else Ly / max(ny - 1, 1)

    X, Y = np.meshgrid(x, y)

    return X, Y, dx, dy


# ============================================================
#  有限元网格信息 (映射 382_fem_to_xml)
# ============================================================
def generate_mesh_info(nx: int, ny: int, Lx: float, Ly: float
                       ) -> dict:
    """
    生成有限元网格的元数据。

    映射 382_fem_to_xml: 生成网格节点和单元信息,
    用于随机 Galerkin 有限元离散化。

    参数:
        nx, ny: 各方向网格数
        Lx, Ly: 域尺寸

    返回:
        mesh_info: 网格信息字典
    """
    n_nodes = (nx + 1) * (ny + 1)
    n_elements = nx * ny * 2  # 三角形单元
    dx = Lx / nx
    dy = Ly / ny
    area = dx * dy / 2.0

    mesh_info = {
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "dx": dx,
        "dy": dy,
        "element_area": area,
        "total_area": Lx * Ly,
        "element_type": "triangle",
        "spatial_dim": 2,
    }
    return mesh_info


# ============================================================
#  随机空间各向异性网格 (映射 932_pyramid_grid)
# ============================================================
def anisotropic_stochastic_grid(measures: list,
                                levels: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造各向异性随机空间网格。

    映射 932_pyramid_grid: 将三维金字塔网格生成推广为
    多维随机空间中的各向异性求积网格。

    在每个维度 k, 使用 levels[k] 个求积点。
    总体结构类似金字塔: 低维方向用更少的点。

    参数:
        measures: 各维度的测度列表
        levels:   shape (d,), 各维度的求积点数

    返回:
        nodes:   shape (N_total, d), 求积节点
        weights: shape (N_total,), 求积权重
    """
    d = len(measures)
    levels = np.asarray(levels, dtype=int)

    nodes_1d = []
    weights_1d = []
    for k in range(d):
        nd, wt = gauss_quadrature(measures[k], levels[k])
        nodes_1d.append(nd)
        weights_1d.append(wt)

    # 张量积
    grids = np.meshgrid(*nodes_1d, indexing='ij')
    nodes = np.column_stack([g.ravel() for g in grids])

    weight_grids = np.meshgrid(*weights_1d, indexing='ij')
    weights = np.ones(weight_grids[0].size)
    for wg in weight_grids:
        weights *= wg.ravel()

    return nodes, weights


# ============================================================
#  Smolyak 稀疏网格 (映射 932_pyramid_grid 的层级结构)
# ============================================================
def smolyak_sparse_grid(measures: list, level: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 Smolyak 稀疏网格。

    Smolyak 公式:
      A(q, d) = Σ_{q-|i|_1 <= q} (-1)^{q-|i|_1}
                C(d-1, q-|i|_1) (Q^{i_1} ⊗ ... ⊗ Q^{i_d})

    其中 q = level + d, i = (i_1, ..., i_d), i_k >= 1。

    映射 932_pyramid_grid: 使用金字塔式的层级结构
    组织稀疏网格的组合。

    参数:
        measures: 各维度的测度
        level:    稀疏网格层级 (>= 0)

    返回:
        nodes:   shape (N, d), 求积节点
        weights: shape (N,), 求积权重
    """
    d = len(measures)
    if d == 0:
        return np.array([[]]), np.array([1.0])

    q = level + d

    # 收集所有满足 |i|_1 = q-d+k, k=0,...,min(level, d-1) 的多索引
    # 即 q - level <= |i|_1 <= q, 且 i_k >= 1
    from itertools import product as iterproduct

    all_nodes = []
    all_weights = []
    all_signs = []

    for k in range(min(level + 1, d + 1)):
        target_sum = q - k
        sign = (-1) ** k
        from math import comb
        coeff = comb(d - 1, k) if k <= d - 1 else 0

        if coeff == 0:
            continue

        # 枚举满足 i_1+...+i_d = target_sum, i_j >= 1 的多索引
        multi_indices = _compositions(target_sum, d)

        for mi in multi_indices:
            # 对每个多索引, 构造张量积求积规则
            levels_arr = np.array(mi)
            nd, wt = anisotropic_stochastic_grid(measures, levels_arr)
            wt_scaled = wt * sign * coeff

            all_nodes.append(nd)
            all_weights.append(wt_scaled)

    if len(all_nodes) == 0:
        return np.zeros((0, d)), np.array([])

    nodes = np.vstack(all_nodes)
    weights = np.concatenate(all_weights)

    # 合并重复节点 (简化: 使用唯一化)
    nodes, weights = _merge_sparse_grid_points(nodes, weights)

    return nodes, weights


def _compositions(n: int, k: int) -> list:
    """
    生成 n 的 k-组合: {i ∈ N^k : i_1+...+i_k = n, i_j >= 1}
    """
    if k == 1:
        return [(n,)] if n >= 1 else []

    result = []
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            result.append((first,) + rest)
    return result


def _merge_sparse_grid_points(nodes: np.ndarray,
                              weights: np.ndarray,
                              tol: float = 1e-12
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    合并稀疏网格中的重复节点, 累加其权重。
    """
    if len(nodes) == 0:
        return nodes, weights

    # 四舍五入到 tol 精度以识别重复
    rounded = np.round(nodes / tol) * tol
    _, unique_idx, inverse = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True)

    merged_nodes = nodes[unique_idx]
    merged_weights = np.zeros(len(unique_idx))
    for i, idx in enumerate(inverse):
        merged_weights[idx] += weights[i]

    # 移除权重接近零的点
    mask = np.abs(merged_weights) > 1e-15
    return merged_nodes[mask], merged_weights[mask]


# ============================================================
#  初始条件生成
# ============================================================
def generate_initial_condition(config: CahnHilliardConfig,
                               X: np.ndarray, Y: np.ndarray
                               ) -> np.ndarray:
    """
    生成 Cahn-Hilliard 初始条件。

    c(x, y, 0) = c0 + ε * η(x, y)

    其中:
      c0: 平均浓度 (通常为 0, 对应等组分混合物)
      ε:  扰动幅度
      η:  均匀分布的随机噪声 ∈ [-1, 1]

    使用确定性种子确保可重复性。

    参数:
        config: CH 配置
        X, Y:   空间网格

    返回:
        c0_field: shape (ny, nx), 初始浓度场
    """
    rng = np.random.RandomState(config.random_seed)
    noise = rng.uniform(-1, 1, X.shape)

    c0_field = config.c0_mean + config.c0_noise_std * noise

    return c0_field
