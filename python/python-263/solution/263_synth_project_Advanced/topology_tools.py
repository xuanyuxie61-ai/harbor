# -*- coding: utf-8 -*-
"""
topology_tools.py
-----------------
磁场拓扑工具: 场线距离度量 (113_box_distance) + 加热率量化 (583_image_quantization).

物理动机
--------
1) 场线连接性距离 (field line connectivity distance):
   两个光球足点之间的磁拓扑距离由场线分离率定义.
   设两条场线起始于位置 x_1, x_2 (光球面上相距 delta_x),
   在日冕中演化到高度 h 后分离距离 delta_r(h). 拓扑距离:
       d_topo(x_1, x_2) = lim_{h -> H} |r_1(h) - r_2(h)| / |x_1 - x_2|

   此量衡量磁拓扑的 "敏感度" (类似 Lyapunov 指数), 高 d_topo 区域
   对应准分隔层 (QSL, 磁重联优先位置).

2) 加热率量化 (heating rate quantization):
   将连续的加热率场 Q(s) 量化为有限个离散水平 (类似灰度图像量化),
   用于识别加热 "平台" 与 "梯度带":
       Q_quant(s) = sum_{k=1}^K q_k * 1_{s in R_k}
   其中 R_k 为 Voronoi 区域 (在 Q 值空间).

   此量化用于简化加热模型, 并识别主导加热结构.
"""
from __future__ import annotations
import numpy as np


# ============================================================
# 磁场拓扑距离 (来自 113_box_distance)
# ============================================================
def field_line_trace(b_field_func, x0: np.ndarray, y0: np.ndarray,
                     z0: float, z_end: float, dz: float) -> np.ndarray:
    """沿磁场方向积分场线: dr/dz = B_perp / B_z.

    Parameters
    ----------
    b_field_func : callable (x, y, z) -> (Bx, By, Bz)
    x0, y0 : 起始位置 (光球面)
    z0, z_end : 积分高度范围
    dz : 步长

    Returns
    -------
    path : ndarray (n_step, 3), 场线路径 (x, y, z).
    """
    n_step = int((z_end - z0) / dz)
    path = np.zeros((n_step + 1, 3))
    path[0] = [x0, y0, z0]

    for k in range(n_step):
        x, y, z = path[k]
        bx, by, bz = b_field_func(x, y, z)
        if abs(bz) < 1.0e-12:
            path[k + 1] = [x, y, z + dz]
            continue
        dx = bx / bz * dz
        dy = by / bz * dz
        path[k + 1] = [x + dx, y + dy, z + dz]
    return path


def topological_separation(path1: np.ndarray,
                           path2: np.ndarray) -> float:
    """两条场线在顶端的分离距离."""
    return float(np.linalg.norm(path1[-1, :2] - path2[-1, :2]))


def connectivity_matrix(x_grid: np.ndarray, y_grid: np.ndarray,
                        b_field_func, z0: float, z_end: float,
                        dz: float) -> np.ndarray:
    """计算光球网格点间的拓扑连接矩阵.

    D[i, j] = 场线从 (x_i, y_i) 与 (x_j, y_j) 出发在 z_end 处的距离.
    """
    n = x_grid.size
    D = np.zeros((n, n))
    paths = []
    for i in range(n):
        p = field_line_trace(b_field_func, x_grid[i], y_grid[i],
                             z0, z_end, dz)
        paths.append(p)
    for i in range(n):
        for j in range(i + 1, n):
            d = topological_separation(paths[i], paths[j])
            D[i, j] = d
            D[j, i] = d
    return D


# ============================================================
# 加热率量化 (来自 583_image_quantization)
# ============================================================
def quantize_heating_rate(q_field: np.ndarray,
                          n_levels: int = 8) -> tuple:
    """将连续加热率场量化为 n_levels 个离散水平.

    使用 Lloyd 算法 (CVT in 1D value space):
    1) 初始化 n_levels 个量化中心 {q_k}.
    2) 将每个 Q(s) 归属到最近的 q_k.
    3) 更新 q_k = mean of Q in region k.
    4) 迭代至收敛.

    Returns
    -------
    q_quant : ndarray, 量化后的加热率场
    centers : ndarray (n_levels,), 量化中心
    labels : ndarray, 每个点的标签
    """
    q_flat = q_field.flatten()
    q_min, q_max = q_flat.min(), q_flat.max()
    if q_max - q_min < 1.0e-12:
        return q_field.copy(), np.array([q_min]), np.zeros_like(q_flat, dtype=int)

    # 初始化中心
    centers = np.linspace(q_min, q_max, n_levels)
    labels = np.zeros(q_flat.size, dtype=int)

    for _iter in range(50):
        # 归属
        dists = np.abs(q_flat[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        # 更新中心
        new_centers = np.zeros(n_levels)
        for k in range(n_levels):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = q_flat[mask].mean()
            else:
                new_centers[k] = centers[k]
        if np.max(np.abs(new_centers - centers)) < 1.0e-8:
            centers = new_centers
            break
        centers = new_centers

    q_quant = centers[labels].reshape(q_field.shape)
    return q_quant, centers, labels


def heating_gradient_identification(q_quant: np.ndarray,
                                    z_grid: np.ndarray) -> np.ndarray:
    """识别加热率的强梯度区域 (加热前沿).

    grad_Q = dQ_quant / dz, 高 |grad_Q| 区域为加热前沿.
    """
    grad_q = np.gradient(q_quant, z_grid)
    return grad_q


def dominant_heating_region(q_quant: np.ndarray,
                            z_grid: np.ndarray,
                            threshold_frac: float = 0.5) -> dict:
    """识别主导加热区域: Q > threshold_frac * Q_max 的区间."""
    q_max = q_quant.max()
    mask = q_quant > threshold_frac * q_max
    if not np.any(mask):
        return dict(z_start=0.0, z_end=0.0, fraction=0.0)
    z_dom = z_grid[mask]
    return dict(
        z_start=float(z_dom.min()),
        z_end=float(z_dom.max()),
        fraction=float(mask.mean()),
        peak_q=float(q_max),
    )
