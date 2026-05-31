"""
Hilbert 空间填充曲线模块
融合来源: 535_hilbert_curve (d2xy, rot, xy2d)

功能:
- 将一维 Hilbert 坐标映射到二维笛卡尔坐标
- 对网格节点进行 Hilbert 曲线重排序，提升稀疏矩阵内存局部性
- 应用于多铁性材料有限元网格的节点编号优化
"""

import numpy as np
from typing import List, Tuple


def rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    """
    对给定象限进行坐标旋转/反射。
    直接源自 hilbert_curve 中 rot.m 的算法逻辑。
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # 交换 x, y
        x, y = y, x
    return x, y


def d2xy(m: int, d: int) -> Tuple[int, int]:
    """
    将一维 Hilbert 坐标 D 转换为二维笛卡尔坐标 (X, Y)。
    网格大小 N = 2^m，坐标范围 [0, N-1]。
    直接源自 hilbert_curve 中 d2xy.m。
    """
    if m <= 0:
        raise ValueError("Hilbert 曲线阶数 m 必须为正整数")
    n = 1 << m
    if d < 0 or d >= n * n:
        raise ValueError(f"Hilbert 坐标 d 必须在 [0, {n*n}) 范围内")

    x, y = 0, 0
    t = d
    s = 1
    while s < n:
        rx = (t // 2) & 1
        if rx == 0:
            ry = t & 1
        else:
            ry = (t ^ rx) & 1
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def xy2d(m: int, x: int, y: int) -> int:
    """
    将二维笛卡尔坐标 (X, Y) 转换为一维 Hilbert 坐标 D。
    源自 hilbert_curve 中 xy2d.m。
    """
    if m <= 0:
        raise ValueError("Hilbert 曲线阶数 m 必须为正整数")
    n = 1 << m
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"坐标 ({x},{y}) 越界 [0, {n})")

    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s //= 2
    return d


def hilbert_sort_points(points: np.ndarray, m: int = 6) -> np.ndarray:
    """
    对二维点集按照 Hilbert 曲线顺序进行重排序。

    参数:
        points: (N, 2) 数组，坐标需在 [0, 1] 范围内
        m: Hilbert 曲线阶数，决定分辨率 N=2^m

    返回:
        order: 重排序的索引数组
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必须是 (N, 2) 数组")
    n = 1 << m
    # 将 [0,1] 映射到 [0, n-1]
    scaled = np.clip(points, 0.0, 1.0) * (n - 1)
    ix = scaled[:, 0].astype(int)
    iy = scaled[:, 1].astype(int)
    hilbert_ids = np.array([xy2d(m, ix[i], iy[i]) for i in range(len(points))])
    order = np.argsort(hilbert_ids)
    return order


def apply_hilbert_reordering(node_xy: np.ndarray, element_node: np.ndarray,
                             m: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对有限元网格节点应用 Hilbert 曲线重排序。
    返回重排序后的节点坐标、元素连接表、以及旧->新的索引映射。
    """
    # 归一化坐标到 [0,1]
    xmin, xmax = node_xy[:, 0].min(), node_xy[:, 0].max()
    ymin, ymax = node_xy[:, 1].min(), node_xy[:, 1].max()
    eps = 1e-12
    norm_xy = np.zeros_like(node_xy)
    norm_xy[:, 0] = (node_xy[:, 0] - xmin) / max(xmax - xmin, eps)
    norm_xy[:, 1] = (node_xy[:, 1] - ymin) / max(ymax - ymin, eps)

    order = hilbert_sort_points(norm_xy, m=m)
    # 旧索引 -> 新索引的映射
    old_to_new = np.empty(len(order), dtype=int)
    old_to_new[order] = np.arange(len(order), dtype=int)

    new_node_xy = node_xy[order]
    new_element_node = old_to_new[element_node]
    return new_node_xy, new_element_node, old_to_new
