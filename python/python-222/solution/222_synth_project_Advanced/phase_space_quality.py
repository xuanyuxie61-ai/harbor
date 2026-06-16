# -*- coding: utf-8 -*-
"""
phase_space_quality.py
======================

相空间采样网格质量评估。

融合种子项目:
    - 958_quality: 网格质量度量 (alpha, beta, gamma, etc.)

物理背景:
    Parton shower 的 Monte Carlo 积分需要在 (x, Q^2, z, phi) 相空间中采样。
    采样网格的质量直接影响积分精度与收敛速度。

    借鉴 958_quality 的三角网格质量度量, 我们定义相空间
    四边体单元的质量指标:
    - alpha: 最小角 / 60度 (越接近 1 越好)
    - beta: 面积 / (最长边^2)
    - gamma: 内切圆半径 / 外接圆半径比
    - condition: 单元 Jacobian 条件数

    这些指标指导自适应相空间网格细化。
"""

from __future__ import annotations
import math
from typing import List, Tuple
import constants as C


# ======================================================================
# 相空间单元定义
# ======================================================================
class PhaseSpaceCell:
    """相空间四边体单元 (x, Q2, z, phi)"""
    def __init__(self, corners: List[List[float]]):
        """corners: 4 个顶点, 每个为 4D 坐标 [x, Q2, z, phi]"""
        self.corners = corners
        self.dim = len(corners[0]) if corners else 0


def cell_volume(cell: PhaseSpaceCell) -> float:
    """
    相空间单元体积 (4D 超体积的简化估计)。
    对正交网格: V = prod_i (max_i - min_i)
    """
    dim = cell.dim
    vol = 1.0
    for d in range(dim):
        vals = [c[d] for c in cell.corners]
        vol *= (max(vals) - min(vals))
    return abs(vol)


def cell_aspect_ratio(cell: PhaseSpaceCell) -> float:
    """
    相空间单元的纵横比:
        AR = max_edge / min_edge
    越接近 1 越好。
    """
    dim = cell.dim
    edges = []
    for d in range(dim):
        vals = [c[d] for c in cell.corners]
        edges.append(max(vals) - min(vals))

    min_e = min(edges) if edges else 1e-15
    max_e = max(edges) if edges else 1e-15
    if min_e < 1e-15:
        return float('inf')
    return max_e / min_e


# ======================================================================
# Alpha 质量度量 (来自 958_quality/alpha_measure.m)
# ======================================================================
def alpha_measure_triangle(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """
    三角形 alpha 质量度量 (来自 958_quality):
        alpha = min_angle / 60 degrees
    等边三角形 alpha = 1, 退化三角形 alpha = 0。
    """
    # 边长
    def dist(a, b):
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

    a = dist(p2, p3)
    b = dist(p1, p3)
    c = dist(p1, p2)

    if a < 1e-15 or b < 1e-15 or c < 1e-15:
        return 0.0

    # 用余弦定理计算角度
    cos_A = (b*b + c*c - a*a) / (2*b*c)
    cos_B = (a*a + c*c - b*b) / (2*a*c)
    cos_C = (a*a + b*b - c*c) / (2*a*b)

    # 截断避免 NaN
    cos_A = max(-1.0, min(1.0, cos_A))
    cos_B = max(-1.0, min(1.0, cos_B))
    cos_C = max(-1.0, min(1.0, cos_C))

    angles = [math.degrees(math.acos(cos_A)),
              math.degrees(math.acos(cos_B)),
              math.degrees(math.acos(cos_C))]

    min_angle = min(angles)
    return min_angle / 60.0


# ======================================================================
# Beta 质量度量 (来自 958_quality/beta_measure.m)
# ======================================================================
def beta_measure_triangle(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """
    Beta 度量 = 4 * sqrt(3) * Area / (a^2 + b^2 + c^2)
    等边三角形 = 1, 退化 = 0。
    """
    def dist(a, b):
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

    a = dist(p2, p3)
    b = dist(p1, p3)
    c = dist(p1, p2)

    # Heron 公式面积
    s = (a + b + c) / 2.0
    arg = s * (s-a) * (s-b) * (s-c)
    if arg < 0:
        return 0.0
    area = math.sqrt(arg)

    denom = a*a + b*b + c*c
    if denom < 1e-15:
        return 0.0

    return 4.0 * math.sqrt(3.0) * area / denom


# ======================================================================
# Gamma 度量 (来自 958_quality/gamma_measure.m)
# ======================================================================
def gamma_measure_triangle(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """
    Gamma 度量 = 2 * r / R
    其中 r 为内切圆半径, R 为外接圆半径。
    等边三角形 = 1。
    """
    def dist(a, b):
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

    a = dist(p2, p3)
    b = dist(p1, p3)
    c = dist(p1, p2)

    s = (a + b + c) / 2.0
    arg = s * (s-a) * (s-b) * (s-c)
    if arg < 0:
        return 0.0
    area = math.sqrt(arg)

    if area < 1e-15:
        return 0.0

    r = area / s  # 内切圆半径
    R = (a * b * c) / (4.0 * area)  # 外接圆半径

    if R < 1e-15:
        return 0.0
    return 2.0 * r / R


# ======================================================================
# 相空间网格质量评估
# ======================================================================
def phase_space_mesh_quality_2d(x_grid: List[float], q2_grid: List[float]) -> dict:
    """
    评估 (x, Q^2) 二维相空间网格的质量。
    将每个矩形单元拆成两个三角形, 评估三角质量。
    """
    alpha_min = 1.0
    beta_min = 1.0
    gamma_min = 1.0
    n_cells = 0

    for i in range(len(x_grid) - 1):
        for j in range(len(q2_grid) - 1):
            # 四个角 (映射到 2D)
            p1 = [x_grid[i], q2_grid[j]]
            p2 = [x_grid[i+1], q2_grid[j]]
            p3 = [x_grid[i+1], q2_grid[j+1]]
            p4 = [x_grid[i], q2_grid[j+1]]

            # 两个三角形
            for tri in [(p1, p2, p3), (p1, p3, p4)]:
                a = alpha_measure_triangle(*tri)
                b = beta_measure_triangle(*tri)
                g = gamma_measure_triangle(*tri)
                alpha_min = min(alpha_min, a)
                beta_min = min(beta_min, b)
                gamma_min = min(gamma_min, g)
                n_cells += 1

    return {
        'n_cells': n_cells,
        'alpha_min': alpha_min,
        'beta_min': beta_min,
        'gamma_min': gamma_min,
        'overall': (alpha_min + beta_min + gamma_min) / 3.0,
    }


# ======================================================================
# 相空间采样密度估计
# ======================================================================
def phase_space_density_estimate(samples: List[List[float]],
                                   n_bins: int = 8) -> List[List[float]]:
    """
    相空间采样密度的直方图估计。
    samples: N 个点, 每个为 [x, Q2, ...]
    返回 n_bins x n_bins 密度矩阵。
    """
    if not samples:
        return [[0.0] * n_bins for _ in range(n_bins)]

    # 范围
    dim = len(samples[0])
    mins = [min(s[d] for s in samples) for d in range(min(2, dim))]
    maxs = [max(s[d] for s in samples) for d in range(min(2, dim))]

    ranges = [maxs[d] - mins[d] for d in range(min(2, dim))]
    for d in range(len(ranges)):
        if ranges[d] < 1e-15:
            ranges[d] = 1.0

    # 直方图
    density = [[0.0] * n_bins for _ in range(n_bins)]
    for s in samples:
        ix = min(int((s[0] - mins[0]) / ranges[0] * n_bins), n_bins - 1)
        iy = min(int((s[1] - mins[1]) / ranges[1] * n_bins), n_bins - 1) if dim > 1 else 0
        density[ix][iy] += 1.0

    # 归一化
    total = sum(sum(row) for row in density)
    if total > 0:
        for i in range(n_bins):
            for j in range(n_bins):
                density[i][j] /= total

    return density


# ======================================================================
# 自适应细化判据
# ======================================================================
def refinement_indicator(density: List[List[float]], threshold: float = 0.05) -> List[Tuple[int, int]]:
    """
    标记需要细化的单元 (密度高于阈值的)。
    返回 [(i, j), ...] 列表。
    """
    cells_to_refine = []
    n = len(density)
    avg = sum(sum(row) for row in density) / (n * n) if n > 0 else 0.0
    for i in range(n):
        for j in range(n):
            if density[i][j] > threshold * avg * n * n:
                cells_to_refine.append((i, j))
    return cells_to_refine


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Triangle Quality Measures ===")
    # 等边三角形 (quality = 1)
    p1 = [0.0, 0.0]
    p2 = [1.0, 0.0]
    p3 = [0.5, math.sqrt(3)/2]
    print(f"  Equilateral: alpha={alpha_measure_triangle(p1,p2,p3):.4f}, "
          f"beta={beta_measure_triangle(p1,p2,p3):.4f}, "
          f"gamma={gamma_measure_triangle(p1,p2,p3):.4f}")

    # 退化三角形
    p4 = [0.5, 0.0]
    print(f"  Degenerate:  alpha={alpha_measure_triangle(p1,p2,p4):.4f}, "
          f"beta={beta_measure_triangle(p1,p2,p4):.4f}")

    print("\n=== Phase Space Mesh Quality ===")
    x_grid = [0.01 + 0.98 * i / 9 for i in range(10)]
    q2_grid = [1.0 + 99.0 * i / 9 for i in range(10)]
    q = phase_space_mesh_quality_2d(x_grid, q2_grid)
    print(f"  Cells: {q['n_cells']}")
    print(f"  Alpha min: {q['alpha_min']:.4f}")
    print(f"  Beta min: {q['beta_min']:.4f}")
    print(f"  Overall: {q['overall']:.4f}")
