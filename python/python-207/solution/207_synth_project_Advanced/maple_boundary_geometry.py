"""
maple_boundary_geometry.py — 边界几何提取与区域面积计算

科学背景
========
从 2D 二值场 (黑/白) 中提取黑白边界, 计算边界围成区域的面积.
这在置信区域的几何分析中至关重要:
- 置信区域面积 = 可靠预测的空间范围
- 边界长度 = 置信区域的不确定性梯度

算法来源 (种子项目 714_maple_data)
==================================
种子 714 从图像中提取黑白边界:
1. 将图像转为二值 (阈值 0.5)
2. 跟踪边界像素 (bwboundaries)
3. 计算精确面积 (白像素/总像素)

在本项目中的角色
================
1. 从 2D 置信场中提取边界
2. 计算置信区域的精确相对面积
3. 分析边界的分形维数

核心公式
========
1. 精确面积:  A_exact = #{white pixels} / #{total pixels}
2. 边界长度:  L = Σ_i √((Δx_i)² + (Δy_i)²)
3. 分形维数:  D = lim_{ε→0} ln(N(ε)) / ln(1/ε)
"""

import numpy as np


def extract_boundary_binary(binary_field):
    """从二值场中提取边界 (移植种子 714).

    参数
    ----
    binary_field : ndarray, shape (ny, nx), dtype int
        0 = 背景, 1 = 前景

    返回
    ----
    boundary_pixels : ndarray, shape (M, 2)
        边界像素的 (row, col) 坐标
    """
    ny, nx = binary_field.shape
    # 确保黑边框
    bf = binary_field.copy()
    bf[0, :] = 0
    bf[-1, :] = 0
    bf[:, 0] = 0
    bf[:, -1] = 0

    boundary = []
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if bf[j, i] == 1:
                # 检查 4-邻域是否有 0
                has_bg = (bf[j-1, i] == 0 or bf[j+1, i] == 0 or
                          bf[j, i-1] == 0 or bf[j, i+1] == 0)
                if has_bg:
                    boundary.append([j, i])
            elif bf[j, i] == 0:
                # 检查 4-邻域是否有 1
                has_fg = (bf[j-1, i] == 1 or bf[j+1, i] == 1 or
                          bf[j, i-1] == 1 or bf[j, i+1] == 1)
                if has_fg:
                    boundary.append([j, i])

    if len(boundary) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.array(boundary, dtype=int)


def trace_boundary_ordered(boundary_pixels):
    """将无序边界像素排序为连续轮廓.

    使用最近邻贪心法.

    参数
    ----
    boundary_pixels : ndarray, shape (M, 2)

    返回
    ----
    ordered : ndarray, shape (M, 2)
    """
    if len(boundary_pixels) <= 2:
        return boundary_pixels.copy()

    remaining = list(range(len(boundary_pixels)))
    ordered_idx = [remaining.pop(0)]

    while remaining:
        last = boundary_pixels[ordered_idx[-1]]
        dists = [np.sum((boundary_pixels[r] - last) ** 2) for r in remaining]
        nearest = np.argmin(dists)
        ordered_idx.append(remaining.pop(nearest))

    return boundary_pixels[ordered_idx]


def compute_boundary_length(ordered_boundary, pixel_size=1.0):
    """计算边界总长度.

    L = Σ_i √((Δr_i)² + (Δc_i)²) · pixel_size

    参数
    ----
    ordered_boundary : ndarray, shape (M, 2)
    pixel_size : float

    返回
    ----
    length : float
    """
    if len(ordered_boundary) < 2:
        return 0.0
    diffs = np.diff(ordered_boundary, axis=0)
    segments = np.sqrt(np.sum(diffs ** 2, axis=1))
    return np.sum(segments) * pixel_size


def compute_exact_area(binary_field):
    """计算精确相对面积 (种子 714).

    A_rel = #{foreground pixels} / #{total pixels}

    参数
    ----
    binary_field : ndarray

    返回
    ----
    area_rel : float
    n_fg : int
    n_total : int
    """
    n_total = binary_field.size
    n_fg = np.sum(binary_field > 0)
    return n_fg / n_total, int(n_fg), n_total


def box_counting_dimension(boundary_pixels, min_box=4, max_box=None):
    """盒计数法估计分形维数.

    D = lim_{ε→0} ln(N(ε)) / ln(1/ε)

    参数
    ----
    boundary_pixels : ndarray, shape (M, 2)
    min_box : int
    max_box : int or None

    返回
    ----
    fractal_dim : float
    scales : ndarray
    counts : ndarray
    """
    if len(boundary_pixels) < 4:
        return 1.0, np.array([1]), np.array([1])

    if max_box is None:
        max_box = max(int(np.max(boundary_pixels) - np.min(boundary_pixels)), 8)

    # 使用 2 的幂次作为盒子大小
    scales = []
    counts = []
    box_size = max_box
    while box_size >= min_box:
        # 计算覆盖边界所需的盒子数
        r_min, c_min = np.min(boundary_pixels, axis=0)
        shifted = boundary_pixels - np.array([r_min, c_min])
        grid_idx = shifted // box_size
        unique_boxes = len(set(map(tuple, grid_idx)))
        scales.append(box_size)
        counts.append(unique_boxes)
        box_size = box_size // 2

    scales = np.array(scales, dtype=float)
    counts = np.array(counts, dtype=float)

    if len(scales) < 2:
        return 1.0, scales, counts

    # 线性回归: ln(N) = D · ln(1/ε) + C
    log_s = np.log(1.0 / scales)
    log_n = np.log(counts)
    # 最小二乘
    n_pts = len(log_s)
    sum_x = np.sum(log_s)
    sum_y = np.sum(log_n)
    sum_xy = np.sum(log_s * log_n)
    sum_x2 = np.sum(log_s ** 2)
    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-30:
        return 1.0, scales, counts
    fractal_dim = (n_pts * sum_xy - sum_x * sum_y) / denom

    return float(fractal_dim), scales, counts


def analyze_confidence_region_geometry(field_2d, threshold):
    """分析置信区域的几何特征.

    参数
    ----
    field_2d : ndarray, shape (ny, nx)
    threshold : float

    返回
    ----
    info : dict
    """
    binary = (field_2d > threshold).astype(int)
    boundary = extract_boundary_binary(binary)

    if len(boundary) == 0:
        return {
            'area_relative': np.mean(binary),
            'n_boundary_pixels': 0,
            'boundary_length': 0.0,
            'fractal_dimension': 1.0,
        }

    ordered = trace_boundary_ordered(boundary)
    length = compute_boundary_length(ordered)
    area_rel, _, _ = compute_exact_area(binary)
    frac_dim, _, _ = box_counting_dimension(boundary)

    return {
        'area_relative': area_rel,
        'n_boundary_pixels': len(boundary),
        'boundary_length': length,
        'fractal_dimension': frac_dim,
    }
