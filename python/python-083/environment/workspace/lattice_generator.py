"""
lattice_generator.py
====================
晶格结构生成模块：基于 CVT (Centroidal Voronoi Tessellation) 的增材制造可打印晶格。
整合自：
  - 1378_usa_cvt_geo：Lloyd 迭代 CVT 算法
  - 502_hand_data：复杂几何边界描述与裁剪
  - 185_circles：圆形胞元/孔洞几何参数化
  - 1282_tortoise：边界词编码用于打印路径描述

物理背景：
  增材制造（特别是 LPBF）中，晶格结构可显著降低重量同时保持承载能力。
  CVT 生成的晶格节点分布均匀，有利于减少局部应力集中。
  每个 CVT 单元可进一步参数化为圆形/多边形胞元，通过调整半径控制相对密度。
"""

import numpy as np
from typing import Tuple, List, Optional


# =============================================================================
# 1. CVT 生成器 (usa_cvt_geo 思想)
# =============================================================================

def generate_cvt_points(n_generators: int, domain_bounds: Tuple[float, float, float, float],
                        n_samples: int = 5000, n_lloyd: int = 20,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    在矩形域 [xmin, xmax] × [ymin, ymax] 内生成 CVT 节点。

    Lloyd 迭代算法：
        1. 初始随机撒点 G = {g_i}
        2. 在域内随机采样大量点 X = {x_j}
        3. 对每个生成器 g_i，找到其 Voronoi 区域 V_i 内的所有采样点
        4. 更新 g_i <- mean(V_i 内的采样点)
        5. 重复 2-4 直到收敛

    返回生成器坐标，shape (n_generators, 2)。
    """
    if seed is not None:
        np.random.seed(seed)
    xmin, xmax, ymin, ymax = domain_bounds

    # 初始随机生成器
    generators = np.column_stack([
        np.random.uniform(xmin, xmax, n_generators),
        np.random.uniform(ymin, ymax, n_generators)
    ])

    for _ in range(n_lloyd):
        # 域内均匀采样
        samples = np.column_stack([
            np.random.uniform(xmin, xmax, n_samples),
            np.random.uniform(ymin, ymax, n_samples)
        ])

        # 为每个采样点找到最近的生成器
        # 距离矩阵 (n_samples, n_generators)
        # 使用向量化计算
        dx = samples[:, 0:1] - generators[:, 0].reshape(1, -1)
        dy = samples[:, 1:2] - generators[:, 1].reshape(1, -1)
        dists = dx**2 + dy**2
        nearest = np.argmin(dists, axis=1)

        # 更新生成器为所属采样点的质心
        new_gens = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=np.int32)
        for i in range(n_generators):
            mask = nearest == i
            if np.any(mask):
                new_gens[i] = np.mean(samples[mask], axis=0)
                counts[i] = np.sum(mask)
            else:
                # 空区域，随机重新放置
                new_gens[i] = np.array([
                    np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax)
                ])

        generators = new_gens

    return generators


# =============================================================================
# 2. 复杂边界裁剪 (hand_data 思想)
# =============================================================================

def is_inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    判断点集是否在多边形内部（射线法）。
    polygon : ndarray, shape (n_vertices, 2)，首尾不重复。
    """
    n_points = points.shape[0]
    inside = np.zeros(n_points, dtype=bool)
    n_vert = polygon.shape[0]

    for i in range(n_points):
        x, y = points[i]
        crossings = 0
        for j in range(n_vert):
            x1, y1 = polygon[j]
            x2, y2 = polygon[(j + 1) % n_vert]
            # 检查边是否跨越水平射线 y = y_i
            if ((y1 > y) != (y2 > y)):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-14)
                if x_intersect > x:
                    crossings += 1
        inside[i] = (crossings % 2 == 1)
    return inside


def clip_points_to_boundary(points: np.ndarray,
                             boundary_polygon: np.ndarray) -> np.ndarray:
    """
    将 CVT 生成器裁剪到复杂边界多边形内部。
    整合 hand_data 的复杂几何轮廓处理思想。
    """
    mask = is_inside_polygon(points, boundary_polygon)
    return points[mask]


def generate_hand_like_boundary(lx: float = 10.0, ly: float = 6.0,
                                 n_points: int = 40) -> np.ndarray:
    """
    生成一个类似手部轮廓的复杂闭合多边形（参数化曲线）。
    用于模拟增材制造中复杂零件的边界。
    """
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    # 使用参数方程构造类似手/爪形的轮廓
    # r(θ) = a + b·cos(3θ) + c·sin(5θ)
    a = 0.45 * min(lx, ly)
    b = 0.15 * min(lx, ly)
    c = 0.08 * min(lx, ly)
    r = a + b * np.cos(3*t) + c * np.sin(5*t)
    x = 0.5 * lx + r * np.cos(t)
    y = 0.5 * ly + r * np.sin(t) * 0.7  # 压扁一些
    return np.column_stack([x, y])


# =============================================================================
# 3. 圆形胞元参数化 (circles 思想)
# =============================================================================

def generate_circle_lattice(centers: np.ndarray, radii: np.ndarray,
                             n_segments: int = 16) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    为每个 CVT 生成器生成圆形/多边形胞元的轮廓点。

    参数
    ----
    centers : ndarray, shape (n_cells, 2)
    radii : ndarray, shape (n_cells,)
    n_segments : int
        每个圆离散为正 n_segments 边形。

    Returns
    -------
    all_points : ndarray
        所有轮廓点的坐标。
    cell_rings : list of ndarray
        每个胞元的轮廓点索引。
    """
    n_cells = len(centers)
    all_points_list = []
    cell_rings = []
    offset = 0
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)

    for i in range(n_cells):
        cx, cy = centers[i]
        r = radii[i]
        # 正多边形顶点
        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        pts = np.column_stack([px, py])
        all_points_list.append(pts)
        cell_rings.append(np.arange(offset, offset + n_segments))
        offset += n_segments

    if len(all_points_list) == 0:
        return np.zeros((0, 2)), []
    all_points = np.vstack(all_points_list)
    return all_points, cell_rings


def compute_lattice_relative_density(centers: np.ndarray, radii: np.ndarray,
                                      domain_area: float) -> float:
    """
    计算圆形胞元晶格的相对密度（体积分数）。
    ρ_rel = Σ_i π r_i² / A_domain
    """
    total_area = np.sum(np.pi * radii**2)
    return total_area / domain_area


# =============================================================================
# 4. 边界词编码 (tortoise 思想)
# =============================================================================

def boundary_word_encode(path_points: np.ndarray,
                          step_letters: str = "ABCDEFGHIJKL") -> str:
    """
    将二维路径点序列编码为边界词（tortoise 思想）。
    使用 12 个方向字母描述相邻点之间的步进方向（每步 30°）。

    方向编码（0° = A, 30° = B, ..., 330° = L）
    """
    n = len(path_points)
    if n < 2:
        return ""
    word = []
    for i in range(n - 1):
        dx = path_points[i+1, 0] - path_points[i, 0]
        dy = path_points[i+1, 1] - path_points[i, 1]
        angle = np.arctan2(dy, dx)
        # 映射到 12 个方向
        sector = int(np.round(angle / (np.pi / 6.0))) % 12
        word.append(step_letters[sector])
    return "".join(word)


def decode_boundary_word(word: str, start_point: np.ndarray,
                          step_length: float = 1.0,
                          step_letters: str = "ABCDEFGHIJKL") -> np.ndarray:
    """
    由边界词解码出路径点坐标。
    """
    n = len(word)
    points = np.zeros((n + 1, 2), dtype=np.float64)
    points[0] = start_point
    for i, ch in enumerate(word):
        idx = step_letters.index(ch)
        angle = idx * np.pi / 6.0
        points[i+1, 0] = points[i, 0] + step_length * np.cos(angle)
        points[i+1, 1] = points[i, 1] + step_length * np.sin(angle)
    return points


def generate_print_path_word(centers: np.ndarray, radii: np.ndarray,
                              n_segments: int = 16) -> List[str]:
    """
    为每个胞元生成增材制造打印路径的边界词描述。
    返回每个胞元的边界词列表。
    """
    words = []
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    for i in range(len(centers)):
        cx, cy = centers[i]
        r = radii[i]
        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        pts = np.column_stack([px, py])
        # 闭合路径
        pts_closed = np.vstack([pts, pts[0:1]])
        word = boundary_word_encode(pts_closed)
        words.append(word)
    return words


# =============================================================================
# 5. 主生成接口
# =============================================================================

def generate_am_lattice(domain_bounds: Tuple[float, float, float, float],
                        n_cells: int = 50, target_density: float = 0.3,
                        use_complex_boundary: bool = False,
                        seed: Optional[int] = None) -> dict:
    """
    生成面向增材制造的晶格结构。

    Returns
    -------
    dict with keys:
        'centers', 'radii', 'relative_density', 'boundary_polygon',
        'print_path_words', 'all_ring_points', 'cell_rings'
    """
    xmin, xmax, ymin, ymax = domain_bounds
    domain_area = (xmax - xmin) * (ymax - ymin)

    # 生成 CVT 节点
    centers = generate_cvt_points(n_cells, domain_bounds, n_samples=5000,
                                   n_lloyd=15, seed=seed)

    boundary_polygon = None
    if use_complex_boundary:
        boundary_polygon = generate_hand_like_boundary(xmax - xmin, ymax - ymin)
        # 将边界平移到域起点
        boundary_polygon += np.array([xmin, ymin])
        centers = clip_points_to_boundary(centers, boundary_polygon)
        # 如果裁剪后太少，重新生成
        if len(centers) < max(3, n_cells // 4):
            centers = generate_cvt_points(n_cells * 2, domain_bounds,
                                           n_samples=5000, n_lloyd=15, seed=seed)
            centers = clip_points_to_boundary(centers, boundary_polygon)

    n_actual = len(centers)
    # 计算半径使得总体积分数接近 target_density
    # ρ_rel = n · π r² / A  =>  r = sqrt(ρ_rel · A / (n · π))
    if n_actual > 0:
        r_base = np.sqrt(target_density * domain_area / (n_actual * np.pi))
    else:
        r_base = 0.1
    # 引入轻微随机扰动模拟工艺不确定性
    radii = r_base * (0.8 + 0.4 * np.random.rand(n_actual))

    # 确保半径不会导致胞元过度重叠（简单限制）
    # 计算最近邻距离
    if n_actual > 1:
        for i in range(n_actual):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            min_dist = np.min(dists)
            max_r = 0.45 * min_dist
            radii[i] = min(radii[i], max_r)

    rel_density = compute_lattice_relative_density(centers, radii, domain_area)

    all_ring_points, cell_rings = generate_circle_lattice(centers, radii)
    words = generate_print_path_word(centers, radii)

    return {
        "centers": centers,
        "radii": radii,
        "relative_density": rel_density,
        "boundary_polygon": boundary_polygon,
        "print_path_words": words,
        "all_ring_points": all_ring_points,
        "cell_rings": cell_rings,
    }
