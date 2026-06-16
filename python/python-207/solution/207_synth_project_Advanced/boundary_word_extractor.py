"""
boundary_word_extractor.py — 边界字提取与置信区域拓扑分析

科学背景
========
在 2D 不确定性量化中, 置信区域 {(x,y): u(x,y) ∈ CI(x,y)}
的边界可以由"边界字" (boundary word) 编码.
边界字是一个符号序列, 描述沿区域边界的拓扑变化.

算法来源 (种子项目 106_boundary_word_drafter)
=============================================
种子 106 定义了边界字的代数操作:
- word_reflect: 字反射
- word_rotate: 字旋转
- word_reverse: 字反转
- word_translate: 字平移
- word_parity: 字奇偶性
- word_to_edge: 字到边映射

在本项目中的角色
================
1. 编码 2D 置信区域的边界拓扑
2. 检测置信区域的连通分量数
3. 计算边界字的拓扑不变量 (Euler 特征)

核心公式
========
1. 边界字: W = (s_1, s_2, ..., s_N),  s_i ∈ {N, S, E, W}
2. 环绕数:  W(W) = (1/2π) · Σ Δθ_i
3. Euler 特征: χ = V - E + F
"""

import numpy as np


def extract_boundary_from_levelset(field_2d, threshold):
    """从 2D 标量场中提取等值线边界.

    使用 Marching Squares 的简化版本.

    参数
    ----
    field_2d : ndarray, shape (ny, nx)
    threshold : float

    返回
    ----
    boundary_points : list of (int, int)
        边界像素坐标
    """
    ny, nx = field_2d.shape
    binary = (field_2d > threshold).astype(int)
    boundary = []

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if binary[j, i] == 1:
                # 检查是否与 0 相邻
                neighbors = [
                    binary[j - 1, i], binary[j + 1, i],
                    binary[j, i - 1], binary[j, i + 1]
                ]
                if 0 in neighbors:
                    boundary.append((i, j))

    return boundary


def boundary_to_word(boundary_points, center):
    """将边界点序列编码为边界字.

    对每个边界点, 计算相对于中心的方向:
    N (北), S (南), E (东), W (西), NE, NW, SE, SW

    参数
    ----
    boundary_points : list of (int, int)
    center : (float, float)

    返回
    ----
    word : list of str
    """
    cx, cy = center
    directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    word = []

    for (px, py) in boundary_points:
        dx = px - cx
        dy = py - cy
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            continue
        angle = np.arctan2(dy, dx)
        # 量化为 8 方向
        idx = int(np.round(angle / (np.pi / 4))) % 8
        word.append(directions[idx])

    return word


def word_parity(word):
    """计算边界字的奇偶性 (种子 106).

    参数
    ----
    word : list of str

    返回
    ----
    parity : int
        0 = 偶, 1 = 奇
    """
    return len(word) % 2


def word_reverse(word):
    """反转边界字 (种子 106)."""
    return list(reversed(word))


def word_rotate(word, k=1):
    """旋转边界字 k 步 (种子 106)."""
    if len(word) == 0:
        return word
    k = k % len(word)
    return word[k:] + word[:k]


def word_reflect(word, axis='horizontal'):
    """反射边界字 (种子 106)."""
    reflect_map_h = {'N': 'S', 'S': 'N', 'E': 'E', 'W': 'W',
                     'NE': 'SE', 'NW': 'SW', 'SE': 'NE', 'SW': 'NW'}
    reflect_map_v = {'N': 'N', 'S': 'S', 'E': 'W', 'W': 'E',
                     'NE': 'NW', 'NW': 'NE', 'SE': 'SW', 'SW': 'SE'}
    rmap = reflect_map_h if axis == 'horizontal' else reflect_map_v
    return [rmap.get(c, c) for c in word]


def compute_winding_number(word):
    """计算边界字的环绕数.

    W = (1/2π) · Σ Δθ_i

    参数
    ----
    word : list of str

    返回
    ----
    winding : int
    """
    if len(word) < 2:
        return 0

    dir_to_angle = {
        'E': 0, 'NE': np.pi / 4, 'N': np.pi / 2, 'NW': 3 * np.pi / 4,
        'W': np.pi, 'SW': -3 * np.pi / 4, 'S': -np.pi / 2, 'SE': -np.pi / 4
    }

    total_angle = 0.0
    for i in range(len(word) - 1):
        a1 = dir_to_angle.get(word[i], 0)
        a2 = dir_to_angle.get(word[i + 1], 0)
        da = a2 - a1
        # 归一化到 [-π, π]
        while da > np.pi:
            da -= 2 * np.pi
        while da < -np.pi:
            da += 2 * np.pi
        total_angle += da

    winding = int(np.round(total_angle / (2 * np.pi)))
    return winding


def analyze_confidence_region_topology(field_2d, ci_lower, ci_upper):
    """分析 2D 置信区域的拓扑结构.

    参数
    ----
    field_2d : ndarray, shape (ny, nx)  均值场
    ci_lower, ci_upper : ndarray, shape (ny, nx)  置信带

    返回
    ----
    info : dict
    """
    # 在置信带内的区域
    in_band = ((field_2d >= ci_lower) & (field_2d <= ci_upper)).astype(int)

    # 提取边界
    boundary_pts = extract_boundary_from_levelset(in_band.astype(float), 0.5)

    if len(boundary_pts) == 0:
        return {
            'n_boundary_points': 0,
            'winding_number': 0,
            'boundary_word': [],
            'parity': 0,
            'area_fraction': np.mean(in_band),
        }

    # 计算中心
    pts = np.array(boundary_pts)
    center = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))

    # 编码为字
    word = boundary_to_word(boundary_pts, center)

    return {
        'n_boundary_points': len(boundary_pts),
        'winding_number': compute_winding_number(word),
        'boundary_word_length': len(word),
        'parity': word_parity(word),
        'area_fraction': np.mean(in_band),
        'n_components_estimate': max(1, abs(compute_winding_number(word))),
    }
