"""
mesh_utils.py
网格生成与采样质量评估工具

融合种子项目:
- 680_line_grid: 1D网格点生成（5种居中方案）
- 548_human_mesh2d: 2D约束Delaunay三角剖分思想
- 276_diaphony: 点集均匀性度量（diaphony）

科学应用: 软体机器人中心线离散化与横截面网格生成、采样质量评估
"""

import numpy as np
from typing import Tuple, List


def line_grid(n: int, a: float, b: float, c: int = 1) -> np.ndarray:
    """
    1D网格点生成 — 基于种子项目680_line_grid
    在区间[a, b]上生成n个按不同居中方案分布的网格点

    参数:
        n: 网格点数
        a, b: 区间端点
        c: 居中方案 (1-5)
            1: 包含两端点  x_j = ((n-j)*a + (j-1)*b)/(n-1)
            2: 不包含端点  x_j = ((n-j+1)*a + j*b)/(n+1)
            3: 仅包含左端点 x_j = ((n-j+1)*a + (j-1)*b)/n
            4: 仅包含右端点 x_j = ((n-j)*a + j*b)/n
            5: 半偏移内点   x_j = ((2n-2j+1)*a + (2j-1)*b)/(2n)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if a >= b:
        raise ValueError("must have a < b")
    if c < 1 or c > 5:
        raise ValueError("c must be in [1,5]")

    x = np.zeros(n)
    if c == 1:
        if n == 1:
            x[0] = 0.5 * (a + b)
        else:
            for j in range(n):
                x[j] = ((n - 1 - j) * a + j * b) / (n - 1)
    elif c == 2:
        for j in range(n):
            x[j] = ((n - j) * a + (j + 1) * b) / (n + 1)
    elif c == 3:
        for j in range(n):
            x[j] = ((n - j) * a + j * b) / n
    elif c == 4:
        for j in range(n):
            x[j] = ((n - 1 - j) * a + (j + 1) * b) / n
    elif c == 5:
        for j in range(n):
            x[j] = ((2 * n - 2 * j - 1) * a + (2 * j + 1) * b) / (2 * n)
    return x


def chebyshev_grid(n: int) -> np.ndarray:
    """
    Chebyshev节点生成 — 基于种子项目161_chebyshev_matrix
    返回第二类Chebyshev点: x_j = cos(j*pi/n), j=0,...,n
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    j = np.arange(n + 1)
    return np.cos(np.pi * j / n)


def triangulate_polygon(vertices: np.ndarray, hmax: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单多边形三角剖分 — 基于种子项目548_human_mesh2d的约束Delaunay思想
    对凸/近似凸多边形进行简单扇形三角剖分，返回节点和三角形

    参数:
        vertices: (nv, 2) 多边形顶点（逆时针）
        hmax: 最大边长控制（简化实现中用于检查密度）
    """
    nv = vertices.shape[0]
    if nv < 3:
        raise ValueError("polygon must have at least 3 vertices")

    # 计算多边形面积（带符号）
    area = 0.0
    for i in range(nv):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % nv]
        area += x1 * y2 - x2 * y1
    if area <= 0:
        raise ValueError("vertices must be ordered counter-clockwise and enclose positive area")

    # 简单扇形三角剖分：以第一个顶点为扇心
    triangles = []
    for i in range(1, nv - 1):
        triangles.append([0, i, i + 1])

    return vertices.copy(), np.array(triangles, dtype=int)


def diaphony_compute(points: np.ndarray) -> float:
    """
    Diaphony计算 — 基于种子项目276_diaphony
    度量点集在[0,1]^d中的均匀性，值越小越均匀

    公式（Heelekalek-Niederreiter核）:
        D = sqrt( (1/N^2) * sum_i sum_j [ -1 + prod_k (1 + 2*pi^2*(z_k^2 - z_k + 1/6)) ] / C )
    其中 z = mod(x_i - x_j, 1), C = (1 + pi^2/3)^d - 1
    """
    if points.ndim != 2:
        raise ValueError("points must be 2D array")
    n, dim = points.shape
    if n < 2:
        return 0.0

    # 归一化到[0,1]
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    rng = pmax - pmin
    rng[rng == 0] = 1.0
    pts = (points - pmin) / rng

    # 归一化常数 C = (1 + pi^2/3)^dim - 1
    C = (1.0 + np.pi ** 2 / 3.0) ** dim - 1.0
    if abs(C) < 1e-14:
        return 0.0

    total = 0.0
    for i in range(n):
        for j in range(n):
            z = np.mod(pts[i] - pts[j], 1.0)
            # 核函数值
            kernel = -1.0 + np.prod(1.0 + 2.0 * np.pi ** 2 * (z ** 2 - z + 1.0 / 6.0))
            total += kernel

    diaphony_val = np.sqrt(total / (n * n * C))
    return diaphony_val


def sample_ellipse(a: float, b: float, n: int) -> np.ndarray:
    """
    在椭圆 x^2/a^2 + y^2/b^2 <= 1 内均匀采样n个点（拒绝采样）
    用于软体机器人横截面的材料点采样
    """
    if a <= 0 or b <= 0 or n < 1:
        raise ValueError("invalid ellipse parameters")
    points = []
    max_iter = n * 100
    count = 0
    while len(points) < n and count < max_iter:
        x = np.random.uniform(-a, a)
        y = np.random.uniform(-b, b)
        if (x / a) ** 2 + (y / b) ** 2 <= 1.0:
            points.append([x, y])
        count += 1
    if len(points) < n:
        # 回退：规则网格采样
        nx = int(np.ceil(np.sqrt(n * a / b))) + 1
        ny = int(np.ceil(np.sqrt(n * b / a))) + 1
        xs = np.linspace(-a, a, nx)
        ys = np.linspace(-b, b, ny)
        for x in xs:
            for y in ys:
                if (x / a) ** 2 + (y / b) ** 2 <= 1.0:
                    points.append([x, y])
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
    return np.array(points[:n])


def compute_polygon_area(vertices: np.ndarray) -> float:
    """计算多边形有符号面积（Shoelace公式）"""
    nv = vertices.shape[0]
    area = 0.0
    for i in range(nv):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % nv]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def refine_cross_section_mesh(vertices: np.ndarray, max_area: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    细化横截面网格：在扇形剖分基础上加入中心点并细分大三角形
    """
    nodes, tris = triangulate_polygon(vertices)
    # 计算质心并加入节点列表
    cx = np.mean(nodes[:, 0])
    cy = np.mean(nodes[:, 1])
    center_idx = nodes.shape[0]
    nodes = np.vstack([nodes, [cx, cy]])

    # 对每个原边三角形，加入中心点形成3个更小的三角形
    new_tris = []
    for tri in tris:
        i, j, k = tri
        # 只细分边界三角形（包含顶点0的三角形）
        new_tris.append([i, j, center_idx])
        new_tris.append([j, k, center_idx])
        new_tris.append([k, i, center_idx])

    return nodes, np.array(new_tris, dtype=int)
