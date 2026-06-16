"""
mesh_topology.py  —  粒子-网格拓扑与宇宙网三角剖分
================================================

科学来源种子:
  - 1059_Alex-castro-quim_Glyphosate_crystallization / Topol_creation.py
    借鉴其拓扑创建思想: 将 OpenFF 分子拓扑创建推广到
    粒子-网格拓扑,即网格顶点、边、面的邻接结构。
  - 884_polygon_distance / polygon_triangulate.m, polygon_sample.m,
    triangle_area.m
    直接使用其多边形三角剖分、三角形采样与面积计算。
    在宇宙学中对应 Delaunay 三角剖分 (宇宙网 filament 识别)。
  - 236_cube_surface_distance / cube_surface_distance_stats.m,
    cube_surface_sample.m
    直接使用其立方体表面距离统计与采样方法。
    映射到 3D 周期盒子内粒子对距离分布。

物理背景:
  宇宙大尺度结构呈现"宇宙网"形态,包含节点 (halos)、纤维
  (filaments)、壁 (walls)、空洞 (voids)。Delaney tessellation
  将粒子集划分为四面体,四面体体积的倒数正比于局部密度:
      ρ_i = m_p / V_tet(i)
  其中 m_p 为粒子质量, V_tet 为粒子 i 所属四面体体积。
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List


# ---------------------------------------------------------------------------- #
#                 网格拓扑创建 (源自 Topol_creation.py)
# ---------------------------------------------------------------------------- #
class MeshTopology:
    """
    3D 周期网格的拓扑结构。
    节点: (N+1)³ 个 (含周期副本)
    边: 每节点 6 个方向 (±x, ±y, ±z)
    面: 每节点 12 个对角面
    体: 每节点 8 个立方体胞
    """
    def __init__(self, n_grid: int, box_length: float):
        self.n = n_grid
        self.L = box_length
        self.h = box_length / n_grid
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """构建邻接表 (周期边界)。"""
        n = self.n
        # 节点总数 = n³
        self.n_nodes = n * n * n
        # 边邻接 (每节点 6 邻居):
        self.edges = np.zeros((self.n_nodes, 6), dtype=int)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    idx = i * n * n + j * n + k
                    self.edges[idx, 0] = ((i + 1) % n) * n * n + j * n + k
                    self.edges[idx, 1] = ((i - 1) % n) * n * n + j * n + k
                    self.edges[idx, 2] = i * n * n + ((j + 1) % n) * n + k
                    self.edges[idx, 3] = i * n * n + ((j - 1) % n) * n + k
                    self.edges[idx, 4] = i * n * n + j * n + ((k + 1) % n)
                    self.edges[idx, 5] = i * n * n + j * n + ((k - 1) % n)

    def node_to_ijk(self, idx: int) -> Tuple[int, int, int]:
        n = self.n
        k = idx % n
        j = (idx // n) % n
        i = idx // (n * n)
        return (i, j, k)

    def ijk_to_node(self, i: int, j: int, k: int) -> int:
        return (i % self.n) * self.n * self.n + (j % self.n) * self.n + (k % self.n)

    def node_position(self, idx: int) -> NDArray:
        """返回节点 idx 的物理坐标 (中心)。"""
        i, j, k = self.node_to_ijk(idx)
        return np.array([
            (i + 0.5) * self.h,
            (j + 0.5) * self.h,
            (k + 0.5) * self.h,
        ])

    def periodic_distance(self, a: NDArray, b: NDArray) -> float:
        """周期盒子中最短距离。"""
        d = a - b
        d = d - self.L * np.round(d / self.L)
        return float(np.linalg.norm(d))


# ---------------------------------------------------------------------------- #
#              2D 多边形三角剖分 (源自 polygon_triangulate.m)
# ---------------------------------------------------------------------------- #
def polygon_triangulate(vertices: NDArray) -> List[Tuple[int, int, int]]:
    """
    简单多边形的耳切法三角剖分 (对应 polygon_triangulate.m)。

    Parameters
    ----------
    vertices : (nv, 2) array  多边形顶点 (逆时针)

    Returns
    -------
    triangles : list of (i, j, k)  顶点索引三元组
    """
    nv = vertices.shape[0]
    if nv < 3:
        return []
    remaining = list(range(nv))
    triangles: List[Tuple[int, int, int]] = []
    max_iter = nv * nv
    it = 0
    while len(remaining) > 3 and it < max_iter:
        it += 1
        for idx in range(len(remaining)):
            i_prev = remaining[(idx - 1) % len(remaining)]
            i_curr = remaining[idx]
            i_next = remaining[(idx + 1) % len(remaining)]
            a = vertices[i_prev]
            b = vertices[i_curr]
            c = vertices[i_next]
            if _is_convex(a, b, c) and _no_point_inside(
                vertices, remaining, i_prev, i_curr, i_next, a, b, c
            ):
                triangles.append((i_prev, i_curr, i_next))
                remaining.pop(idx)
                break
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    return triangles


def _is_convex(a: NDArray, b: NDArray, c: NDArray) -> bool:
    """检查角 abc 是否为凸 (2D 叉积 > 0)。"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


def _no_point_inside(vertices, remaining, i_prev, i_curr, i_next,
                     a, b, c) -> bool:
    """检查三角形 (a, b, c) 内是否无其他顶点。"""
    for idx in remaining:
        if idx in (i_prev, i_curr, i_next):
            continue
        p = vertices[idx]
        if _point_in_triangle(p, a, b, c):
            return False
    return True


def _point_in_triangle(p: NDArray, a: NDArray, b: NDArray, c: NDArray) -> bool:
    """重心坐标法判定点 p 是否在三角形 (a,b,c) 内。"""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv = 1.0 / max(dot00 * dot11 - dot01 * dot01, 1e-30)
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def triangle_area(a: NDArray, b: NDArray, c: NDArray) -> float:
    """三角形面积 (源自 triangle_area.m): 0.5 |AB × AC| (2D 用叉积)。"""
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


# ---------------------------------------------------------------------------- #
#          3D 周期盒子内粒子对距离分布 (源自 cube_surface_distance)
# ---------------------------------------------------------------------------- #
def periodic_distance_sample(n_samples: int, box_length: float,
                             seed: int = 42) -> NDArray:
    """
    在立方体 [0, L]³ 表面均匀采样 n 个点 (源自 cube_surface_sample.m):
        6 个面,每面概率 1/6; 面上均匀采样另两坐标。
    返回采样点 (n, 3)。
    """
    rng = np.random.default_rng(seed)
    points = np.zeros((n_samples, 3))
    faces = rng.integers(0, 6, size=n_samples)
    for i in range(n_samples):
        f = faces[i]
        u = rng.uniform(0, box_length, size=2)
        if f == 0:
            points[i] = [0.0, u[0], u[1]]
        elif f == 1:
            points[i] = [box_length, u[0], u[1]]
        elif f == 2:
            points[i] = [u[0], 0.0, u[1]]
        elif f == 3:
            points[i] = [u[0], box_length, u[1]]
        elif f == 4:
            points[i] = [u[0], u[1], 0.0]
        else:
            points[i] = [u[0], u[1], box_length]
    return points


def cube_surface_distance_stats(n_samples: int, box_length: float = 1.0,
                                seed: int = 42) -> dict:
    """
    立方体表面两点距离的均值、方差、最小、最大值
    (源自 cube_surface_distance_stats.m)。

    Returns
    -------
    dict with keys {mean, var, min, max}
    """
    p1 = periodic_distance_sample(n_samples, box_length, seed)
    p2 = periodic_distance_sample(n_samples, box_length, seed + 1)
    d = np.linalg.norm(p1 - p2, axis=1)
    return {
        "mean": float(d.mean()),
        "var": float(d.var()),
        "min": float(d.min()),
        "max": float(d.max()),
    }


def particle_pair_distance_stats(positions: NDArray, box_length: float,
                                 max_pairs: int = 10000,
                                 seed: int = 42) -> dict:
    """
    粒子对周期距离统计。用于两点相关函数 ξ(r) 的蒙特卡洛估计。
    """
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]
    d = positions[idx_a] - positions[idx_b]
    d -= box_length * np.round(d / box_length)
    dists = np.linalg.norm(d, axis=1)
    return {
        "mean": float(dists.mean()),
        "var": float(dists.var()),
        "min": float(dists.min()),
        "max": float(dists.max()),
    }
