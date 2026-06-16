"""
mesh.py — 有限元网格生成与操作 (6节点三角形)
============================================
来源项目: 513_hello (初始化), 874_ply_to_tri_surface (表面三角化/PLY格式),
          1343_triangulation_order6_contour (6节点三角形拓扑)

本模块实现:
  1. 结构化矩形域上的 6 节点二次三角形网格生成
  2. PLY 格式三角面片的扇形三角化 (fan triangulation)
  3. 单元质量度量 (aspect ratio, skewness)
  4. 节点-单元关联数据结构

6 节点三角形节点编号约定:
    n3
   /  \
  n6   n5     n1,n2,n3 = 顶点
 /      \    n4 = n1-n2 中点
n1--n4--n2   n5 = n2-n3 中点
              n6 = n3-n1 中点
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from config import EPS_NUM


# ============================================================
#  数据结构: 6节点三角形网格
# ============================================================
class TriMesh6:
    """6 节点二次三角形有限元网格

    Attributes:
        nodes: 节点坐标 (n_nodes, 2)
        elements: 单元连接 (n_elements, 6), 每行 [n1,n2,n3,n4,n5,n6]
        n_nodes: 节点总数
        n_elements: 单元总数
        boundary_nodes: 边界节点索引集合
        node_to_elements: 节点 → 关联单元映射
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

        if self.elements.shape[1] != 6:
            raise ValueError(f"6节点三角形要求 elements.shape[1]==6, got {self.elements.shape[1]}")

        self.boundary_nodes = self._find_boundary_nodes()
        self.node_to_elements = self._build_node_to_element_map()
        self.element_volumes = self._compute_element_areas()

    def _find_boundary_nodes(self) -> set:
        """识别边界节点 (位于域边界的节点)"""
        boundary = set()
        # 边界节点: 仅被一侧的单元共享
        edge_count: Dict[Tuple[int, int], int] = {}
        for e in range(self.n_elements):
            verts = [self.elements[e, 0], self.elements[e, 1], self.elements[e, 2]]
            for i in range(3):
                edge = tuple(sorted((verts[i], verts[(i + 1) % 3])))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        for edge, count in edge_count.items():
            if count == 1:  # 边界边
                boundary.add(edge[0])
                boundary.add(edge[1])
        return boundary

    def _build_node_to_element_map(self) -> Dict[int, List[int]]:
        """构建节点到单元的邻接映射"""
        n2e: Dict[int, List[int]] = {}
        for e in range(self.n_elements):
            for local_n in range(6):
                global_n = self.elements[e, local_n]
                if global_n not in n2e:
                    n2e[global_n] = []
                n2e[global_n].append(e)
        return n2e

    def _compute_element_areas(self) -> np.ndarray:
        """计算每个三角形的面积 (基于顶点)"""
        areas = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            n1, n2, n3 = self.elements[e, 0], self.elements[e, 1], self.elements[e, 2]
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            x3, y3 = self.nodes[n3]
            # 有向面积 = 0.5 * |cross product|
            areas[e] = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        return areas

    def get_element_coords(self, e: int) -> np.ndarray:
        """获取第 e 个单元的 6 个节点坐标 (6, 2)"""
        return self.nodes[self.elements[e]]

    def get_linear_subtriangles(self, e: int) -> np.ndarray:
        """将 6 节点三角形分裂为 4 个线性子三角形 (来源: 1343)
        分裂方式:
          [n1,n2,n3,n4,n5,n6] → (n1,n4,n6), (n2,n5,n4), (n3,n6,n5), (n4,n5,n6)
        """
        n = self.elements[e]
        return np.array([
            [n[0], n[3], n[5]],  # 子三角形 1
            [n[1], n[4], n[3]],  # 子三角形 2
            [n[2], n[5], n[4]],  # 子三角形 3
            [n[3], n[4], n[5]],  # 中心子三角形
        ])

    def min_quality(self) -> float:
        """计算网格最小单元质量 (面积/最长边^2)"""
        qualities = []
        for e in range(self.n_elements):
            coords = self.get_element_coords(e)[:3]  # 只用顶点
            v = [coords[i] for i in range(3)]
            edges = [np.linalg.norm(v[(i+1)%3] - v[i]) for i in range(3)]
            max_edge = max(edges)
            if max_edge < EPS_NUM:
                qualities.append(0.0)
            else:
                qualities.append(self.element_volumes[e] / (max_edge ** 2))
        return min(qualities) if qualities else 0.0


# ============================================================
#  结构化 6 节点三角形网格生成
# ============================================================
def generate_structured_mesh(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0) -> TriMesh6:
    """在 [0, Lx] x [0, Ly] 上生成结构化 6 节点三角形网格

    每个矩形单元 (i, j) 分裂为 2 个三角形 (对角线方向固定以保证一致性).
    节点编号:
      - 顶点: (j*(2*nx+1) + i*2) 类型
      - x 边中点: 在顶点之间
      - y 边中点: 在顶点之间
      - 面中心: 每个三角形一个

    节点布局 (2*nx+1) x (2*ny+1):
      (2*j, 2*i)     = 矩形顶点 (i, j)
      (2*j, 2*i+1)   = x 方向边中点
      (2*j+1, 2*i)   = y 方向边中点
      面中心节点单独编号

    Args:
        nx, ny: 矩形单元数量
        Lx, Ly: 域尺寸

    Returns:
        TriMesh6 对象
    """
    hx = Lx / nx
    hy = Ly / ny

    # ---- 节点生成 ----
    # 第一类: 结构化节点 (2*nx+1) x (2*ny+1)
    n_struct = (2 * nx + 1) * (2 * ny + 1)
    # 第二类: 面中心节点 (每个矩形 2 个三角形, 每个 1 个中心)
    n_face = 2 * nx * ny
    n_total = n_struct + n_face

    nodes = np.zeros((n_total, 2))

    # 结构化节点坐标
    for jj in range(2 * ny + 1):
        for ii in range(2 * nx + 1):
            idx = jj * (2 * nx + 1) + ii
            nodes[idx, 0] = ii * hx / 2.0
            nodes[idx, 1] = jj * hy / 2.0

    # 面中心节点
    face_idx = n_struct
    for j in range(ny):
        for i in range(nx):
            # 矩形 (i, j) 的中心
            cx = (i + 0.5) * hx
            cy = (j + 0.5) * hy
            # 下三角形中心
            nodes[face_idx, 0] = cx - hx / 6.0
            nodes[face_idx, 1] = cy - hy / 6.0
            face_idx += 1
            # 上三角形中心
            nodes[face_idx, 0] = cx + hx / 6.0
            nodes[face_idx, 1] = cy + hy / 6.0
            face_idx += 1

    def node_id(ii: int, jj: int) -> int:
        """结构化节点的全局 ID"""
        return jj * (2 * nx + 1) + ii

    # ---- 单元生成 ----
    elements = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    e_idx = 0
    face_center_start = n_struct

    for j in range(ny):
        for i in range(nx):
            # 矩形四角
            sw = node_id(2 * i, 2 * j)       # 西南
            se = node_id(2 * i + 2, 2 * j)   # 东南
            nw = node_id(2 * i, 2 * j + 2)   # 西北
            ne = node_id(2 * i + 2, 2 * j + 2)  # 东北
            # 边中点
            s_mid = node_id(2 * i + 1, 2 * j)    # 南边中点
            e_mid = node_id(2 * i + 2, 2 * j + 1)  # 东边中点
            n_mid = node_id(2 * i + 1, 2 * j + 2)  # 北边中点
            w_mid = node_id(2 * i, 2 * j + 1)    # 西边中点

            fc1 = face_center_start + 2 * (j * nx + i)      # 下三角形中心
            fc2 = face_center_start + 2 * (j * nx + i) + 1  # 上三角形中心

            # 下三角形 (SW, SE, NE), 对角线 SW-NE
            # 6节点: [v1, v2, v3, mid12, mid23, mid31]
            elements[e_idx] = [sw, se, ne, s_mid, e_mid, fc1]
            e_idx += 1

            # 上三角形 (SW, NE, NW)
            elements[e_idx] = [sw, ne, nw, fc2, n_mid, w_mid]
            e_idx += 1

    mesh = TriMesh6(nodes, elements)
    return mesh


# ============================================================
#  PLY 格式扇形三角化 (来源: 874_ply_to_tri_surface)
# ============================================================
def fan_triangulate(polygon_vertices: List[np.ndarray]) -> List[Tuple[int, int, int]]:
    """将凸多边形扇形三角化

    对于 n 边形 [v0, v1, ..., v_{n-1}], 生成三角形:
      (v0, v1, v2), (v0, v2, v3), ..., (v0, v_{n-2}, v_{n-1})
    共 n-2 个三角形.

    Args:
        polygon_vertices: 多边形顶点列表 (按逆时针顺序)

    Returns:
        三角形列表 [(i, j, k), ...]
    """
    n = len(polygon_vertices)
    if n < 3:
        raise ValueError(f"多边形至少需要 3 个顶点, got {n}")
    triangles = []
    for k in range(1, n - 1):
        triangles.append((0, k, k + 1))
    return triangles


def compute_polygon_area_3d(vertices: List[np.ndarray]) -> float:
    """计算 3D 多边形面积 (Newell 方法)

    A = 0.5 * ||sum_{i} (v_i x v_{i+1})||
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    normal = np.zeros(3)
    for i in range(n):
        v_curr = vertices[i]
        v_next = vertices[(i + 1) % n]
        normal += np.cross(v_curr, v_next)
    return 0.5 * np.linalg.norm(normal)


# ============================================================
#  单元质量度量
# ============================================================
def compute_aspect_ratio(coords: np.ndarray) -> float:
    """计算三角形宽高比
    AR = (最长边) / (2 * 内切圆半径)
    等边三角形 AR = 2/sqrt(3) ≈ 1.1547
    """
    v = coords[:3]
    a = np.linalg.norm(v[1] - v[0])
    b = np.linalg.norm(v[2] - v[1])
    c = np.linalg.norm(v[0] - v[2])
    s = (a + b + c) / 2.0
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
    if area < EPS_NUM:
        return float('inf')
    inradius = area / s
    if inradius < EPS_NUM:
        return float('inf')
    return max(a, b, c) / (2.0 * inradius)


def compute_skewness(coords: np.ndarray) -> float:
    """计算三角形偏斜度 (0=等边, 1=退化)
    skewness = 1 - (6*sqrt(3)*area) / (a^2 + b^2 + c^2)
    """
    v = coords[:3]
    a2 = np.sum((v[1] - v[0]) ** 2)
    b2 = np.sum((v[2] - v[1]) ** 2)
    c2 = np.sum((v[0] - v[2]) ** 2)
    denom = a2 + b2 + c2
    if denom < EPS_NUM:
        return 1.0
    if len(coords) >= 3 and coords.shape[1] >= 2:
        area = 0.5 * abs(
            (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1]) -
            (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
        )
    else:
        area = 0.0
    return max(0.0, 1.0 - 6.0 * SQRT3 * area / (denom + EPS_NUM))


# 常量导入
SQRT3 = np.sqrt(3.0)
