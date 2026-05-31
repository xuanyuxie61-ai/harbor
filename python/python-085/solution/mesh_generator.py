"""
mesh_generator.py
网格生成与三角剖分模块
融合种子项目：
  - 1340_triangulation_node_to_element（节点到单元的映射与平均）
  - 1167_stla_to_tri_surface（STL 三角表面解析与拓扑处理）
"""
import numpy as np
from typing import Tuple, List, Optional


class TriMesh2D:
    r"""
    二维三角网格数据结构，用于有限元分析。

    网格包含：
    - nodes: 节点坐标，形状 (n_nodes, 2)
    - elements: 三角形单元，形状 (n_elements, 3)，每行是三个节点索引（0-based）
    - node_to_element: 节点到单元的邻接表
    - boundaries: 边界边列表

    核心公式（线性三角形面积）：
    对于节点 (x1,y1), (x2,y2), (x3,y3)，面积
    A = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must be (n_nodes, 2)")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError("elements must be (n_elements, 3)")
        self.nodes = np.array(nodes, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self._node_to_element: Optional[List[List[int]]] = None
        self._boundary_edges: Optional[np.ndarray] = None
        self._areas: Optional[np.ndarray] = None

    def compute_areas(self) -> np.ndarray:
        r"""
        计算每个三角形单元的有向面积。
        公式：A_e = 0.5 * ((x2-x1)(y3-y1) - (x3-x1)(y2-y1))
        """
        if self._areas is not None:
            return self._areas
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        e = self.elements
        area = 0.5 * ((x[e[:, 1]] - x[e[:, 0]]) * (y[e[:, 2]] - y[e[:, 0]])
                      - (x[e[:, 2]] - x[e[:, 0]]) * (y[e[:, 1]] - y[e[:, 0]]))
        self._areas = area
        return area

    def total_area(self) -> float:
        """网格总面积（取绝对值）。"""
        return float(np.sum(np.abs(self.compute_areas())))

    def build_node_to_element(self) -> List[List[int]]:
        r"""
        建立节点到单元的邻接表。
        融合自 1340_triangulation_node_to_element 的节点-单元映射思想。
        """
        if self._node_to_element is not None:
            return self._node_to_element
        ne = [[] for _ in range(self.n_nodes)]
        for elem_id, tri in enumerate(self.elements):
            for node_id in tri:
                if 0 <= node_id < self.n_nodes:
                    ne[node_id].append(elem_id)
        self._node_to_element = ne
        return ne

    def node_to_element_average(self, node_values: np.ndarray) -> np.ndarray:
        r"""
        将节点值平均到单元值。
        公式：v_e = (1/3) * \sum_{i \in e} v_i
        直接来自 1340_triangulation_node_to_element 的核心算法。
        """
        if node_values.ndim == 1:
            node_values = node_values.reshape(-1, 1)
        if node_values.shape[0] != self.n_nodes:
            raise ValueError("node_values size mismatch")
        elem_vals = np.zeros((self.n_elements, node_values.shape[1]))
        for elem_id, tri in enumerate(self.elements):
            elem_vals[elem_id] = np.mean(node_values[tri], axis=0)
        return elem_vals

    def element_to_node_average(self, elem_values: np.ndarray) -> np.ndarray:
        r"""
        将单元值平均到节点值（面积加权）。
        公式：v_i = (\sum_{e \ni i} A_e * v_e) / (\sum_{e \ni i} A_e)
        """
        if elem_values.ndim == 1:
            elem_values = elem_values.reshape(-1, 1)
        areas = np.abs(self.compute_areas()).reshape(-1, 1)
        ne = self.build_node_to_element()
        node_vals = np.zeros((self.n_nodes, elem_values.shape[1]))
        node_weights = np.zeros(self.n_nodes)
        for elem_id, tri in enumerate(self.elements):
            w = areas[elem_id, 0]
            for node_id in tri:
                node_vals[node_id] += w * elem_values[elem_id]
                node_weights[node_id] += w
        for i in range(self.n_nodes):
            if node_weights[i] > 0:
                node_vals[i] /= node_weights[i]
        return node_vals

    def boundary_edges(self) -> np.ndarray:
        r"""
        提取边界边。
        边界边恰好只属于一个三角形单元。
        """
        if self._boundary_edges is not None:
            return self._boundary_edges
        edge_count = {}
        for tri in self.elements:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                key = (min(e), max(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary = [np.array(e) for e, cnt in edge_count.items() if cnt == 1]
        self._boundary_edges = np.array(boundary, dtype=int) if boundary else np.zeros((0, 2), dtype=int)
        return self._boundary_edges

    def find_bottom_boundary_nodes(self, tol: float = 1e-8) -> np.ndarray:
        r"""
        找到 y 坐标最小的边界节点（假设为接触边界）。
        用于 Signorini 接触条件施加。
        """
        y_min = np.min(self.nodes[:, 1])
        mask = np.abs(self.nodes[:, 1] - y_min) < tol
        return np.where(mask)[0]

    def shape_functions_gradients(self, elem_id: int) -> np.ndarray:
        r"""
        计算单元 elem_id 上三个线性形函数的梯度（常数）。
        对于节点 (x1,y1), (x2,y2), (x3,y3)：
        \nabla N_1 = (1/(2A)) * [ y2-y3, x3-x2 ]^T
        \nabla N_2 = (1/(2A)) * [ y3-y1, x1-x3 ]^T
        \nabla N_3 = (1/(2A)) * [ y1-y2, x2-x1 ]^T
        返回 (3, 2) 数组。
        """
        tri = self.elements[elem_id]
        x = self.nodes[tri, 0]
        y = self.nodes[tri, 1]
        A2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(A2) < 1e-20:
            raise ValueError(f"Degenerate element {elem_id}")
        dN = np.zeros((3, 2))
        dN[0, 0] = (y[1] - y[2]) / A2
        dN[0, 1] = (x[2] - x[1]) / A2
        dN[1, 0] = (y[2] - y[0]) / A2
        dN[1, 1] = (x[0] - x[2]) / A2
        dN[2, 0] = (y[0] - y[1]) / A2
        dN[2, 1] = (x[1] - x[0]) / A2
        return dN

    def integrate_over_elements(self, elem_integrand: np.ndarray) -> float:
        r"""
        在网格上积分：
        I = \sum_e A_e * f_e
        其中 f_e 是单元常数。
        """
        areas = np.abs(self.compute_areas())
        return float(np.dot(areas, elem_integrand))


def generate_rectangular_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                               shift_y: float = 0.0) -> TriMesh2D:
    r"""
    生成矩形区域 [0, lx] x [shift_y, shift_y+ly] 的三角网格。
    使用对角线切分每个四边形为两个三角形。
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(shift_y, shift_y + ly, ny)
    nodes = []
    node_id = {}
    for j in range(ny):
        for i in range(nx):
            nodes.append([x[i], y[j]])
            node_id[(i, j)] = len(nodes) - 1
    nodes = np.array(nodes)
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = node_id[(i, j)]
            n10 = node_id[(i + 1, j)]
            n01 = node_id[(i, j + 1)]
            n11 = node_id[(i + 1, j + 1)]
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])
    elements = np.array(elements, dtype=int)
    return TriMesh2D(nodes, elements)


def parse_stla_surface(stla_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    简化版 ASCII STL 解析器。
    融合自 1167_stla_to_tri_surface 的 STL 读取逻辑。
    返回 (nodes, faces)，其中 faces 是三角形索引（0-based）。
    """
    vertices = []
    normals = []
    state = 0
    current_normal = np.zeros(3)
    for line in stla_lines:
        line = line.strip().upper()
        if not line or line.startswith("SOLID") or line.startswith("ENDSOLID"):
            continue
        if line.startswith("FACET NORMAL"):
            parts = line.split()
            if len(parts) >= 6:
                current_normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            state = 1
        elif line.startswith("VERTEX"):
            parts = line.split()
            if len(parts) >= 4:
                v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                vertices.append(v)
                normals.append(current_normal.copy())
            state = 2
    if len(vertices) % 3 != 0:
        raise ValueError("Incomplete STL data: vertex count not divisible by 3")
    n_faces = len(vertices) // 3
    # 去重节点
    tol = 1e-8
    unique_nodes = []
    node_map = {}
    for v in vertices:
        key = tuple(np.round(v / tol).astype(int))
        if key not in node_map:
            node_map[key] = len(unique_nodes)
            unique_nodes.append(v)
    faces = np.zeros((n_faces, 3), dtype=int)
    for f in range(n_faces):
        for k in range(3):
            v = vertices[f * 3 + k]
            key = tuple(np.round(v / tol).astype(int))
            faces[f, k] = node_map[key]
    nodes = np.array(unique_nodes)
    return nodes, faces
