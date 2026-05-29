"""
mesh_topology.py
================================================================================
有限元网格拓扑管理与解析模块

基于种子项目：
  - 749_medit_to_ice：MEDIT 网格格式解析与 NetCDF 写入

科学背景：
  地下水溶质运移的数值模拟通常基于有限元（FEM）或有限差分（FDM）方法，
  需要高质量的计算网格。本模块提供一维线段网格和二维三角形/四边形网格的
  拓扑数据结构，支持：
    - 节点、单元、边界的编号管理
    - 材料分区标记（用于刻画非均质含水层）
    - 网格质量度量（纵横比、最小角、Jacobian）

  一维网格用于解析解对比和快速概念模型；二维网格用于剖面或平面视图上的
  溶质羽流模拟。
================================================================================
"""

import numpy as np
from typing import List, Tuple, Optional


class Mesh1D:
    """
    一维有限元网格：线段单元上的节点分布。

    对于定义在 [x_min, x_max] 上的 1D 对流-弥散方程，
    网格离散为 n_elements 个单元，共 n_nodes = n_elements + 1 个节点。
    """

    def __init__(self, x_min: float, x_max: float, n_elements: int,
                 zone_labels: Optional[np.ndarray] = None):
        if x_min >= x_max:
            raise ValueError("必须满足 x_min < x_max")
        if n_elements < 1:
            raise ValueError("单元数必须 ≥ 1")
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.n_elements = int(n_elements)
        self.n_nodes = n_elements + 1
        self.nodes = np.linspace(x_min, x_max, self.n_nodes)
        self.element_length = (x_max - x_min) / n_elements

        # 材料分区标记：每个节点或单元可属于不同水文地质单元
        if zone_labels is not None:
            if len(zone_labels) != self.n_nodes:
                raise ValueError("zone_labels 长度必须等于节点数")
            self.zone_labels = np.array(zone_labels, dtype=int)
        else:
            self.zone_labels = np.zeros(self.n_nodes, dtype=int)

    def node_index_to_coordinate(self, i: int) -> float:
        if i < 0 or i >= self.n_nodes:
            raise IndexError("节点索引越界")
        return self.nodes[i]

    def element_nodes(self, e: int) -> Tuple[int, int]:
        """返回单元 e 的左、右节点全局编号。"""
        if e < 0 or e >= self.n_elements:
            raise IndexError("单元索引越界")
        return (e, e + 1)

    def element_center(self, e: int) -> float:
        left, right = self.element_nodes(e)
        return 0.5 * (self.nodes[left] + self.nodes[right])

    def refine(self, factor: int = 2) -> "Mesh1D":
        """均匀细化网格，factor 为每个单元细分的份数。"""
        if factor < 2:
            raise ValueError("细化倍数必须 ≥ 2")
        new_ne = self.n_elements * factor
        # 简单插值 zone_labels
        new_labels = np.repeat(self.zone_labels, factor)[:new_ne + 1]
        return Mesh1D(self.x_min, self.x_max, new_ne, new_labels)

    def quality_report(self) -> dict:
        """返回网格质量指标。"""
        h_min = np.min(np.diff(self.nodes))
        h_max = np.max(np.diff(self.nodes))
        return {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "h_min": h_min,
            "h_max": h_max,
            "aspect_ratio": h_max / (h_min + 1e-15),
        }


class Mesh2D:
    """
    二维有限元网格：简化的结构化矩形/三角形网格。

    对于平面二维溶质运移问题，结构化网格由 (nx, ny) 个单元组成，
    节点按行优先排列：node_id = i + j * (nx+1)。
    """

    def __init__(self, xlim: Tuple[float, float], ylim: Tuple[float, float],
                 nx: int, ny: int, triangular: bool = False):
        if nx < 1 or ny < 1:
            raise ValueError("nx, ny 必须 ≥ 1")
        self.xlim = xlim
        self.ylim = ylim
        self.nx = nx
        self.ny = ny
        self.triangular = triangular
        self.n_nodes_x = nx + 1
        self.n_nodes_y = ny + 1
        self.n_nodes = self.n_nodes_x * self.n_nodes_y
        self.n_elements = nx * ny * (2 if triangular else 1)

        # 生成节点坐标
        x = np.linspace(xlim[0], xlim[1], self.n_nodes_x)
        y = np.linspace(ylim[0], ylim[1], self.n_nodes_y)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        self.node_coords = np.column_stack([xv.ravel(), yv.ravel()])

        # 生成单元连通性
        self.elements = self._build_elements()

        # 边界节点标记：0=内部, 1=左, 2=右, 4=下, 8=上
        self.boundary_markers = np.zeros(self.n_nodes, dtype=int)
        for j in range(self.n_nodes_y):
            for i in range(self.n_nodes_x):
                nid = i + j * self.n_nodes_x
                if i == 0:
                    self.boundary_markers[nid] |= 1
                if i == self.n_nodes_x - 1:
                    self.boundary_markers[nid] |= 2
                if j == 0:
                    self.boundary_markers[nid] |= 4
                if j == self.n_nodes_y - 1:
                    self.boundary_markers[nid] |= 8

    def _build_elements(self) -> List[List[int]]:
        elems = []
        for j in range(self.ny):
            for i in range(self.nx):
                n0 = i + j * self.n_nodes_x
                n1 = n0 + 1
                n2 = n0 + self.n_nodes_x
                n3 = n2 + 1
                if self.triangular:
                    elems.append([n0, n1, n3])
                    elems.append([n0, n3, n2])
                else:
                    elems.append([n0, n1, n3, n2])
        return elems

    def element_area(self, e: int) -> float:
        """计算单元面积（三角形或四边形）。"""
        if e < 0 or e >= len(self.elements):
            raise IndexError("单元索引越界")
        conn = self.elements[e]
        coords = self.node_coords[conn, :]
        if len(conn) == 3:
            # 三角形面积 = 0.5 * |cross|
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            return 0.5 * abs(np.cross(v1, v2))
        elif len(conn) == 4:
            # 四边形拆分为两个三角形
            v1 = coords[1] - coords[0]
            v2 = coords[3] - coords[0]
            a1 = 0.5 * abs(np.cross(v1, v2))
            v3 = coords[2] - coords[1]
            v4 = coords[3] - coords[1]
            a2 = 0.5 * abs(np.cross(v3, v4))
            return a1 + a2
        else:
            raise ValueError("Unsupported element type")

    def get_boundary_nodes(self, marker_mask: int) -> np.ndarray:
        """返回具有指定边界标记的节点索引。"""
        return np.where((self.boundary_markers & marker_mask) != 0)[0]

    def quality_report(self) -> dict:
        areas = np.array([self.element_area(e) for e in range(len(self.elements))])
        return {
            "n_nodes": self.n_nodes,
            "n_elements": len(self.elements),
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "total_area": float(np.sum(areas)),
        }


def parse_well_locations(data_lines: List[str]) -> List[dict]:
    """
    解析 ASCII 格式的监测井坐标数据。

    输入格式示例：
        # well_id  x  y  depth
        MW-01  100.0  200.0  45.0
        MW-02  150.0  220.0  50.0
    """
    wells = []
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            well = {
                "well_id": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "depth": float(parts[3]) if len(parts) > 3 else None
            }
            wells.append(well)
        except ValueError:
            continue
    return wells


if __name__ == "__main__":
    m1d = Mesh1D(0.0, 100.0, 20)
    assert m1d.n_nodes == 21
    q1 = m1d.quality_report()
    assert q1["aspect_ratio"] == 1.0

    m2d = Mesh2D((0.0, 10.0), (0.0, 5.0), 4, 2, triangular=True)
    assert m2d.n_elements == 16
    q2 = m2d.quality_report()
    assert q2["total_area"] == 50.0

    wells = parse_well_locations(["MW-01  10.0  20.0  30.0", "# comment", "MW-02  15.0  25.0"])
    assert len(wells) == 2
    print("mesh_topology: 自测试通过")
