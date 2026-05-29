"""
bone_geometry.py
骨骼几何建模与三角剖分模块

融合来源：
- 1354_triangulation_triangle_neighbors: 三角剖分邻接关系计算
- 1199_tec_to_vtk: 网格数据格式转换思想
- 150_cg_lab_triangles: 点到直线有符号距离

科学背景：
骨骼截面可抽象为二维区域 Omega，包含皮质骨(cortical bone)外壳与
松质骨(trabecular bone)内核。通过三角剖分建立有限元网格，利用
点到边界距离区分不同骨组织区域。
"""

import numpy as np
from typing import Tuple, List, Optional


class BoneGeometry:
    """
    骨骼几何模型类。

    生成骨骼截面的二维三角剖分网格，并计算：
    - 三角形邻接关系（用于后处理与边界识别）
    - 节点到边界的距离（用于区分皮质骨/松质骨）
    - 单元面积
    """

    def __init__(self, width: float = 20.0, height: float = 30.0,
                 cortical_thickness: float = 2.0, nx: int = 17, ny: int = 17):
        """
        初始化骨骼几何参数。

        Parameters
        ----------
        width : float
            骨骼截面宽度 (mm)
        height : float
            骨骼截面高度 (mm)
        cortical_thickness : float
            皮质骨厚度 (mm)
        nx, ny : int
            网格控制点数（奇数，保证中边节点存在）
        """
        if nx % 2 == 0 or ny % 2 == 0:
            raise ValueError("nx and ny must be odd for quadratic elements.")
        if width <= 0 or height <= 0 or cortical_thickness <= 0:
            raise ValueError("Geometric dimensions must be positive.")
        if cortical_thickness >= min(width, height) / 2.0:
            raise ValueError("Cortical thickness too large for given bone dimensions.")

        self.width = width
        self.height = height
        self.cortical_thickness = cortical_thickness
        self.nx = nx
        self.ny = ny
        self.element_order = 6  # T6 quadratic triangle

        # 计算节点与单元数量
        self.element_num = (nx - 1) * (ny - 1) * 2
        self.node_num = (2 * nx - 1) * (2 * ny - 1)

        # 生成网格
        self.node_xy = self._generate_nodes()
        self.element_node = self._generate_t6_grid()
        self.element_area = self._compute_element_areas()
        self.triangle_neighbors = self._compute_triangle_neighbors()
        self.node_distances = self._compute_node_boundary_distances()

    # ------------------------------------------------------------------
    # 节点生成（来自 fem2d_poisson_rectangle 的 xy_set 思想）
    # ------------------------------------------------------------------
    def _generate_nodes(self) -> np.ndarray:
        """
        在矩形区域 [0, width] x [0, height] 上生成 T6 二次三角形的节点坐标。

        节点编号顺序：先沿 x 方向，再沿 y 方向。
        粗网格 (nx x ny) 的角节点 + 中边节点构成细网格节点。
        """
        nx, ny = self.nx, self.ny
        node_num = self.node_num
        node_xy = np.zeros((2, node_num))

        # 粗网格步长
        dx = self.width / (nx - 1)
        dy = self.height / (ny - 1)

        # 细网格分辨率
        nx2 = 2 * nx - 1
        ny2 = 2 * ny - 1
        dx2 = self.width / (nx2 - 1)
        dy2 = self.height / (ny2 - 1)

        node = 0
        for j in range(ny2):
            for i in range(nx2):
                node_xy[0, node] = i * dx2
                node_xy[1, node] = j * dy2
                node += 1

        if node != node_num:
            raise RuntimeError(f"Node count mismatch: {node} != {node_num}")
        return node_xy

    # ------------------------------------------------------------------
    # T6 三角形网格生成（来自 408_fem2d_poisson_rectangle 的 grid_t6）
    # ------------------------------------------------------------------
    def _generate_t6_grid(self) -> np.ndarray:
        """
        生成成对的6节点三角形单元。

        每个粗网格矩形被分成两个二次三角形单元。
        """
        nx, ny = self.nx, self.ny
        element_num = self.element_num
        element_order = self.element_order
        element_node = np.zeros((element_order, element_num), dtype=int)

        # 细网格尺寸
        nx2 = 2 * nx - 1

        element = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                # 粗网格左下角节点在细网格中的索引（0-based）
                # 细网格中第 j 行第 i 列的节点索引
                sw = 2 * j * nx2 + 2 * i          # 西南角（细网格）
                se = sw + 2
                nw = sw + 2 * nx2
                ne = nw + 2
                s_mid = sw + 1
                e_mid = se + nx2
                n_mid = nw + 1
                w_mid = sw + nx2
                c_mid = sw + nx2 + 1

                # 第一个三角形（西南-东北方向）
                element_node[0, element] = sw
                element_node[1, element] = se
                element_node[2, element] = nw
                element_node[3, element] = s_mid
                element_node[4, element] = c_mid
                element_node[5, element] = w_mid
                element += 1

                # 第二个三角形（东北-西南方向）
                element_node[0, element] = ne
                element_node[1, element] = nw
                element_node[2, element] = se
                element_node[3, element] = n_mid
                element_node[4, element] = c_mid
                element_node[5, element] = e_mid
                element += 1

        if element != element_num:
            raise RuntimeError(f"Element count mismatch: {element} != {element_num}")
        return element_node

    # ------------------------------------------------------------------
    # 单元面积计算（来自 408_fem2d_poisson_rectangle 的 area_set）
    # ------------------------------------------------------------------
    def _compute_element_areas(self) -> np.ndarray:
        """
        用三个角节点计算每个三角形单元的面积。
        """
        element_num = self.element_num
        element_area = np.zeros(element_num)

        for e in range(element_num):
            i1 = self.element_node[0, e]
            i2 = self.element_node[1, e]
            i3 = self.element_node[2, e]

            x1, y1 = self.node_xy[:, i1]
            x2, y2 = self.node_xy[:, i2]
            x3, y3 = self.node_xy[:, i3]

            area = 0.5 * abs(
                y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)
            )
            if area < 1e-14:
                raise ValueError(f"Degenerate triangle element {e} with area {area}")
            element_area[e] = area

        return element_area

    # ------------------------------------------------------------------
    # 三角形邻接关系（来自 1354_triangulation_triangle_neighbors）
    # ------------------------------------------------------------------
    def _compute_triangle_neighbors(self) -> np.ndarray:
        """
        计算每个三角形的三个邻接三角形索引。

        算法：边匹配法
        1. 对每个三角形生成3条边（端点按字典序排序）
        2. 对所有边按端点对排序
        3. 扫描排序后的列表，相同端点对的两条边属于相邻三角形
        """
        element_num = self.element_num
        element_order = self.element_order
        neighbors = np.full((3, element_num), -1, dtype=int)

        # 边列表: (min_node, max_node, side_index, element_index)
        edge_list = []
        for e in range(element_num):
            # T6 三角形的前3个节点是角节点
            n1 = self.element_node[0, e]
            n2 = self.element_node[1, e]
            n3 = self.element_node[2, e]
            edges = [(n1, n2), (n2, n3), (n3, n1)]
            for side, (a, b) in enumerate(edges):
                edge_list.append((min(a, b), max(a, b), side, e))

        # 按端点对排序
        edge_list.sort(key=lambda t: (t[0], t[1]))

        # 扫描匹配
        i = 0
        while i < len(edge_list):
            j = i + 1
            if j < len(edge_list) and edge_list[i][0] == edge_list[j][0] and \
                    edge_list[i][1] == edge_list[j][1]:
                side1, elem1 = edge_list[i][2], edge_list[i][3]
                side2, elem2 = edge_list[j][2], edge_list[j][3]
                neighbors[side1, elem1] = elem2
                neighbors[side2, elem2] = elem1
                i += 2
            else:
                i += 1

        return neighbors

    # ------------------------------------------------------------------
    # 点到边界距离（来自 150_cg_lab_triangles 的 point_line_distance_signed）
    # ------------------------------------------------------------------
    def _compute_node_boundary_distances(self) -> np.ndarray:
        """
        计算每个节点到骨骼外边界的最近有符号距离。

        骨骼外边界为矩形 [0, width] x [0, height]。
        正值表示在区域内部，负值表示在外部。
        这里用于区分皮质骨（靠近边界）与松质骨（内部）。
        """
        node_num = self.node_num
        distances = np.zeros(node_num)

        w, h = self.width, self.height

        for node in range(node_num):
            x, y = self.node_xy[:, node]

            # 到四条边的距离
            d_left = x
            d_right = w - x
            d_bottom = y
            d_top = h - y

            # 最近边界距离（内部为正）
            distances[node] = min(d_left, d_right, d_bottom, d_top)

        return distances

    def is_cortical(self, node_index: int) -> bool:
        """
        判断节点是否位于皮质骨区域。
        """
        return self.node_distances[node_index] <= self.cortical_thickness

    def classify_nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        将节点分类为皮质骨和松质骨。

        Returns
        -------
        cortical_nodes : np.ndarray
            皮质骨节点索引
        trabecular_nodes : np.ndarray
            松质骨节点索引
        """
        cortical = np.array([i for i in range(self.node_num)
                             if self.is_cortical(i)], dtype=int)
        trabecular = np.array([i for i in range(self.node_num)
                               if not self.is_cortical(i)], dtype=int)
        return cortical, trabecular

    def export_nodes_elements(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        导出节点坐标和单元连接关系（1-based索引，兼容传统有限元格式）。

        融合 1199_tec_to_vtk 的网格数据导出思想。
        """
        return self.node_xy.copy(), self.element_node.copy() + 1

    def compute_half_bandwidth(self) -> int:
        """
        计算刚度矩阵的半带宽。
        """
        nhba = 0
        for e in range(self.element_num):
            for iln in range(self.element_order):
                i = self.element_node[iln, e]
                if i >= 0:
                    for jln in range(self.element_order):
                        j = self.element_node[jln, e]
                        nhba = max(nhba, abs(j - i))
        return nhba


# =====================================================================
# 以下函数为独立的点到直线有符号距离计算（来自 150_cg_lab_triangles）
# =====================================================================
def point_line_distance_signed(p1: np.ndarray, p2: np.ndarray,
                               p: np.ndarray) -> float:
    """
    计算点 p 到过 p1, p2 的直线的有符号距离。

    数学公式：
        方向向量: l_dv = p2 - p1
        法向量:   l_nv = [-l_dv_y, l_dv_x] / ||l_nv||
        有符号距离: d = l_nv^T (p - p1)

    Parameters
    ----------
    p1, p2 : np.ndarray, shape (2,)
        直线上两点
    p : np.ndarray, shape (2,)
        待测点

    Returns
    -------
    float
        有符号距离
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p = np.asarray(p, dtype=float)

    l_dv = p2 - p1
    norm_dv = np.linalg.norm(l_dv)
    if norm_dv < 1e-14:
        raise ValueError("p1 and p2 must be distinct.")

    l_nv = np.array([-l_dv[1], l_dv[0]])
    l_nv = l_nv / np.linalg.norm(l_nv)

    dist = float(np.dot(l_nv, p - p1))
    return dist
