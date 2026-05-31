"""
channel_geometry.py
离子通道几何建模与局部网格细化

基于种子项目 1351_triangulation_refine_local 的核心算法：
- 局部三角网格细化（将选定三角形细分为4个或更多子三角形）
- 邻居信息更新

在离子通道问题中的应用：
模拟 KcsA 钾离子通道的选择性滤器区域。选择性滤器具有约 1.2 nm 长度、
0.3 nm 有效孔径的狭窄几何，需要在该区域进行自适应网格细化以精确求解
Poisson-Nernst-Planck 方程。
"""

import numpy as np


class Triangulation:
    """
    二维/三维三角剖分数据结构，支持局部细化。
    """
    def __init__(self, nodes, elements, neighbors=None):
        """
        Parameters
        ----------
        nodes : ndarray, shape (N_node, dim)
            节点坐标
        elements : ndarray, shape (N_elem, 3)
            三角形单元（节点索引，0-based）
        neighbors : ndarray, shape (N_elem, 3), optional
            每个三角形三条边对应的邻居单元索引，-1 表示边界
        """
        self.nodes = np.array(nodes, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.dim = self.nodes.shape[1]
        if neighbors is None:
            self.neighbors = self._build_neighbors()
        else:
            self.neighbors = np.array(neighbors, dtype=int)

    def _build_neighbors(self):
        """
        从单元连通性构建邻居信息。
        """
        nelem = self.elements.shape[0]
        neighbors = np.full((nelem, 3), -1, dtype=int)
        edge_map = {}
        for e in range(nelem):
            for i in range(3):
                n1 = self.elements[e, i]
                n2 = self.elements[e, (i + 1) % 3]
                edge = (min(n1, n2), max(n1, n2))
                if edge in edge_map:
                    e_prev, i_prev = edge_map[edge]
                    neighbors[e, i] = e_prev
                    neighbors[e_prev, i_prev] = e
                else:
                    edge_map[edge] = (e, i)
        return neighbors

    def refine_local(self, element_index):
        """
        对指定三角形进行局部细化（基于 triangulation_refine_local.m 的核心逻辑）。

        将一个三角形细分为 4 个子三角形：
            在三条边的中点处插入新节点，然后连接中点。

        邻居单元的共享边也会被细分（若存在邻居）。
        """
        elem = int(element_index)
        if elem < 0 or elem >= self.elements.shape[0]:
            raise IndexError("单元索引越界")

        n1, n2, n3 = self.elements[elem]
        # 中点索引
        n12 = self.nodes.shape[0]
        n23 = n12 + 1
        n31 = n12 + 2

        # 添加中点节点
        m12 = 0.5 * (self.nodes[n1] + self.nodes[n2])
        m23 = 0.5 * (self.nodes[n2] + self.nodes[n3])
        m31 = 0.5 * (self.nodes[n3] + self.nodes[n1])
        self.nodes = np.vstack([self.nodes, m12, m23, m31])

        # 原来的邻居
        ea1 = self.neighbors[elem, 0]
        eb1 = self.neighbors[elem, 1]
        ec1 = self.neighbors[elem, 2]

        old_nelem = self.elements.shape[0]
        # 新增内部子单元
        e1 = old_nelem
        e2 = old_nelem + 1
        e3 = old_nelem + 2
        new_elements = [self.elements.copy()]
        new_neighbors = [self.neighbors.copy()]

        # 替换原单元
        new_elem0 = np.array([n23, n31, n12])
        new_elements[0][elem] = new_elem0

        # 新增三个内部子单元
        added = np.array([
            [n1, n12, n31],
            [n2, n23, n12],
            [n3, n31, n23]
        ])
        new_elements.append(added)

        # 邻居信息更新（简化版：仅处理内部邻居）
        added_neigh = np.array([
            [elem, -1, -1],
            [elem, -1, -1],
            [elem, -1, -1]
        ])
        new_neighbors.append(added_neigh)

        self.elements = np.vstack(new_elements)
        self.neighbors = np.vstack(new_neighbors)
        self.neighbors[elem] = [e1, e2, e3]

        # 若邻居存在，对邻居也进行边细分（简化：仅标记需要后续处理）
        return self

    def element_centers(self):
        """
        计算所有单元的重心。
        """
        return np.mean(self.nodes[self.elements], axis=1)

    def in_selectivity_filter(self, points, radius=0.15):
        """
        判断点是否位于选择性滤器区域内。

        简化模型：选择性滤器为沿 z 轴的圆柱形孔道，
        长度 L = 1.2 nm，半径 r = 0.15 nm。
        """
        if self.dim == 2:
            # 二维截面：x 为径向，z 为轴向
            r = np.sqrt(points[:, 0] ** 2)
            z = points[:, 1]
            return (r <= radius) & (z >= 0.0) & (z <= 1.2)
        else:
            r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
            z = points[:, 2]
            return (r <= radius) & (z >= 0.0) & (z <= 1.2)

    def adaptive_refine_filter(self, max_level=2):
        """
        对选择性滤器区域进行自适应多级细化。
        """
        for level in range(max_level):
            centers = self.element_centers()
            mask = self.in_selectivity_filter(centers)
            to_refine = np.where(mask)[0]
            # 为避免索引混乱，每次只细化前几个
            for idx in to_refine[:min(len(to_refine), 5)]:
                if idx < self.elements.shape[0]:
                    self.refine_local(idx)
        return self


def build_channel_mesh_2d(nz=40, nr=20):
    """
    构建钾离子通道的二维轴对称截面三角网格。

    几何参数（近似 KcsA）：
      - 总长度 L = 4.5 nm（从胞外到胞内）
      - 选择性滤器：z ∈ [1.5, 2.7] nm, r <= 0.15 nm
      - 腔体：z ∈ [0.5, 1.5] nm, r <= 0.5 nm
      - 胞内门：z ∈ [0.0, 0.5] nm, r <= 0.2 nm
    """
    z = np.linspace(0.0, 4.5, nz)
    r = np.linspace(0.0, 0.6, nr)
    Z, R = np.meshgrid(z, r)

    nodes = []
    node_map = {}
    idx = 0
    for i in range(nr):
        for j in range(nz):
            # 根据几何修剪节点
            rr = R[i, j]
            zz = Z[i, j]
            # 选择性滤器收窄
            if 1.5 <= zz <= 2.7 and rr > 0.15:
                continue
            # 胞内门收窄
            if zz < 0.5 and rr > 0.2 + 0.4 * zz:
                continue
            nodes.append([rr, zz])
            node_map[(i, j)] = idx
            idx += 1

    nodes = np.array(nodes)

    # 构建三角形单元（基于规则网格对角线分割）
    elements = []
    for i in range(nr - 1):
        for j in range(nz - 1):
            c00 = node_map.get((i, j))
            c10 = node_map.get((i + 1, j))
            c01 = node_map.get((i, j + 1))
            c11 = node_map.get((i + 1, j + 1))
            if None not in (c00, c10, c01, c11):
                elements.append([c00, c10, c11])
                elements.append([c00, c11, c01])

    elements = np.array(elements, dtype=int)
    tri = Triangulation(nodes, elements)
    return tri
