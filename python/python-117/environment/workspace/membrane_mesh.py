"""
membrane_mesh.py
================
膜网格构建与邻接分析模块（源自 seed 755_mesh_etoe 的网格邻接算法）

本模块将无结构网格的 element-to-element 邻接算法应用于**粗粒化磷脂双分子层的
三角化表面网格**。膜被离散为一组三角形单元（elements），每个单元有 3 个顶点。
通过边匹配排序算法，建立单元邻接表，用于后续有限元风格的曲率计算和局部面积守恒。

核心数学：
    1. 三角形面积（Heron 公式）：
        A = sqrt(s (s-a)(s-b)(s-c)),  s = (a+b+c)/2
    2. 高斯曲率近似（角度亏损法）：
        K_G(v) = (2*pi - sum(theta_i)) / A_voronoi(v)
    3. 平均曲率（余切公式）：
        H(v) = 0.5 * sum_{j in N(v)} (cot(alpha_ij) + cot(beta_ij)) * (v - x_j)
"""

import numpy as np
from typing import Tuple, List


class TriangulatedMembrane:
    """
    粗粒化脂质双分子层的三角化表面表示。
    """

    def __init__(self, vertices: np.ndarray, elements: np.ndarray):
        """
        Parameters
        ----------
        vertices : ndarray, shape (n_v, 3)
            顶点坐标（单位：nm）。
        elements : ndarray, shape (n_e, 3)
            每个三角形单元的顶点索引（0-based）。
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_v = self.vertices.shape[0]
        self.n_e = self.elements.shape[0]
        self.e_order = self.elements.shape[1]  # 三角形 = 3
        self._etoe = None
        self._areas = None
        self._normals = None
        self._curvature = None

    def compute_etoe(self) -> np.ndarray:
        """
        计算 element-to-element 邻接矩阵（源自 seed 755_mesh_etoe 核心算法）。

        算法步骤：
        1. 对每个单元的每条边构造记录 (min(v1,v2), max(v1,v2), side, element)；
        2. 按 (min, max) 字典序排序；
        3. 扫描排序后的列表，若相邻记录边顶点相同，则互为邻居；
        4. 无匹配的边标记为 -1（膜边界或孔洞）。

        Returns
        -------
        etoe : ndarray, shape (n_e, e_order)
            etoe[e, s] 表示单元 e 在第 s 条边相邻的单元索引，-1 表示边界。
        """
        e_order = self.e_order
        e_num = self.n_e
        etov = self.elements
        # 构造边记录: (min_v, max_v, side, element)
        records = []
        for e in range(e_num):
            for s in range(e_order):
                v1 = etov[e, s]
                v2 = etov[e, (s + 1) % e_order]
                vmin = min(v1, v2)
                vmax = max(v1, v2)
                records.append((vmin, vmax, s, e))
        # 排序
        records.sort(key=lambda r: (r[0], r[1]))
        etoe = np.full((e_num, e_order), -1, dtype=np.int64)
        i = 0
        while i < len(records):
            j = i + 1
            while j < len(records) and records[j][0] == records[i][0] and records[j][1] == records[i][1]:
                j += 1
            if j - i == 2:
                # 内部边，两个单元共享
                _, _, s1, e1 = records[i]
                _, _, s2, e2 = records[i + 1]
                etoe[e1, s1] = e2
                etoe[e2, s2] = e1
            elif j - i > 2:
                # 非流形边，取前两个匹配，其余忽略（鲁棒性处理）
                _, _, s1, e1 = records[i]
                _, _, s2, e2 = records[i + 1]
                etoe[e1, s1] = e2
                etoe[e2, s2] = e1
            i = j
        self._etoe = etoe
        return etoe

    def compute_element_areas(self) -> np.ndarray:
        """
        计算每个三角形单元的面积。

        向量叉积公式：
            A_e = 0.5 * || (x2 - x1) x (x3 - x1) ||_2
        """
        v = self.vertices
        e = self.elements
        x1 = v[e[:, 0]]
        x2 = v[e[:, 1]]
        x3 = v[e[:, 2]]
        cross = np.cross(x2 - x1, x3 - x1)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        self._areas = areas
        return areas

    def compute_normals(self) -> np.ndarray:
        """
        计算每个单元的外法向量（未归一化）。
        """
        v = self.vertices
        e = self.elements
        x1 = v[e[:, 0]]
        x2 = v[e[:, 1]]
        x3 = v[e[:, 2]]
        normals = np.cross(x2 - x1, x3 - x1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 避免退化三角形导致除零
        self._normals = normals / norms
        return self._normals

    def compute_mean_curvature(self) -> np.ndarray:
        """
        基于余切拉普拉斯（cotangent Laplacian）计算每个顶点的平均曲率近似。

        数学公式：
            Delta_S x_i = sum_{j in N(i)} 0.5*(cot(alpha_ij)+cot(beta_ij))*(x_i - x_j)
            H_i = 0.5 * || Delta_S x_i ||

        其中 alpha_ij 和 beta_ij 是与边 (i,j) 相对的两个角。
        此处采用简化的伞算子（umbrella operator）作为快速近似：
            Delta_S x_i = (1/|N(i)|) * sum_{j in N(i)} (x_i - x_j)
            H_i = 0.5 * || Delta_S x_i ||_2
        """
        adj = [set() for _ in range(self.n_v)]
        for tri in self.elements:
            for k in range(3):
                i = tri[k]
                j = tri[(k + 1) % 3]
                adj[i].add(j)
                adj[j].add(i)
        H = np.zeros(self.n_v, dtype=np.float64)
        for i in range(self.n_v):
            neighbors = list(adj[i])
            if len(neighbors) == 0:
                H[i] = 0.0
                continue
            diff = self.vertices[i] - self.vertices[neighbors]
            laplacian = np.mean(diff, axis=0)
            H[i] = 0.5 * np.linalg.norm(laplacian)
        self._curvature = H
        return H

    def bending_energy(self, kappa: float = 20.0) -> float:
        """
        计算 Helfrich 弹性弯曲能（离散形式）：

            E_bend = (kappa/2) * sum_e A_e * (H_e - H_0)^2

        其中 kappa 为弯曲模量（单位：k_B T），H_0 为自发曲率（这里设为 0，
        对应平面膜）。

        Parameters
        ----------
        kappa : float
            弯曲模量，典型值 10–40 k_B T。

        Returns
        -------
        energy : float
            总弯曲能（单位：k_B T）。
        """
        if self._areas is None:
            self.compute_element_areas()
        if self._curvature is None:
            self.compute_mean_curvature()
        # 将顶点曲率插值到单元质心：取三个顶点平均值
        H0 = 0.0  # 自发曲率
        H_elem = np.mean(self._curvature[self.elements], axis=1)
        energy = 0.5 * kappa * np.sum(self._areas * (H_elem - H0) ** 2)
        return float(energy)

    @classmethod
    def create_planar_sheet(cls, nx: int = 16, ny: int = 16,
                            lx: float = 10.0, ly: float = 10.0) -> "TriangulatedMembrane":
        """
        生成一个位于 z=0 平面的矩形三角化膜片，用于基准测试。

        Parameters
        ----------
        nx, ny : int
            x 和 y 方向的格点数。
        lx, ly : float
            膜片尺寸（nm）。
        """
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        xv, yv = np.meshgrid(x, y)
        vertices = np.column_stack((xv.ravel(), yv.ravel(), np.zeros(nx * ny)))
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = j * nx + i
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v2 + 1
                elements.append([v0, v1, v2])
                elements.append([v1, v3, v2])
        elements = np.array(elements, dtype=np.int64)
        return cls(vertices, elements)
