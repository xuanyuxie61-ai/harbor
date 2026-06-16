"""
domain_decomposition.py
=======================
恒星内部域分解: 三角形网格生成与 6 节点高阶元.

融合种子项目:
  - polygon_triangulate (890): 耳切法多边形三角化
  - triangulation_order6_contour (1343): 6 节点三角形单元
  - FEM2D (412): 有限元组装

科学背景
--------
对于非径向振荡模式, 需要在二维/三维域上求解:

  球壳域: (r, θ) ∈ [0, R] × [0, π]
  或完整的球坐标: (r, θ, φ)

有限元方法将域分解为三角形单元:
  线性元 (3 节点): C⁰ 连续
  二次元 (6 节点): C⁰ 连续, 更高精度

6 节点三角形 (from 1343):
  顶点: 1, 2, 3
  边中点: 4 (边 1-2), 5 (边 2-3), 6 (边 3-1)

形函数:
  N₁ = λ₁(2λ₁ - 1)
  N₂ = λ₂(2λ₂ - 1)
  N₃ = λ₃(2λ₃ - 1)
  N₄ = 4λ₁λ₂
  N₅ = 4λ₂λ₃
  N₆ = 4λ₃λ₁

  其中 λ_i 是面积坐标.

耳切法三角化 (from 890):
  对简单多边形, 反复切掉 "耳朵" (三角形):
  1. 找到一个耳朵顶点 v: 对角线 (v_prev, v_next) 完全在多边形内
  2. 切掉三角形 (v_prev, v, v_next)
  3. 更新多边形, 重复直到只剩 3 个顶点

本模块实现:
  1. 球壳域的三角形网格生成
  2. 6 节点二次元的形函数
  3. 有限元刚度/质量矩阵组装
  4. 耳切法多边形三角化
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


class SphericalShellMesh:
    """
    球壳域的三角形网格.

    在 (r, θ) 平面上生成结构化网格, 用于求解
    非径向振荡的二维本征值问题.

    参数
    ----
    n_r : int
        径向网格点数
    n_theta : int
        角度网格点数
    r_min : float
        最小半径 (相对 R_*)
    r_max : float
        最大半径 (相对 R_*)
    """

    def __init__(
        self,
        n_r: int = 20,
        n_theta: int = 30,
        r_min: float = 0.01,
        r_max: float = 1.0,
    ):
        if n_r < 3 or n_theta < 3:
            raise ValueError(f"网格点数不足: n_r={n_r}, n_theta={n_theta}")

        self.n_r = n_r
        self.n_theta = n_theta
        self.r_min = r_min
        self.r_max = r_max

        # 生成网格点
        self.r = np.linspace(r_min, r_max, n_r)
        self.theta = np.linspace(0.0, np.pi, n_theta)

        self.nodes = self._generate_nodes()
        self.elements_3 = self._generate_3node_elements()
        self.elements_6 = self._generate_6node_elements()

    def _generate_nodes(self) -> np.ndarray:
        """
        生成 (r, θ) 网格节点.

        Returns
        -------
        nodes : ndarray, shape (n_r * n_theta, 2)
            节点坐标 (r, θ)
        """
        nodes = []
        for j in range(self.n_theta):
            for i in range(self.n_r):
                nodes.append([self.r[i], self.theta[j]])
        return np.array(nodes)

    def _generate_3node_elements(self) -> np.ndarray:
        """
        生成 3 节点三角形单元.

        每个矩形格子分为 2 个三角形.

        Returns
        -------
        elements : ndarray, shape (n_elem, 3)
            单元节点索引
        """
        elements = []
        for j in range(self.n_theta - 1):
            for i in range(self.n_r - 1):
                # 节点编号
                n1 = i + j * self.n_r + 1       # 左下
                n2 = i + 1 + j * self.n_r + 1   # 右下
                n3 = i + (j + 1) * self.n_r + 1  # 左上
                n4 = i + 1 + (j + 1) * self.n_r + 1  # 右上

                # 两个三角形
                elements.append([n1, n2, n3])
                elements.append([n2, n4, n3])

        return np.array(elements)

    def _generate_6node_elements(self) -> np.ndarray:
        """
        生成 6 节点三角形单元 (二次).

        在每条边的中点添加节点.

        Returns
        -------
        elements : ndarray, shape (n_elem, 6)
            单元节点索引 (顶点 + 边中点)
        """
        n_base = len(self.nodes)
        edge_midpoints = {}
        new_nodes = list(self.nodes)
        elements_6 = []

        for elem in self.elements_3:
            n1, n2, n3 = elem

            # 找或创建边中点
            mid_nodes = []
            for edge in [(n1, n2), (n2, n3), (n3, n1)]:
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_midpoints:
                    # 计算中点坐标
                    p1 = self.nodes[edge[0] - 1]
                    p2 = self.nodes[edge[1] - 1]
                    mid = (p1 + p2) / 2.0
                    new_nodes.append(mid)
                    edge_midpoints[edge_key] = len(new_nodes)

                mid_nodes.append(edge_midpoints[edge_key])

            # 6 节点单元: [n1, n2, n3, mid12, mid23, mid31]
            elements_6.append([n1, n2, n3] + mid_nodes)

        self.nodes_6 = np.array(new_nodes)
        return np.array(elements_6)

    def compute_element_area(self, elem_3: np.ndarray) -> float:
        """
        计算三角形面积.

        A = 1/2 |x₁(y₂ - y₃) + x₂(y₃ - y₁) + x₃(y₁ - y₂)|

        Parameters
        ----------
        elem_3 : ndarray, shape (3,)
            3 个节点索引

        Returns
        -------
        area : float
        """
        n1, n2, n3 = elem_3
        x1, y1 = self.nodes[n1 - 1]
        x2, y2 = self.nodes[n2 - 1]
        x3, y3 = self.nodes[n3 - 1]
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


class QuadraticShapeFunctions:
    """
    6 节点三角形二次形函数.

    融合 triangulation_order6_contour (1343) 和 FEM2D (412).

    面积坐标: λ₁, λ₂, λ₃ (λ₁ + λ₂ + λ₃ = 1)

    形函数:
      N₁ = λ₁(2λ₁ - 1)
      N₂ = λ₂(2λ₂ - 1)
      N₃ = λ₃(2λ₃ - 1)
      N₄ = 4λ₁λ₂
      N₅ = 4λ₂λ₃
      N₆ = 4λ₃λ₁

    梯度 (关于面积坐标):
      ∂N₁/∂λ₁ = 4λ₁ - 1
      ∂N₄/∂λ₁ = 4λ₂
      ...

    参数
    ----
    element_coords : ndarray, shape (6, 2)
        6 个节点的坐标
    """

    def __init__(self, element_coords: np.ndarray):
        self.coords = element_coords

    def area_coordinates(self, point: np.ndarray) -> np.ndarray:
        """
        计算点相对于三角形顶点的面积坐标.

        λ₁ = A₂₃ / A, λ₂ = A₃₁ / A, λ₃ = A₁₂ / A

        其中 A_i 是点与对边组成的三角形面积.

        Parameters
        ----------
        point : ndarray, shape (2,)
            计算点坐标

        Returns
        -------
        lam : ndarray, shape (3,)
            面积坐标 [λ₁, λ₂, λ₃]
        """
        v1 = self.coords[0]
        v2 = self.coords[1]
        v3 = self.coords[2]

        # 总面积
        A = 0.5 * ((v2[0] - v1[0]) * (v3[1] - v1[1])
                    - (v3[0] - v1[0]) * (v2[1] - v1[1]))

        if abs(A) < 1e-30:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # 面积坐标
        lam1 = ((v2[0] - point[0]) * (v3[1] - point[1])
                 - (v3[0] - point[0]) * (v2[1] - point[1])) / (2.0 * A)
        lam2 = ((v3[0] - point[0]) * (v1[1] - point[1])
                 - (v1[0] - point[0]) * (v3[1] - point[1])) / (2.0 * A)
        lam3 = 1.0 - lam1 - lam2

        return np.array([lam1, lam2, lam3])

    def shape_values(self, lam: np.ndarray) -> np.ndarray:
        """
        计算 6 个形函数的值.

        Parameters
        ----------
        lam : ndarray, shape (3,)
            面积坐标

        Returns
        -------
        N : ndarray, shape (6,)
            形函数值
        """
        l1, l2, l3 = lam
        N = np.zeros(6)
        N[0] = l1 * (2.0 * l1 - 1.0)
        N[1] = l2 * (2.0 * l2 - 1.0)
        N[2] = l3 * (2.0 * l3 - 1.0)
        N[3] = 4.0 * l1 * l2
        N[4] = 4.0 * l2 * l3
        N[5] = 4.0 * l3 * l1
        return N

    def shape_derivatives(self, lam: np.ndarray) -> np.ndarray:
        """
        计算形函数关于面积坐标的导数.

        Parameters
        ----------
        lam : ndarray, shape (3,)

        Returns
        -------
        dN : ndarray, shape (6, 3)
            dN[i, j] = ∂N_i/∂λ_j
        """
        l1, l2, l3 = lam
        dN = np.zeros((6, 3))

        # ∂/∂λ₁, ∂/∂λ₂, ∂/∂λ₃
        dN[0] = [4.0 * l1 - 1.0, 0.0, 0.0]
        dN[1] = [0.0, 4.0 * l2 - 1.0, 0.0]
        dN[2] = [0.0, 0.0, 4.0 * l3 - 1.0]
        dN[3] = [4.0 * l2, 4.0 * l1, 0.0]
        dN[4] = [0.0, 4.0 * l3, 4.0 * l2]
        dN[5] = [4.0 * l3, 0.0, 4.0 * l1]

        return dN


class EarClippingTriangulator:
    """
    耳切法多边形三角化.

    融合 polygon_triangulate (890) 项目:

    算法:
    1. 初始化双向链表表示多边形
    2. 识别所有 "耳朵" (可以切掉的三角形)
    3. 反复切掉耳朵, 更新链表
    4. 直到只剩 3 个顶点

    耳朵判定条件:
    - 顶点 v 是凸的 (内角 < 180°)
    - 对角线 (v_prev, v_next) 不与其他边相交
    - 对角线完全在多边形内部

    参数
    ----
    angle_tol : float
        角度容差 (弧度)
    """

    def __init__(self, angle_tol: float = 5.7e-5):
        self.angle_tol = angle_tol

    def triangulate(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """
        对简单多边形进行三角化.

        Parameters
        ----------
        vertices : ndarray, shape (n, 2)
            多边形顶点坐标 (逆时针)

        Returns
        -------
        triangles : ndarray, shape (n-2, 3)
            三角形顶点索引
        """
        n = len(vertices)
        if n < 3:
            raise ValueError(f"多边形至少需要 3 个顶点, 收到 {n}")

        # 验证面积
        area = self._polygon_area(vertices)
        if area <= 0:
            raise ValueError("多边形面积为零或负 (可能为顺时针)")

        # 验证相邻顶点不重合
        for i in range(n):
            j = (i + 1) % n
            if np.allclose(vertices[i], vertices[j]):
                raise ValueError(f"相邻顶点 {i} 和 {j} 重合")

        # 初始化链表
        prev_node = [(i - 1) % n for i in range(n)]
        next_node = [(i + 1) % n for i in range(n)]

        # 识别耳朵
        ear = [False] * n
        for i in range(n):
            ear[i] = self._is_ear(i, vertices, prev_node, next_node)

        triangles = []
        remaining = n

        i = 0
        while remaining > 3:
            if ear[i]:
                # 切掉耳朵
                i_prev = prev_node[i]
                i_next = next_node[i]

                triangles.append([i_prev, i, i_next])

                # 从链表中移除 i
                next_node[i_prev] = i_next
                prev_node[i_next] = i_prev

                # 更新邻居的耳朵状态
                ear[i_prev] = self._is_ear(
                    i_prev, vertices, prev_node, next_node
                )
                ear[i_next] = self._is_ear(
                    i_next, vertices, prev_node, next_node
                )

                remaining -= 1

            i = next_node[i]
            if i == 0 and remaining > 3:
                # 完整扫描未找到耳朵, 强制推进
                break

        # 最后一个三角形
        i = 0
        while not ear[i] and i < n:
            i = (i + 1) % n

        # 找到剩余的三个顶点
        remaining_verts = []
        visited = set()
        curr = i
        for _ in range(remaining):
            if curr not in visited:
                remaining_verts.append(curr)
                visited.add(curr)
            curr = next_node[curr]
            if len(remaining_verts) >= 3:
                break

        if len(remaining_verts) >= 3:
            triangles.append(remaining_verts[:3])

        return np.array(triangles)

    def _polygon_area(self, vertices: np.ndarray) -> float:
        """计算多边形面积 (Shoelace 公式)."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return 0.5 * area

    def _triangle_area_signed(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
    ) -> float:
        """有符号三角形面积."""
        return 0.5 * ((v2[0] - v1[0]) * (v3[1] - v1[1])
                       - (v3[0] - v1[0]) * (v2[1] - v1[1]))

    def _is_ear(
        self,
        i: int,
        vertices: np.ndarray,
        prev_node: List[int],
        next_node: List[int],
    ) -> bool:
        """检查顶点 i 是否为耳朵."""
        i_prev = prev_node[i]
        i_next = next_node[i]

        v_prev = vertices[i_prev]
        v_curr = vertices[i]
        v_next = vertices[i_next]

        # 检查凸性
        area = self._triangle_area_signed(v_prev, v_curr, v_next)
        if area <= 0:
            return False  # 凹顶点不是耳朵

        # 检查没有其他顶点在三角形内
        j = next_node[i_next]
        while j != i_prev:
            v_j = vertices[j]
            if self._point_in_triangle(v_j, v_prev, v_curr, v_next):
                return False
            j = next_node[j]

        return True

    def _point_in_triangle(
        self,
        p: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
    ) -> bool:
        """检查点 p 是否在三角形 (v1, v2, v3) 内."""
        a1 = self._triangle_area_signed(p, v2, v3)
        a2 = self._triangle_area_signed(v1, p, v3)
        a3 = self._triangle_area_signed(v1, v2, p)
        return a1 >= 0 and a2 >= 0 and a3 >= 0


def assemble_stiffness_matrix(
    mesh: SphericalShellMesh,
    stiffness_func,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    组装有限元刚度矩阵和质量矩阵.

    融合 FEM2D (412) 的组装过程.

    K_{ij} = Σ_e ∫_e c_s² ∇φ_i · ∇φ_j dA
    M_{ij} = Σ_e ∫_e ρ φ_i φ_j dA

    Parameters
    ----------
    mesh : SphericalShellMesh
        网格
    stiffness_func : callable
        计算局部刚度矩阵的函数

    Returns
    -------
    K : ndarray
        全局刚度矩阵
    M : ndarray
        全局质量矩阵
    """
    n_nodes = len(mesh.nodes)
    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))

    for elem in mesh.elements_3:
        n1, n2, n3 = elem
        coords = mesh.nodes[[n1 - 1, n2 - 1, n3 - 1]]

        # 三角形面积
        area = mesh.compute_element_area(elem)
        if area < 1e-30:
            continue

        # 局部刚度矩阵 (3×3)
        # 梯度: ∇φ_i = (1/(2A)) [y_j - y_k, x_k - x_j]
        K_local = np.zeros((3, 3))
        M_local = np.zeros((3, 3))

        for i_loc in range(3):
            i1 = (i_loc + 1) % 3
            i2 = (i_loc + 2) % 3

            ni1 = elem[i1] - 1
            ni2 = elem[i2] - 1

            grad_i = np.array([
                coords[i1, 1] - coords[i2, 1],
                coords[i2, 0] - coords[i1, 0],
            ]) / (2.0 * area)

            for j_loc in range(3):
                j1 = (j_loc + 1) % 3
                j2 = (j_loc + 2) % 3

                nj1 = elem[j1] - 1
                nj2 = elem[j2] - 1

                grad_j = np.array([
                    coords[j1, 1] - coords[j2, 1],
                    coords[j2, 0] - coords[j1, 0],
                ]) / (2.0 * area)

                # 刚度矩阵 (简化: c_s² = 1)
                K_local[i_loc, j_loc] = area / 3.0 * np.dot(grad_i, grad_j)

                # 质量矩阵
                M_local[i_loc, j_loc] = area / 12.0 * (
                    2.0 if i_loc == j_loc else 1.0
                )

        # 组装到全局矩阵
        for i_loc in range(3):
            i_glob = elem[i_loc] - 1
            for j_loc in range(3):
                j_glob = elem[j_loc] - 1
                K[i_glob, j_glob] += K_local[i_loc, j_loc]
                M[i_glob, j_glob] += M_local[i_loc, j_loc]

    return K, M
