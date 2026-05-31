"""
lattice_geometry.py
================================================================================
晶格几何、对称操作与有限元网格离散化。

融合来源：
  - 107_boundary_word_equilateral（多钻石边界词与对称反射）
  - 503_hand_mesh2d（二维网格生成）
  - 372_fem_basis_q4_display（Q4 有限元基函数）

物理背景：
  在量子退火中，自旋通常排列在特定晶格拓扑上（如 Chimera、Pegasus）。
  对于组合优化问题，我们可以将变量嵌入到二维空间中的几何图结构上，
  利用空间局域性降低耦合复杂度。

  本模块实现：
    1) 六边形晶格（honeycomb）的边界词描述与对称操作群 C_{6v}
    2) 基于边界多边形的 Delaunay 三角化网格生成
    3) Q4 双线性有限元基函数，用于离散化连续空间中的量子势场

核心公式：
  六边形晶格基矢：
    a_1 = a (1, 0)
    a_2 = a (1/2, √3/2)

  倒格矢：
    b_1 = (2π/a) (1, -1/√3)
    b_2 = (2π/a) (0,  2/√3)

  Q4 双线性形函数（参考单元 [-1,1]×[-1,1]）：
    N_1(ξ,η) = (1-ξ)(1-η)/4
    N_2(ξ,η) = (1+ξ)(1-η)/4
    N_3(ξ,η) = (1+ξ)(1+η)/4
    N_4(ξ,η) = (1-ξ)(1+η)/4
"""

import numpy as np
from typing import List, Tuple


class HexLattice:
    """
    六边形晶格，用于构造量子退火耦合图的几何嵌入。
    """

    def __init__(self, lattice_constant: float = 1.0):
        self.a = float(lattice_constant)
        self.a1 = np.array([self.a, 0.0])
        self.a2 = np.array([self.a * 0.5, self.a * np.sqrt(3.0) / 2.0])
        # 倒格矢
        self.b1 = (2.0 * np.pi / self.a) * np.array([1.0, -1.0 / np.sqrt(3.0)])
        self.b2 = (2.0 * np.pi / self.a) * np.array([0.0, 2.0 / np.sqrt(3.0)])

    def site_position(self, i: int, j: int) -> np.ndarray:
        """格点 (i,j) 的笛卡尔坐标 R = i a_1 + j a_2"""
        return i * self.a1 + j * self.a2

    def generate_sites(self, n_ring: int) -> np.ndarray:
        """
        生成 n_ring 层六边形蜂窝的格点坐标，总格点数 N = 1 + 3n(n+1)。
        """
        if n_ring < 0:
            raise ValueError("n_ring must be non-negative")
        sites = []
        for q in range(n_ring + 1):
            for s in range(-q, q + 1):
                for t in range(max(-q, -q - s), min(q, q - s) + 1):
                    sites.append(self.site_position(s, t))
        return np.array(sites)

    def neighbor_vectors(self) -> List[np.ndarray]:
        """
        六方晶格的 6 个最近邻矢量：
            δ_0 =  a_2
            δ_1 =  a_1
            δ_2 =  a_1 - a_2
            δ_3 = -a_2
            δ_4 = -a_1
            δ_5 = -a_1 + a_2
        """
        return [
            self.a2,
            self.a1,
            self.a1 - self.a2,
            -self.a2,
            -self.a1,
            -self.a1 + self.a2,
        ]

    def boundary_word_to_path(self, word: str, start: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        将边界词（0-5 六方向编码）转换为顶点路径。

        方向编码：
            0 -> (0,  +1)
            1 -> (+1,  0)
            2 -> (+1, -1)
            3 -> (0,  -1)
            4 -> (-1,  0)
            5 -> (-1, +1)
        """
        dirs = {
            '0': np.array([0, 1]),
            '1': np.array([1, 0]),
            '2': np.array([1, -1]),
            '3': np.array([0, -1]),
            '4': np.array([-1, 0]),
            '5': np.array([-1, 1]),
        }
        pos = np.array(start, dtype=float)
        path = [pos.copy()]
        for ch in word:
            if ch not in dirs:
                raise ValueError(f"Invalid boundary word character: {ch}")
            pos = pos + dirs[ch]
            path.append(pos.copy())
        return np.array(path)

    def reflect_boundary(self, word: str, reflection_type: int = 1) -> str:
        """
        对边界词施加六方晶格对称群的反射操作。

        reflection_type:
            0: 恒等
            1: 关于 I 轴反射  (k -> 5-k)
            2: 关于 J 轴反射  (k -> (k+3) mod 6 的变体)
            3: 中心反演         (k -> (k+3) mod 6)
        """
        c1 = ['0', '1', '2', '3', '4', '5']
        if reflection_type == 0:
            return word
        elif reflection_type == 1:
            c2 = ['5', '4', '3', '2', '1', '0']
        elif reflection_type == 2:
            c2 = ['2', '1', '0', '5', '4', '3']
        elif reflection_type == 3:
            c2 = ['3', '4', '5', '0', '1', '2']
        else:
            raise ValueError("reflection_type must be 0,1,2,3")
        trans = dict(zip(c1, c2))
        return ''.join(trans[ch] for ch in word)

    def coupling_graph_from_geometry(self, sites: np.ndarray,
                                      cutoff_radius: float = 1.1) -> np.ndarray:
        """
        根据几何距离生成耦合图邻接矩阵：
            J_{ij} = 1  if |R_i - R_j| ≤ cutoff_radius * a
        """
        n = sites.shape[0]
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(sites[i] - sites[j])
                if dist <= cutoff_radius * self.a:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj


class Q4Basis:
    """
    Q4 四节点双线性有限元基函数。

    用于在二维空间离散化量子势场 V(x,y)，将连续问题转化为
    图上的离散组合优化问题。
    """

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        """
        参考单元 [-1,1]×[-1,1] 上的四个形函数值。
        """
        if not (-1.0 - 1e-12 <= xi <= 1.0 + 1e-12 and -1.0 - 1e-12 <= eta <= 1.0 + 1e-12):
            raise ValueError("xi, eta must be in [-1,1]")
        N = np.array([
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ])
        return N

    @staticmethod
    def shape_derivatives(xi: float, eta: float) -> np.ndarray:
        """
        形函数对参考坐标 (ξ,η) 的偏导数，返回 4×2 矩阵。
        """
        dN_dxi = np.array([
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ])
        dN_deta = np.array([
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ])
        return np.column_stack((dN_dxi, dN_deta))

    @staticmethod
    def physical_to_reference(nodes: np.ndarray, x: float, y: float,
                               max_iter: int = 20, tol: float = 1e-12) -> Tuple[float, float]:
        """
        使用牛顿迭代将物理坐标 (x,y) 映射到参考坐标 (ξ,η)。

        nodes: 4×2 数组，物理单元四个节点的坐标。
        """
        if nodes.shape != (4, 2):
            raise ValueError("nodes must be shape (4,2)")
        xi, eta = 0.0, 0.0
        for _ in range(max_iter):
            N = Q4Basis.shape_functions(xi, eta)
            dN = Q4Basis.shape_derivatives(xi, eta)
            # 当前物理坐标估计
            x_est = np.dot(N, nodes[:, 0])
            y_est = np.dot(N, nodes[:, 1])
            # 雅可比矩阵
            Jmat = dN.T @ nodes  # 2×2
            detJ = Jmat[0, 0] * Jmat[1, 1] - Jmat[0, 1] * Jmat[1, 0]
            if abs(detJ) < 1e-15:
                raise RuntimeError("Singular Jacobian in inverse mapping")
            # 牛顿修正
            dx = x - x_est
            dy = y - y_est
            dxi = (Jmat[1, 1] * dx - Jmat[0, 1] * dy) / detJ
            deta = (-Jmat[1, 0] * dx + Jmat[0, 0] * dy) / detJ
            xi += dxi
            eta += deta
            if abs(dxi) < tol and abs(deta) < tol:
                break
        else:
            # 未收敛，裁剪到参考单元
            xi = np.clip(xi, -1.0, 1.0)
            eta = np.clip(eta, -1.0, 1.0)
        return float(xi), float(eta)

    @staticmethod
    def interpolate_scalar_field(nodes: np.ndarray, nodal_values: np.ndarray,
                                  x: float, y: float) -> float:
        """
        在 Q4 单元内插值标量场（如量子势 V(x,y)）。
        """
        xi, eta = Q4Basis.physical_to_reference(nodes, x, y)
        N = Q4Basis.shape_functions(xi, eta)
        return float(np.dot(N, nodal_values))


class Mesh2D:
    """
    简易二维三角网格生成器（基于边界多边形的约束 Delaunay 思想简化版）。
    """

    def __init__(self, boundary_vertices: np.ndarray, max_area: float = 0.1):
        if boundary_vertices.ndim != 2 or boundary_vertices.shape[1] != 2:
            raise ValueError("boundary_vertices must be N×2")
        self.boundary = np.array(boundary_vertices, dtype=float)
        self.max_area = float(max_area)
        self.nodes: np.ndarray = np.zeros((0, 2))
        self.triangles: np.ndarray = np.zeros((0, 3), dtype=int)
        self._generate()

    def _generate(self) -> None:
        """
        简化网格生成：在边界框内均匀撒点，保留边界内部点，
        使用 Bowyer-Watson 算法的简化版进行三角化。
        """
        # 边界框
        xmin, ymin = self.boundary.min(axis=0)
        xmax, ymax = self.boundary.max(axis=0)
        # 估算步长
        h = np.sqrt(self.max_area)
        nx = max(int((xmax - xmin) / h) + 1, 2)
        ny = max(int((ymax - ymin) / h) + 1, 2)
        xg = np.linspace(xmin, xmax, nx)
        yg = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xg, yg)
        points = np.column_stack((XX.ravel(), YY.ravel()))

        # 使用射线法判断点是否在多边形内部
        def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
            x, y = pt
            inside = False
            n = poly.shape[0]
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)):
                    xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-15) + x1
                    if x < xinters:
                        inside = not inside
            return inside

        interior = np.array([point_in_polygon(p, self.boundary) for p in points])
        self.nodes = points[interior]
        if self.nodes.shape[0] < 3:
            #  fallback：仅使用边界点
            self.nodes = self.boundary.copy()

        # 简化三角化：使用 scipy 如果可用，否则手动生成
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(self.nodes)
            self.triangles = tri.simplices.astype(int)
        except Exception:
            # 极度简化：将节点排序后连接为扇形三角
            n = self.nodes.shape[0]
            if n >= 3:
                centroid = self.nodes.mean(axis=0)
                angles = np.arctan2(self.nodes[:, 1] - centroid[1],
                                    self.nodes[:, 0] - centroid[0])
                order = np.argsort(angles)
                tris = []
                for i in range(n):
                    tris.append([order[i], order[(i + 1) % n], order[(i + 2) % n]])
                self.triangles = np.array(tris, dtype=int)
            else:
                self.triangles = np.zeros((0, 3), dtype=int)

    def integrate_scalar_over_mesh(self, field_values: np.ndarray) -> float:
        """
        使用三角单元上的常数近似积分标量场：
            ∫_Ω f dA ≈ Σ_T  A_T * (f_i + f_j + f_k)/3
        """
        if field_values.size != self.nodes.shape[0]:
            raise ValueError("field_values size must match number of nodes")
        total = 0.0
        for tri in self.triangles:
            i, j, k = tri
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
            x3, y3 = self.nodes[k]
            area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            avg_val = (field_values[i] + field_values[j] + field_values[k]) / 3.0
            total += area * avg_val
        return float(total)
