"""
fem_solver.py
二维有限元求解器模块

基于 P1（T3）和 P2（T6）拉格朗日有限元，
实现从粒子负载到连续场的投影，以及泊松方程的弱形式离散。

核心数学：
    - T3 线性基函数（面积坐标）:
        给定三角形顶点 T = [ (x1,y1), (x2,y2), (x3,y3) ]
        面积 A = 0.5 * |det([[1, x1, y1], [1, x2, y2], [1, x3, y3]])|
        
        phi_1(x,y) = [(x2-x3)(y-y3) - (y2-y3)(x-x3)] / (2A)
        phi_2(x,y) = [(x3-x1)(y-y1) - (y3-y1)(x-x1)] / (2A)
        phi_3(x,y) = [(x1-x2)(y-y2) - (y1-y2)(x-x2)] / (2A)
    
    - 基函数导数:
        dphi_1/dx = (y2 - y3) / (2A),   dphi_1/dy = (x3 - x2) / (2A)
        dphi_2/dx = (y3 - y1) / (2A),   dphi_2/dy = (x1 - x3) / (2A)
        dphi_3/dx = (y1 - y2) / (2A),   dphi_3/dy = (x2 - x1) / (2A)
    
    - 刚度矩阵元素 (Poisson 方程 -laplacian u = f):
        A_{ij} = integral_{Omega} grad(phi_i) . grad(phi_j) dx dy
               = A_e * [dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy]
        
        其中 A_e 为三角形面积（中点求积时权重为 1/3）。
    
    - 质量矩阵（L2投影）:
        M_{ij} = integral_{Omega} phi_i * phi_j dx dy
        
        对 P1 元，质量矩阵的显式公式:
        M_{ii} = A_e / 6,   M_{ij} = A_e / 12  (i != j, 同属一个单元)
    
    - L2 投影方程:
        M * u = b,   其中 b_i = integral f * phi_i dx dy
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
from utils import compute_triangle_area, check_bounds


class FEMSystem:
    """
    二维有限元系统。
    
    管理节点、三角形单元、刚度矩阵、质量矩阵的组装与求解。
    """
    def __init__(self, nodes: np.ndarray, triangles: np.ndarray):
        """
        Parameters
        ----------
        nodes : np.ndarray, shape (n_nodes, 2)
            节点坐标
        triangles : np.ndarray, shape (n_tri, 3)
            三角形单元节点索引（1-based）
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_tri = self.triangles.shape[0]

        # 边界节点检测（位于矩形外边界上的节点）
        self.boundary_nodes = self._detect_boundary_nodes()

    def _detect_boundary_nodes(self) -> np.ndarray:
        """检测位于外矩形边界的节点。"""
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        tol = 1e-9 * max(xmax - xmin, ymax - ymin)
        mask = (
            (np.abs(x - xmin) < tol) | (np.abs(x - xmax) < tol) |
            (np.abs(y - ymin) < tol) | (np.abs(y - ymax) < tol)
        )
        return np.where(mask)[0]

    def basis_t3(self, tri_idx: int, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 T3 单元在点 p 处的基函数值及其导数。
        
        Parameters
        ----------
        tri_idx : int
            三角形索引（0-based）
        p : np.ndarray, shape (n, 2)
            评估点坐标
        
        Returns
        -------
        phi : np.ndarray, shape (3, n)
            基函数值
        dphidx : np.ndarray, shape (3, n)
            x方向导数
        dphidy : np.ndarray, shape (3, n)
            y方向导数
        """
        p = np.asarray(p, dtype=float)
        nodes = self.triangles[tri_idx] - 1  # 转为0-based
        t = self.nodes[nodes]  # shape (3, 2)

        area2 = (t[0, 0] * (t[1, 1] - t[2, 1])
                 + t[1, 0] * (t[2, 1] - t[0, 1])
                 + t[2, 0] * (t[0, 1] - t[1, 1]))

        if abs(area2) < 1e-14:
            raise ValueError(f"Degenerate triangle {tri_idx}: area ~ 0")

        n = p.shape[0]
        phi = np.zeros((3, n), dtype=float)
        dphidx = np.zeros((3, n), dtype=float)
        dphidy = np.zeros((3, n), dtype=float)

        phi[0, :] = ((t[2, 0] - t[1, 0]) * (p[:, 1] - t[1, 1])
                     - (t[2, 1] - t[1, 1]) * (p[:, 0] - t[1, 0]))
        dphidx[0, :] = -(t[2, 1] - t[1, 1])
        dphidy[0, :] = (t[2, 0] - t[1, 0])

        phi[1, :] = ((t[0, 0] - t[2, 0]) * (p[:, 1] - t[2, 1])
                     - (t[0, 1] - t[2, 1]) * (p[:, 0] - t[2, 0]))
        dphidx[1, :] = -(t[0, 1] - t[2, 1])
        dphidy[1, :] = (t[0, 0] - t[2, 0])

        phi[2, :] = ((t[1, 0] - t[0, 0]) * (p[:, 1] - t[0, 1])
                     - (t[1, 1] - t[0, 1]) * (p[:, 0] - t[0, 0]))
        dphidx[2, :] = -(t[1, 1] - t[0, 1])
        dphidy[2, :] = (t[1, 0] - t[0, 0])

        phi /= area2
        dphidx /= area2
        dphidy /= area2

        return phi, dphidx, dphidy

    def assemble_stiffness_matrix(self) -> csr_matrix:
        """
        组装泊松方程的刚度矩阵 A（稀疏）。
        
        A_{ij} = sum_{e} A_e * [dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy]
        
        Returns
        -------
        scipy.sparse.csr_matrix
            (n_nodes, n_nodes) 稀疏刚度矩阵
        """
        row_ind = []
        col_ind = []
        data = []

        for e in range(self.n_tri):
            nodes = self.triangles[e] - 1
            t = self.nodes[nodes]
            area = abs(compute_triangle_area(t[0], t[1], t[2]))
            if area < 1e-14:
                continue

            # 计算导数（在单元上为常数）
            dphi = np.zeros((3, 2))
            dphi[0, 0] = (t[1, 1] - t[2, 1]) / (2.0 * area)
            dphi[0, 1] = (t[2, 0] - t[1, 0]) / (2.0 * area)
            dphi[1, 0] = (t[2, 1] - t[0, 1]) / (2.0 * area)
            dphi[1, 1] = (t[0, 0] - t[2, 0]) / (2.0 * area)
            dphi[2, 0] = (t[0, 1] - t[1, 1]) / (2.0 * area)
            dphi[2, 1] = (t[1, 0] - t[0, 0]) / (2.0 * area)

            # TODO(Hole_2): 组装刚度矩阵元素
            # 数学：对泊松方程 -laplacian(u) = f，弱形式离散后
            #   A_{ij}^{(e)} = A_e * [dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy]
            # 其中 A_e 为三角形面积，dphi 为 P1 基函数的导数（在单元上为常数）
            # 注意：nodes 的索引约定需与 mesh_generator.py 返回的 triangles 一致
            raise NotImplementedError("Hole_2: fem_solver.py stiffness matrix assembly 待实现")

        A = csr_matrix((data, (row_ind, col_ind)), shape=(self.n_nodes, self.n_nodes))
        return A

    def assemble_mass_matrix(self) -> csr_matrix:
        """
        组装质量矩阵 M（一致质量矩阵）。
        
        对 P1 元:
            M_{ii}^{(e)} = A_e / 6
            M_{ij}^{(e)} = A_e / 12  (i != j)
        
        Returns
        -------
        scipy.sparse.csr_matrix
            质量矩阵
        """
        row_ind = []
        col_ind = []
        data = []

        for e in range(self.n_tri):
            nodes = self.triangles[e] - 1
            t = self.nodes[nodes]
            area = abs(compute_triangle_area(t[0], t[1], t[2]))
            if area < 1e-14:
                continue

            for i in range(3):
                for j in range(3):
                    val = area / 12.0 if i != j else area / 6.0
                    row_ind.append(nodes[i])
                    col_ind.append(nodes[j])
                    data.append(val)

        M = csr_matrix((data, (row_ind, col_ind)), shape=(self.n_nodes, self.n_nodes))
        return M

    def project_function_l2(self, f_values: np.ndarray) -> np.ndarray:
        """
        将节点上的函数值进行 L2 投影（此时质量矩阵与向量相乘即可，
        若 f_values 已定义在节点上，则一致质量矩阵作用给出投影系数）。
        
        更一般地，若要求解 u 使得:
            integral u_h * v_h = integral f * v_h   对所有 v_h in V_h
        则离散方程为:
            M * u = b,   b_i = sum_e integral_e f * phi_i
        
        Parameters
        ----------
        f_values : np.ndarray, shape (n_nodes,)
            节点函数值
        
        Returns
        -------
        np.ndarray
            投影系数（若 M 可逆，则 u = M^{-1} * M * f = f）
        """
        M = self.assemble_mass_matrix()
        b = M @ f_values
        # 使用直接求解器
        u = spsolve(M, b)
        return u

    def solve_poisson(self, rhs: np.ndarray,
                      bc_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        求解泊松方程: -laplacian(u) = rhs，带 Dirichlet 边界条件。
        
        弱形式:
            Find u in H_0^1(Omega) such that
            integral grad(u) . grad(v) = integral rhs * v    for all v in H_0^1
        
        离散后:
            A * u = b,   其中 b_i = integral rhs * phi_i
        
        Parameters
        ----------
        rhs : np.ndarray, shape (n_nodes,)
            右端项（定义在节点上）
        bc_values : np.ndarray, optional
            边界节点上的 Dirichlet 值；若为None，则使用0
        
        Returns
        -------
        np.ndarray
            解向量
        """
        A = self.assemble_stiffness_matrix()
        M = self.assemble_mass_matrix()
        b = M @ rhs  # 右端项投影到节点基

        if bc_values is None:
            bc_values = np.zeros(len(self.boundary_nodes))

        u = np.zeros(self.n_nodes, dtype=float)

        # 处理边界条件（直接消去法）
        interior = np.setdiff1d(np.arange(self.n_nodes), self.boundary_nodes)
        A_int = A[interior][:, interior]
        b_int = b[interior] - A[interior][:, self.boundary_nodes] @ bc_values

        u_interior = spsolve(A_int, b_int)
        u[interior] = u_interior
        u[self.boundary_nodes] = bc_values

        return u

    def interpolate_to_points(self, u: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        将有限元解 u 插值到任意点集。
        
        对每个点，找到包含它的三角形，用面积坐标进行线性插值。
        
        Parameters
        ----------
        u : np.ndarray, shape (n_nodes,)
            有限元解
        points : np.ndarray, shape (m, 2)
            插值点
        
        Returns
        -------
        np.ndarray
            插值结果
        """
        points = np.asarray(points, dtype=float)
        m = points.shape[0]
        result = np.zeros(m, dtype=float)

        for pi in range(m):
            px, py = points[pi]
            found = False
            for e in range(self.n_tri):
                nodes = self.triangles[e] - 1
                t = self.nodes[nodes]
                # 使用重心坐标判断点是否在三角形内
                area = compute_triangle_area(t[0], t[1], t[2])
                if abs(area) < 1e-14:
                    continue
                # 子三角形面积（带符号）
                a1 = compute_triangle_area(np.array([px, py]), t[1], t[2])
                a2 = compute_triangle_area(t[0], np.array([px, py]), t[2])
                a3 = compute_triangle_area(t[0], t[1], np.array([px, py]))

                # 重心坐标
                L1 = a1 / area
                L2 = a2 / area
                L3 = a3 / area

                # 检查是否在内部（允许微小数值误差）
                if L1 >= -1e-10 and L2 >= -1e-10 and L3 >= -1e-10 and abs(L1 + L2 + L3 - 1.0) < 1e-8:
                    result[pi] = L1 * u[nodes[0]] + L2 * u[nodes[1]] + L3 * u[nodes[2]]
                    found = True
                    break
            if not found:
                # 回退：找最近节点
                dists = np.sum((self.nodes - np.array([px, py])) ** 2, axis=1)
                nearest = np.argmin(dists)
                result[pi] = u[nearest]

        return result
