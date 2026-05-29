"""
fem_solver.py
=============
二维有限元求解器模块

基于种子项目:
  - 416_fem2d_scalar_display_gpl: 二维 FEM 三角网格与数据读写思想

科学背景:
  人工耳蜗电刺激电场满足非齐次 Poisson 方程:
      ∇·(σ ∇V) = -I_e δ(r - r_e)

  其中 σ 为耳蜗组织电导率张量 (S/m)，V 为电势 (V)。
  在二维截面近似下，使用 Galerkin FEM 在三角网格上离散求解。

  弱形式推导:
      对任意测试函数 w ∈ H_0^1(Ω):
      ∫_Ω σ ∇V · ∇w dΩ = ∫_Ω I_e w δ(r-r_e) dΩ = I_e w(r_e)

  离散后得到线性系统:
      K V = F

  其中刚度矩阵 K_ij = ∫_Ω σ ∇φ_i · ∇φ_j dΩ，
  载荷向量 F_i = Σ_e I_e φ_i(r_e)。
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM2DSolver:
    """
    二维三角网格有限元 Poisson 方程求解器。
    """

    def __init__(self, nodes, elements, conductivity=0.3):
        """
        Parameters
        ----------
        nodes : ndarray, shape (N, 2)
            节点坐标 (mm)
        elements : ndarray, shape (M, 3), int
            三角形单元，每行三个节点索引 (0-based)
        conductivity : float or ndarray
            电导率 σ (S/m)。若为标量则均匀，若为 (M,) 则按单元变分。
        """
        nodes = np.asarray(nodes, dtype=float)
        elements = np.asarray(elements, dtype=int)

        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must be of shape (N, 2)")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError("elements must be of shape (M, 3)")
        if np.any(elements < 0) or np.any(elements >= nodes.shape[0]):
            raise ValueError("elements 包含越界节点索引")

        self.nodes = nodes
        self.elements = elements
        self.n_nodes = nodes.shape[0]
        self.n_elements = elements.shape[0]

        if np.isscalar(conductivity):
            self.sigma = np.full(self.n_elements, float(conductivity))
        else:
            self.sigma = np.asarray(conductivity, dtype=float)
            if self.sigma.shape != (self.n_elements,):
                raise ValueError("conductivity 长度必须与单元数相同")

        self._stiffness = None
        self._built = False

    def _build_stiffness_matrix(self):
        """组装全局刚度矩阵。"""
        row_idx = []
        col_idx = []
        data = []

        for elem_idx, elem in enumerate(self.elements):
            pts = self.nodes[elem]  # (3, 2)
            # 单元面积
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area < 1e-14:
                continue

            # 线性形函数梯度 (常数)
            # N_i = a_i + b_i x + c_i y
            # b_i = ∂N_i/∂x, c_i = ∂N_i/∂y
            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)

            # 单元刚度矩阵
            sigma = self.sigma[elem_idx]
            ke = sigma * area * (np.outer(b, b) + np.outer(c, c))

            for i in range(3):
                for j in range(3):
                    row_idx.append(elem[i])
                    col_idx.append(elem[j])
                    data.append(ke[i, j])

        K = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(self.n_nodes, self.n_nodes)
        )
        self._stiffness = K
        self._built = True

    def solve(self, source, dirichlet_nodes=None, dirichlet_values=None):
        """
        求解 Poisson 方程。

        Parameters
        ----------
        source : ndarray, shape (N,)
            源项 (A/mm^2)
        dirichlet_nodes : ndarray or None
            Dirichlet 边界节点索引
        dirichlet_values : ndarray or None
            对应边界值 (V)

        Returns
        -------
        V : ndarray, shape (N,)
            电势分布 (V)
        """
        source = np.asarray(source, dtype=float)
        if source.shape != (self.n_nodes,):
            raise ValueError("source 长度必须与节点数相同")

        if not self._built:
            self._build_stiffness_matrix()

        K = self._stiffness.copy()
        F = source.copy()

        # 施加 Dirichlet 边界条件
        if dirichlet_nodes is not None and dirichlet_values is not None:
            dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=int)
            dirichlet_values = np.asarray(dirichlet_values, dtype=float)
            for idx, val in zip(dirichlet_nodes, dirichlet_values):
                # 消去第 idx 行和列
                F -= K[:, idx].toarray().flatten() * val
                F[idx] = val
                # 修改矩阵
                K[idx, :] = 0.0
                K[:, idx] = 0.0
                K[idx, idx] = 1.0

        # 求解
        V = spsolve(K, F)
        if V is None:
            raise RuntimeError("线性系统求解失败，刚度矩阵可能奇异")
        return V

    def compute_gradient(self, V):
        """
        计算每个单元上的电势梯度 ∇V。

        Parameters
        ----------
        V : ndarray, shape (N,)
            节点电势

        Returns
        -------
        grad : ndarray, shape (M, 2)
            单元梯度 (∂V/∂x, ∂V/∂y)
        """
        V = np.asarray(V, dtype=float)
        grad = np.zeros((self.n_elements, 2))
        for elem_idx, elem in enumerate(self.elements):
            pts = self.nodes[elem]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area < 1e-14:
                continue
            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)
            v_elem = V[elem]
            grad[elem_idx, 0] = np.dot(b, v_elem)
            grad[elem_idx, 1] = np.dot(c, v_elem)
        return grad

    def compute_activation_function(self, V):
        """
        计算激活函数 AF = ∂²V/∂x²，沿蜗轴方向的二阶导数。

        在神经刺激理论中，激活函数正比于细胞外电势沿神经纤维方向的
        二阶空间导数:
            AF(x) = ∂²V/∂x²

        Returns
        -------
        af : ndarray, shape (N,)
            节点激活函数值 (V/mm^2)
        """
        # 使用节点平均梯度近似二阶导
        grad = self.compute_gradient(V)
        # 将单元梯度投影到节点
        node_grad = np.zeros((self.n_nodes, 2))
        node_count = np.zeros(self.n_nodes)
        for elem_idx, elem in enumerate(self.elements):
            for ni in elem:
                node_grad[ni] += grad[elem_idx]
                node_count[ni] += 1
        node_count = np.where(node_count < 1, 1, node_count)
        node_grad /= node_count[:, np.newaxis]

        # TODO: 修复 Hole 1
        # 使用节点邻域最小二乘二次曲面拟合估计激活函数 AF = ∂²V/∂x²
        # 提示:
        #   1. 对每个节点，找其空间最近邻（排除自身）
        #   2. 建立局部坐标偏移 (dx, dy)
        #   3. 构建最小二乘矩阵 A_mat = [dx, dy, dx^2, dy^2, dx*dy]
        #   4. 求解 coeffs = lstsq(A_mat, V[neighbors]-V[i])
        #   5. AF = 2 * coeffs[2]  (二次项系数对应 ∂²V/∂x²)
        af = np.zeros(self.n_nodes)
        raise NotImplementedError("Hole 1: 请实现 compute_activation_function 的最小二乘拟合")

        return af


def generate_cochlea_mesh(geometry, n_radial=20, n_angular=80):
    """
    生成耳蜗鼓阶截面的三角网格。

    Parameters
    ----------
    geometry : CochleaGeometry
    n_radial : int
        径向网格数
    n_angular : int
        角度方向网格数

    Returns
    -------
    nodes : ndarray, shape (N, 2)
    elements : ndarray, shape (M, 3)
    """
    theta = np.linspace(0.0, geometry.theta_max, n_angular)
    nodes_list = []
    node_indices = np.empty((n_angular, n_radial), dtype=int)
    idx = 0
    for ti, th in enumerate(theta):
        center = geometry.centerline_at(th)
        r = geometry.r0 * np.exp(-geometry.b * th)
        # 法向
        dr = -geometry.b * r
        dx = dr * np.cos(th) - r * np.sin(th)
        dy = dr * np.sin(th) + r * np.cos(th)
        tangent = np.array([dx, dy])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-14)
        normal = np.array([-tangent[1], tangent[0]])

        # 径向从内侧到外侧
        for ri in range(n_radial):
            frac = ri / (n_radial - 1)
            offset = -geometry.scala_width / 2 + frac * geometry.scala_width
            pos = center + offset * normal
            nodes_list.append(pos)
            node_indices[ti, ri] = idx
            idx += 1

    nodes = np.array(nodes_list)

    # 生成三角形单元
    elements = []
    for ti in range(n_angular - 1):
        for ri in range(n_radial - 1):
            n1 = node_indices[ti, ri]
            n2 = node_indices[ti + 1, ri]
            n3 = node_indices[ti, ri + 1]
            n4 = node_indices[ti + 1, ri + 1]
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])

    elements = np.array(elements, dtype=int)
    return nodes, elements
