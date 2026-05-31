"""
fem_elasticity.py
二维弹性力学有限元求解模块。

融合种子项目:
  - 114_box_flow: Navier-Stokes 的 Galerkin 有限元框架（质量矩阵、刚度矩阵、梯度矩阵组装）
  - 373_fem_basis_t3_display: T3 (线性三角形) 基函数及其导数
  - 395_fem1d_pack: 一维有限元局部基函数与插值思想

在 InSAR 形变反演中的应用:
  1. 在断层周围区域建立二维弹性平面应变有限元模型；
  2. 求解 Navier 位移方程（弹性静力学平衡方程）：
        ∇·σ + f = 0
     其中 σ = C : ε 为应力张量，C 为弹性刚度张量；
  3. 组装刚度矩阵 K、质量矩阵 M、梯度矩阵 B；
  4. 施加断层位错作为内部边界条件（位移不连续）。

核心公式:
  应变-位移关系:
      ε_{ij} = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
  本构关系 (平面应变):
      σ_{ij} = λ δ_{ij} ε_{kk} + 2μ ε_{ij}
  其中 λ = E ν / [(1+ν)(1-2ν)], μ = E / [2(1+ν)] 为 Lamé 参数。

  弱形式:
      ∫_Ω σ(u) : ε(v) dΩ = ∫_Ω f·v dΩ + ∫_Γ t·v dΓ
"""

import numpy as np
from numerical_quadrature import triangle_gauss_rule
from utils import check_finite, compute_triangle_area
from sparse_matrix import CCSMatrix


class FEMElasticity2D:
    """
    二维弹性力学平面应变有限元求解器（T3 三角形单元）。
    """

    def __init__(self, nodes, elements, E, nu):
        """
        参数:
            nodes:    (N_node, 2) 节点坐标
            elements: (N_elem, 3) 三角形单元节点索引
            E:  杨氏模量 (Pa)
            nu: 泊松比
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self.E = E
        self.nu = nu

        # Lamé 参数
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))

        # 平面应变弹性矩阵 D (3×3)
        # [σ_xx, σ_yy, σ_xy]^T = D [ε_xx, ε_yy, 2ε_xy]^T
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.D = factor * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]
        ], dtype=float)

        # 全局自由度: 每个节点 2 个自由度 (u_x, u_y)
        self.dof_per_node = 2
        self.n_dof = self.n_nodes * self.dof_per_node

    def _t3_basis_derivatives(self, p1, p2, p3):
        """
        计算 T3 单元上线性基函数的偏导数（常数）。

        面积坐标:
            N_1(x,y) = [(x2*y3 - x3*y2) + (y2-y3)*x + (x3-x2)*y] / (2*A)
            N_2(x,y) = [(x3*y1 - x1*y3) + (y3-y1)*x + (x1-x3)*y] / (2*A)
            N_3(x,y) = [(x1*y2 - x2*y1) + (y1-y2)*x + (x2-x1)*y] / (2*A)

        返回:
            dN_dx: shape (3,)
            dN_dy: shape (3,)
        """
        area = compute_triangle_area(p1, p2, p3)
        if area < 1e-14:
            raise ValueError("_t3_basis_derivatives: triangle area too small")
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        dN_dx = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
        dN_dy = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)
        return dN_dx, dN_dy

    def _build_B_matrix(self, dN_dx, dN_dy):
        """
        构造应变-位移矩阵 B (3 × 6)，用于 T3 单元。
        B @ u_e = [ε_xx, ε_yy, 2ε_xy]^T
        """
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = dN_dx[i]
            B[1, 2 * i + 1] = dN_dy[i]
            B[2, 2 * i] = dN_dy[i]
            B[2, 2 * i + 1] = dN_dx[i]
        return B

    def assemble_stiffness_matrix(self, use_sparse=True):
        """
        组装全局刚度矩阵 K (n_dof × n_dof)。
        K_ij = ∫_Ω B_i^T D B_j dΩ

        对于 T3 单元，B 为常数，因此:
            K_e = A_e * B_e^T * D * B_e
        """
        K_dense = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elements):
            n1, n2, n3 = self.elements[e]
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            p3 = self.nodes[n3]
            area = compute_triangle_area(p1, p2, p3)
            if area < 1e-14:
                continue
            dN_dx, dN_dy = self._t3_basis_derivatives(p1, p2, p3)
            B = self._build_B_matrix(dN_dx, dN_dy)
            Ke = area * (B.T @ self.D @ B)

            # 组装到全局
            local_dofs = []
            for nid in [n1, n2, n3]:
                local_dofs.extend([2 * nid, 2 * nid + 1])
            for i_local, i_global in enumerate(local_dofs):
                for j_local, j_global in enumerate(local_dofs):
                    K_dense[i_global, j_global] += Ke[i_local, j_local]

        check_finite(K_dense, "assemble_stiffness_matrix K")
        if use_sparse:
            return CCSMatrix.from_dense(K_dense, tol=1e-12)
        return K_dense

    def assemble_mass_matrix(self, use_sparse=True):
        """
        组装一致质量矩阵 M (n_dof × n_dof)。
        对于 T3 单元，一致质量矩阵为:
            M_e = (A_e / 12) * [2 1 1; 1 2 1; 1 1 2] 对 u 和 v 分别
        简化为 lumped 质量矩阵以提高稳定性。
        """
        M_dense = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elements):
            n1, n2, n3 = self.elements[e]
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            p3 = self.nodes[n3]
            area = compute_triangle_area(p1, p2, p3)
            if area < 1e-14:
                continue
            # Lumped mass: 每个节点分配 A_e / 3
            m_diag = area / 3.0
            for nid in [n1, n2, n3]:
                M_dense[2 * nid, 2 * nid] += m_diag
                M_dense[2 * nid + 1, 2 * nid + 1] += m_diag

        check_finite(M_dense, "assemble_mass_matrix M")
        if use_sparse:
            return CCSMatrix.from_dense(M_dense, tol=1e-12)
        return M_dense

    def apply_dirichlet_bc(self, K, F, bc_nodes, bc_values):
        """
        施加 Dirichlet 边界条件。
        参数:
            K: 全局刚度矩阵 (稠密或 CCS)
            F: 右端向量 (n_dof,)
            bc_nodes: list of (node_id, dof_id, value)
                      dof_id: 0 for u_x, 1 for u_y
            bc_values: 对应值
        返回:
            K_mod, F_mod
        """
        if isinstance(K, CCSMatrix):
            K_dense = K.to_dense()
        else:
            K_dense = K.copy()
        F_mod = F.copy()

        for node_id, dof_id, value in bc_nodes:
            gdof = 2 * node_id + dof_id
            K_dense[gdof, :] = 0.0
            K_dense[:, gdof] = 0.0
            K_dense[gdof, gdof] = 1.0
            F_mod[gdof] = value

        return K_dense, F_mod

    def solve_static(self, force_vector, bc_nodes=None):
        """
        求解静力学平衡方程 K u = F。

        参数:
            force_vector: (n_dof,) 体力/面力等效节点力
            bc_nodes: Dirichlet BC 列表

        返回:
            u: (n_nodes, 2) 位移场
        """
        K = self.assemble_stiffness_matrix(use_sparse=False)
        F = np.asarray(force_vector, dtype=float).copy()

        if bc_nodes is not None:
            K, F = self.apply_dirichlet_bc(K, F, bc_nodes, None)

        # 检查条件数
        cond = np.linalg.cond(K)
        if cond > 1e14:
            # 添加微小正则化
            K += 1e-10 * np.eye(self.n_dof)

        u = np.linalg.solve(K, F)
        check_finite(u, "solve_static u")
        u_mat = u.reshape(self.n_nodes, 2)
        return u_mat

    def compute_stress_at_elements(self, u_mat):
        """
        计算每个单元中心的应力张量 [σ_xx, σ_yy, σ_xy]。
        """
        stresses = np.zeros((self.n_elements, 3))
        for e in range(self.n_elements):
            n1, n2, n3 = self.elements[e]
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            p3 = self.nodes[n3]
            dN_dx, dN_dy = self._t3_basis_derivatives(p1, p2, p3)
            B = self._build_B_matrix(dN_dx, dN_dy)
            u_e = np.zeros(6)
            for i, nid in enumerate([n1, n2, n3]):
                u_e[2 * i] = u_mat[nid, 0]
                u_e[2 * i + 1] = u_mat[nid, 1]
            eps = B @ u_e  # [ε_xx, ε_yy, 2ε_xy]
            sigma = self.D @ eps
            stresses[e] = sigma
        return stresses


class FEM1DBasis:
    """
    一维有限元局部基函数（融合 fem1d_pack 思想）。
    用于断层深度方向的一维插值与投影。
    """

    @staticmethod
    def local_basis_1d(order, node_x, eval_x):
        """
        计算一维拉格朗日基函数在 eval_x 处的值。
        order: 单元阶数（节点数）
        node_x: 单元节点坐标，shape (order,)
        eval_x: 评估点，shape (n_eval,)
        返回: phi, shape (n_eval, order)
        """
        eval_x = np.atleast_1d(eval_x)
        n_eval = len(eval_x)
        phi = np.zeros((n_eval, order))
        for i in range(order):
            p = np.ones(n_eval)
            for j in range(order):
                if j != i:
                    denom = node_x[i] - node_x[j]
                    if abs(denom) < 1e-14:
                        raise ValueError("local_basis_1d: duplicate nodes")
                    p *= (eval_x - node_x[j]) / denom
            phi[:, i] = p
        return phi

    @staticmethod
    def interpolate_1d(node_x, node_v, eval_x):
        """
        一维有限元插值。
        node_x: 节点坐标
        node_v: 节点值
        eval_x: 评估点
        """
        order = len(node_x)
        phi = FEM1DBasis.local_basis_1d(order, node_x, eval_x)
        return phi @ np.asarray(node_v)
