"""
有限元刚度矩阵与质量矩阵组装模块
融合来源: 400_fem2d_bvp_linear (线性元组装 + 高斯积分) + 408_fem2d_poisson_rectangle (二次元 qbf + 边界处理)

功能:
- 基于 T6 二次元组装多铁性 TDGL 方程的有限元离散算子
- 扩散算子 (梯度能项) 的刚度矩阵 K
- 反应算子的质量矩阵 M
- 边界条件处理 (Dirichlet 型固定序参量)
"""

import numpy as np
from typing import Tuple
from multiferroic_mesh import MultiferroicMesh, qbf_t6, generate_quadrature_points
from sparse_matrix_utils import SparseMatrixCOO


class FEMAssembler:
    """
    有限元组装器，针对多铁性 TDGL 方程的二维离散化。
    """

    def __init__(self, mesh: MultiferroicMesh, nq: int = 3):
        self.mesh = mesh
        self.nq = nq
        self.wq, self.xq, self.yq = generate_quadrature_points(mesh, nq)

    def assemble_stiffness_diffusion(self, diffusion_coeff: np.ndarray) -> SparseMatrixCOO:
        """
        组装扩散刚度矩阵 K，对应梯度能项:
            ∫ D ∇φ·∇ψ dΩ
        diffusion_coeff[element] 为各元素扩散系数。

        数学公式:
            K_{ij} = Σ_e Σ_q w_q |J_e| D_e [∂N_i/∂x ∂N_j/∂x + ∂N_i/∂y ∂N_j/∂y]
        """
        coo = SparseMatrixCOO(self.mesh.node_num, self.mesh.node_num)
        for e in range(self.mesh.element_num):
            area = self.mesh.element_area(e)
            D = float(diffusion_coeff[e]) if e < len(diffusion_coeff) else 1.0
            for q in range(self.nq):
                x = self.xq[q, e]
                y = self.yq[q, e]
                w = area * self.wq[q]
                for test in range(self.mesh.element_order):
                    i = self.mesh.element_node[test, e]
                    bi, dbidx, dbidy = qbf_t6(x, y, e, test, self.mesh)
                    for basis in range(self.mesh.element_order):
                        j = self.mesh.element_node[basis, e]
                        bj, dbjdx, dbjdy = qbf_t6(x, y, e, basis, self.mesh)
                        val = D * (dbidx * dbjdx + dbidy * dbjdy) * w
                        if np.isfinite(val):
                            coo.add_entry(i, j, val)
        return coo

    def assemble_mass_matrix(self) -> SparseMatrixCOO:
        """
        组装质量矩阵 M (L2 投影):
            M_{ij} = ∫ N_i N_j dΩ
        """
        coo = SparseMatrixCOO(self.mesh.node_num, self.mesh.node_num)
        for e in range(self.mesh.element_num):
            area = self.mesh.element_area(e)
            for q in range(self.nq):
                x = self.xq[q, e]
                y = self.yq[q, e]
                w = area * self.wq[q]
                for test in range(self.mesh.element_order):
                    i = self.mesh.element_node[test, e]
                    bi, _, _ = qbf_t6(x, y, e, test, self.mesh)
                    for basis in range(self.mesh.element_order):
                        j = self.mesh.element_node[basis, e]
                        bj, _, _ = qbf_t6(x, y, e, basis, self.mesh)
                        val = bi * bj * w
                        if np.isfinite(val):
                            coo.add_entry(i, j, val)
        return coo

    def assemble_reaction_matrix(self, reaction_func) -> Tuple[SparseMatrixCOO, np.ndarray]:
        """
        组装反应项右端向量及其雅可比近似矩阵。
        reaction_func(x, y) -> scalar，为局部反应源项。

        返回:
            J: 反应项对解的线性化雅可比（对角近似）
            F: 右端载荷向量
        """
        F = np.zeros(self.mesh.node_num, dtype=float)
        coo = SparseMatrixCOO(self.mesh.node_num, self.mesh.node_num)
        for e in range(self.mesh.element_num):
            area = self.mesh.element_area(e)
            for q in range(self.nq):
                x = self.xq[q, e]
                y = self.yq[q, e]
                w = area * self.wq[q]
                r = reaction_func(x, y)
                if not np.isfinite(r):
                    r = 0.0
                for test in range(self.mesh.element_order):
                    i = self.mesh.element_node[test, e]
                    bi, _, _ = qbf_t6(x, y, e, test, self.mesh)
                    F[i] += bi * r * w
                    # 对角雅可比近似
                    coo.add_entry(i, i, bi * bi * w * max(r, 0.0))
        return coo, F

    def apply_dirichlet_boundary(self, coo: SparseMatrixCOO, rhs: np.ndarray,
                                  bc_value: np.ndarray):
        """
        施加 Dirichlet 边界条件。
        对边界节点 k: A[k,k]=1, A[k,j≠k]=0, rhs[k]=bc_value[k]
        融合 fem2d_bvp_linear 和 fem2d_poisson_rectangle 中的边界处理思想。
        """
        bc_nodes = np.where(self.mesh.boundary_flags == 1)[0]
        # 构造新的 COO
        new_coo = SparseMatrixCOO(self.mesh.node_num, self.mesh.node_num)
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i in bc_nodes or j in bc_nodes:
                continue
            new_coo.add_entry(i, j, v)
        for k in bc_nodes:
            new_coo.add_entry(k, k, 1.0)
            rhs[k] = bc_value[k] if k < len(bc_value) else 0.0
        return new_coo, rhs

    def assemble_coupled_system(self, D_P: float, D_M: float) -> Tuple[SparseMatrixCOO, SparseMatrixCOO]:
        """
        组装耦合扩散系统:
            [ K_P   0  ]
            [ 0   K_M  ]
        其中 K_P, K_M 分别对应极化和磁化的扩散算子。
        返回两个独立的刚度矩阵。
        """
        diff_P = np.full(self.mesh.element_num, D_P, dtype=float)
        diff_M = np.full(self.mesh.element_num, D_M, dtype=float)
        K_P = self.assemble_stiffness_diffusion(diff_P)
        K_M = self.assemble_stiffness_diffusion(diff_M)
        return K_P, K_M
