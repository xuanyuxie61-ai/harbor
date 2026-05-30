
import numpy as np
from typing import Tuple
from multiferroic_mesh import MultiferroicMesh, qbf_t6, generate_quadrature_points
from sparse_matrix_utils import SparseMatrixCOO


class FEMAssembler:

    def __init__(self, mesh: MultiferroicMesh, nq: int = 3):
        self.mesh = mesh
        self.nq = nq
        self.wq, self.xq, self.yq = generate_quadrature_points(mesh, nq)

    def assemble_stiffness_diffusion(self, diffusion_coeff: np.ndarray) -> SparseMatrixCOO:
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

                    coo.add_entry(i, i, bi * bi * w * max(r, 0.0))
        return coo, F

    def apply_dirichlet_boundary(self, coo: SparseMatrixCOO, rhs: np.ndarray,
                                  bc_value: np.ndarray):
        bc_nodes = np.where(self.mesh.boundary_flags == 1)[0]

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
        diff_P = np.full(self.mesh.element_num, D_P, dtype=float)
        diff_M = np.full(self.mesh.element_num, D_M, dtype=float)
        K_P = self.assemble_stiffness_diffusion(diff_P)
        K_M = self.assemble_stiffness_diffusion(diff_M)
        return K_P, K_M
