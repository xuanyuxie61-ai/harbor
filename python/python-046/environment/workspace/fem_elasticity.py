
import numpy as np
from numerical_quadrature import triangle_gauss_rule
from utils import check_finite, compute_triangle_area
from sparse_matrix import CCSMatrix


class FEMElasticity2D:

    def __init__(self, nodes, elements, E, nu):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self.E = E
        self.nu = nu


        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))



        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.D = factor * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]
        ], dtype=float)


        self.dof_per_node = 2
        self.n_dof = self.n_nodes * self.dof_per_node

    def _t3_basis_derivatives(self, p1, p2, p3):
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
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = dN_dx[i]
            B[1, 2 * i + 1] = dN_dy[i]
            B[2, 2 * i] = dN_dy[i]
            B[2, 2 * i + 1] = dN_dx[i]
        return B

    def assemble_stiffness_matrix(self, use_sparse=True):
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
        M_dense = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elements):
            n1, n2, n3 = self.elements[e]
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            p3 = self.nodes[n3]
            area = compute_triangle_area(p1, p2, p3)
            if area < 1e-14:
                continue

            m_diag = area / 3.0
            for nid in [n1, n2, n3]:
                M_dense[2 * nid, 2 * nid] += m_diag
                M_dense[2 * nid + 1, 2 * nid + 1] += m_diag

        check_finite(M_dense, "assemble_mass_matrix M")
        if use_sparse:
            return CCSMatrix.from_dense(M_dense, tol=1e-12)
        return M_dense

    def apply_dirichlet_bc(self, K, F, bc_nodes, bc_values):
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
        K = self.assemble_stiffness_matrix(use_sparse=False)
        F = np.asarray(force_vector, dtype=float).copy()

        if bc_nodes is not None:
            K, F = self.apply_dirichlet_bc(K, F, bc_nodes, None)


        cond = np.linalg.cond(K)
        if cond > 1e14:

            K += 1e-10 * np.eye(self.n_dof)

        u = np.linalg.solve(K, F)
        check_finite(u, "solve_static u")
        u_mat = u.reshape(self.n_nodes, 2)
        return u_mat

    def compute_stress_at_elements(self, u_mat):
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
            eps = B @ u_e
            sigma = self.D @ eps
            stresses[e] = sigma
        return stresses


class FEM1DBasis:

    @staticmethod
    def local_basis_1d(order, node_x, eval_x):
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
        order = len(node_x)
        phi = FEM1DBasis.local_basis_1d(order, node_x, eval_x)
        return phi @ np.asarray(node_v)
