
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Callable





def t6_basis_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.zeros(6)
    dphi_dxi = np.zeros(6)
    dphi_deta = np.zeros(6)

    phi[0] = (1.0 - xi - eta) * (1.0 - 2.0 * xi - 2.0 * eta)
    phi[1] = xi * (2.0 * xi - 1.0)
    phi[2] = eta * (2.0 * eta - 1.0)
    phi[3] = 4.0 * xi * (1.0 - xi - eta)
    phi[4] = 4.0 * xi * eta
    phi[5] = 4.0 * eta * (1.0 - xi - eta)

    dphi_dxi[0] = -3.0 + 4.0 * xi + 4.0 * eta
    dphi_dxi[1] = 4.0 * xi - 1.0
    dphi_dxi[2] = 0.0
    dphi_dxi[3] = 4.0 - 8.0 * xi - 4.0 * eta
    dphi_dxi[4] = 4.0 * eta
    dphi_dxi[5] = -4.0 * eta

    dphi_deta[0] = -3.0 + 4.0 * xi + 4.0 * eta
    dphi_deta[1] = 0.0
    dphi_deta[2] = 4.0 * eta - 1.0
    dphi_deta[3] = -4.0 * xi
    dphi_deta[4] = 4.0 * xi
    dphi_deta[5] = 4.0 - 4.0 * xi - 8.0 * eta

    return phi, dphi_dxi, dphi_deta


def t6_physical_derivatives(xi: float, eta: float,
                            x_phys: np.ndarray, y_phys: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, dphi_dxi, dphi_deta = t6_basis_functions(xi, eta)


    dx_dxi = np.dot(dphi_dxi, x_phys)
    dx_deta = np.dot(dphi_deta, x_phys)
    dy_dxi = np.dot(dphi_dxi, y_phys)
    dy_deta = np.dot(dphi_deta, y_phys)

    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
    if abs(detJ) < 1e-14:
        raise ValueError(f"Singular Jacobian in T6 element, detJ = {detJ}")


    dphi_dx = (dy_deta * dphi_dxi - dy_dxi * dphi_deta) / detJ
    dphi_dy = (-dx_deta * dphi_dxi + dx_dxi * dphi_deta) / detJ

    return phi, dphi_dx, dphi_dy





def elastic_matrix_plane_stress(E: float, nu: float) -> np.ndarray:
    if E <= 0:
        raise ValueError("Young's modulus E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Poisson's ratio nu must be in (-1, 0.5).")

    factor = E / (1.0 - nu * nu)
    D = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    return D





def r8blt_sl(n: int, ml: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != (ml + 1, n):
        raise ValueError(f"Matrix shape mismatch: expected ({ml+1}, {n}), got {a.shape}")
    if b.shape != (n,):
        raise ValueError(f"RHS shape mismatch: expected ({n},), got {b.shape}")

    x = b.copy()
    for j in range(n):
        if abs(a[0, j]) < 1e-15:
            raise ValueError(f"Zero diagonal element at row {j}")
        x[j] = x[j] / a[0, j]
        ihi = min(j + ml + 1, n)
        for i in range(j + 1, ihi):
            x[i] = x[i] - a[i - j, j] * x[j]
    return x


def r8blt_mv(n: int, ml: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.zeros(n)
    for i in range(n):
        jlo = max(0, i - ml)
        for j in range(jlo, i + 1):
            y[i] += a[i - j, j] * x[j]
    return y





class ElasticFEM2D:

    def __init__(self, node_xy: np.ndarray, element_node: np.ndarray,
                 element_area: np.ndarray, E_field: np.ndarray,
                 nu: float = 0.3, thickness: float = 1.0):
        self.node_xy = node_xy
        self.element_node = element_node
        self.element_area = element_area
        self.E_field = E_field
        self.nu = nu
        self.thickness = thickness
        self.n_nodes = node_xy.shape[1]
        self.n_elements = element_node.shape[1]
        self.dof_per_node = 2
        self.n_dofs = self.n_nodes * self.dof_per_node


        self.quad_xi = np.array([0.5, 0.5, 0.0])
        self.quad_eta = np.array([0.0, 0.5, 0.5])
        self.quad_w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        self.nq = 3

    def _assemble_stiffness(self) -> csr_matrix:
        n_dofs = self.n_dofs
        n_elements = self.n_elements
        nq = self.nq

        row_indices = []
        col_indices = []
        data = []

        for e in range(n_elements):

            enodes = self.element_node[:, e]
            x_phys = self.node_xy[0, enodes]
            y_phys = self.node_xy[1, enodes]


            E_avg = np.mean(self.E_field[enodes])
            D = elastic_matrix_plane_stress(E_avg, self.nu)

            Ke = np.zeros((12, 12))

            for q in range(nq):
                xi = self.quad_xi[q]
                eta = self.quad_eta[q]
                w = self.quad_w[q]

                phi, dphi_dx, dphi_dy = t6_physical_derivatives(
                    xi, eta, x_phys, y_phys)


                _, dphi_dxi, dphi_deta = t6_basis_functions(xi, eta)
                dx_dxi = np.dot(dphi_dxi, x_phys)
                dx_deta = np.dot(dphi_deta, x_phys)
                dy_dxi = np.dot(dphi_dxi, y_phys)
                dy_deta = np.dot(dphi_deta, y_phys)
                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi


                B = np.zeros((3, 12))
                for i in range(6):
                    B[0, 2 * i] = dphi_dx[i]
                    B[1, 2 * i + 1] = dphi_dy[i]
                    B[2, 2 * i] = dphi_dy[i]
                    B[2, 2 * i + 1] = dphi_dx[i]


                weight = w * abs(detJ) * self.thickness
                Ke += weight * (B.T @ D @ B)


            for i in range(6):
                for j in range(6):
                    gi = enodes[i]
                    gj = enodes[j]

                    row_indices.append(2 * gi)
                    col_indices.append(2 * gj)
                    data.append(Ke[2 * i, 2 * j])

                    row_indices.append(2 * gi)
                    col_indices.append(2 * gj + 1)
                    data.append(Ke[2 * i, 2 * j + 1])

                    row_indices.append(2 * gi + 1)
                    col_indices.append(2 * gj)
                    data.append(Ke[2 * i + 1, 2 * j])

                    row_indices.append(2 * gi + 1)
                    col_indices.append(2 * gj + 1)
                    data.append(Ke[2 * i + 1, 2 * j + 1])

        K = csr_matrix((data, (row_indices, col_indices)), shape=(n_dofs, n_dofs))
        return K

    def _assemble_load(self, body_force: Optional[Callable] = None) -> np.ndarray:
        F = np.zeros(self.n_dofs)


        if body_force is not None:
            for e in range(self.n_elements):
                enodes = self.element_node[:, e]
                x_phys = self.node_xy[0, enodes]
                y_phys = self.node_xy[1, enodes]

                for q in range(self.nq):
                    xi = self.quad_xi[q]
                    eta = self.quad_eta[q]
                    w = self.quad_w[q]

                    phi, _, _ = t6_physical_derivatives(xi, eta, x_phys, y_phys)


                    xq = np.dot(phi, x_phys)
                    yq = np.dot(phi, y_phys)
                    fx, fy = body_force(xq, yq)

                    _, dphi_dxi, dphi_deta = t6_basis_functions(xi, eta)
                    dx_dxi = np.dot(dphi_dxi, x_phys)
                    dx_deta = np.dot(dphi_deta, x_phys)
                    dy_dxi = np.dot(dphi_dxi, y_phys)
                    dy_deta = np.dot(dphi_deta, y_phys)
                    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
                    weight = w * abs(detJ) * self.thickness

                    for i in range(6):
                        F[2 * enodes[i]] += weight * fx * phi[i]
                        F[2 * enodes[i] + 1] += weight * fy * phi[i]

        return F

    def apply_dirichlet_bc(self, K: csr_matrix, F: np.ndarray,
                           bc_nodes: np.ndarray,
                           bc_values: Optional[np.ndarray] = None) -> Tuple[csr_matrix, np.ndarray]:
        K = K.tolil()
        F = F.copy()

        if bc_values is None:
            bc_values = np.zeros(len(bc_nodes) * 2)

        dof_list = []
        for node in bc_nodes:
            dof_list.append(2 * node)
            dof_list.append(2 * node + 1)

        for dof in dof_list:
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = 0.0


        for i, node in enumerate(bc_nodes):
            F[2 * node] = bc_values[2 * i] if 2 * i < len(bc_values) else 0.0
            F[2 * node + 1] = bc_values[2 * i + 1] if 2 * i + 1 < len(bc_values) else 0.0

        return K.tocsr(), F

    def apply_neumann_bc(self, F: np.ndarray, boundary_elements: np.ndarray,
                         traction: np.ndarray) -> np.ndarray:
        F = F.copy()



        for e in boundary_elements:
            enodes = self.element_node[:, e]
            edge_len = 0.0

            for i in range(6):
                n1 = enodes[i]
                n2 = enodes[(i + 1) % 6]
                dx = self.node_xy[0, n1] - self.node_xy[0, n2]
                dy = self.node_xy[1, n1] - self.node_xy[1, n2]
                edge_len += np.sqrt(dx * dx + dy * dy)


            for i in range(6):
                F[2 * enodes[i]] += traction[0] * edge_len / 6.0 * self.thickness
                F[2 * enodes[i] + 1] += traction[1] * edge_len / 6.0 * self.thickness

        return F

    def solve(self, bc_nodes: np.ndarray,
              body_force: Optional[Callable] = None,
              traction: Optional[Tuple[np.ndarray, np.ndarray]] = None
              ) -> np.ndarray:
        K = self._assemble_stiffness()
        F = self._assemble_load(body_force)

        if traction is not None:
            F = self.apply_neumann_bc(F, traction[0], traction[1])

        K, F = self.apply_dirichlet_bc(K, F, bc_nodes)


        U = spsolve(K, F)
        return U

    def compute_strain_energy_density(self, U: np.ndarray) -> np.ndarray:
        sed = np.zeros(self.n_elements)

        for e in range(self.n_elements):
            enodes = self.element_node[:, e]
            x_phys = self.node_xy[0, enodes]
            y_phys = self.node_xy[1, enodes]

            E_avg = np.mean(self.E_field[enodes])
            D = elastic_matrix_plane_stress(E_avg, self.nu)


            xi_c = 1.0 / 3.0
            eta_c = 1.0 / 3.0
            _, dphi_dx, dphi_dy = t6_physical_derivatives(
                xi_c, eta_c, x_phys, y_phys)


            u_e = np.zeros(12)
            for i in range(6):
                u_e[2 * i] = U[2 * enodes[i]]
                u_e[2 * i + 1] = U[2 * enodes[i] + 1]


            epsilon = np.zeros(3)
            for i in range(6):
                epsilon[0] += dphi_dx[i] * u_e[2 * i]
                epsilon[1] += dphi_dy[i] * u_e[2 * i + 1]
                epsilon[2] += dphi_dy[i] * u_e[2 * i] + dphi_dx[i] * u_e[2 * i + 1]

            sed[e] = 0.5 * np.dot(epsilon, D @ epsilon)

        return sed

    def compute_nodal_stress(self, U: np.ndarray) -> np.ndarray:
        nodal_stress = np.zeros((self.n_nodes, 3))
        nodal_count = np.zeros(self.n_nodes)

        for e in range(self.n_elements):
            enodes = self.element_node[:, e]
            x_phys = self.node_xy[0, enodes]
            y_phys = self.node_xy[1, enodes]

            E_avg = np.mean(self.E_field[enodes])
            D = elastic_matrix_plane_stress(E_avg, self.nu)

            xi_c = 1.0 / 3.0
            eta_c = 1.0 / 3.0
            _, dphi_dx, dphi_dy = t6_physical_derivatives(
                xi_c, eta_c, x_phys, y_phys)

            u_e = np.zeros(12)
            for i in range(6):
                u_e[2 * i] = U[2 * enodes[i]]
                u_e[2 * i + 1] = U[2 * enodes[i] + 1]

            epsilon = np.zeros(3)
            for i in range(6):
                epsilon[0] += dphi_dx[i] * u_e[2 * i]
                epsilon[1] += dphi_dy[i] * u_e[2 * i + 1]
                epsilon[2] += dphi_dy[i] * u_e[2 * i] + dphi_dx[i] * u_e[2 * i + 1]

            sigma = D @ epsilon
            for i in range(6):
                nodal_stress[enodes[i], :] += sigma
                nodal_count[enodes[i]] += 1

        for n in range(self.n_nodes):
            if nodal_count[n] > 0:
                nodal_stress[n, :] /= nodal_count[n]

        return nodal_stress
