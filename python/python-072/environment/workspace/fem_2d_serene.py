
import numpy as np


class SerendipityElement:

    def __init__(self):

        self.quad_xi = np.array([
            -0.7745966692414834,
             0.0,
             0.7745966692414834
        ])
        self.quad_w = np.array([
            0.5555555555555556,
            0.8888888888888889,
            0.5555555555555556
        ])
        self.quad_num = 3

    def basis_functions(self, xi, eta):
        N = np.zeros(8)
        N[0] = (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0) * 0.25
        N[1] = (1.0 - xi ** 2) * (1.0 + eta) * 0.5
        N[2] = (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0) * 0.25
        N[3] = (1.0 - xi) * (1.0 - eta ** 2) * 0.5
        N[4] = (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0) * 0.25
        N[5] = (1.0 - xi ** 2) * (1.0 - eta) * 0.5
        N[6] = (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0) * 0.25
        N[7] = (1.0 + xi) * (1.0 - eta ** 2) * 0.5
        return N

    def basis_derivatives(self, xi, eta):
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)

        dN_dxi[0] = (1.0 + eta) * (2.0 * xi + eta) * 0.25
        dN_dxi[1] = -xi * (1.0 + eta)
        dN_dxi[2] = (1.0 + eta) * (2.0 * xi - eta) * (-0.25)
        dN_dxi[3] = -(1.0 - eta ** 2) * 0.5
        dN_dxi[4] = (1.0 - eta) * (2.0 * xi + eta) * (-0.25)
        dN_dxi[5] = -xi * (1.0 - eta)
        dN_dxi[6] = (1.0 - eta) * (2.0 * xi - eta) * 0.25
        dN_dxi[7] = (1.0 - eta ** 2) * 0.5

        dN_deta[0] = (1.0 + xi) * (xi + 2.0 * eta) * 0.25
        dN_deta[1] = (1.0 - xi ** 2) * 0.5
        dN_deta[2] = (1.0 - xi) * (-xi + 2.0 * eta) * 0.25
        dN_deta[3] = (1.0 - xi) * (-eta)
        dN_deta[4] = (1.0 - xi) * (-xi - 2.0 * eta) * (-0.25)
        dN_deta[5] = (1.0 - xi ** 2) * (-0.5)
        dN_deta[6] = (1.0 + xi) * (xi - 2.0 * eta) * (-0.25)
        dN_deta[7] = (1.0 + xi) * (-eta)

        return dN_dxi, dN_deta

    def jacobian_and_derivatives(self, xi, eta, x_nodes, y_nodes):
        dN_dxi, dN_deta = self.basis_derivatives(xi, eta)

        dx_dxi = np.dot(dN_dxi, x_nodes)
        dy_dxi = np.dot(dN_dxi, y_nodes)
        dx_deta = np.dot(dN_deta, x_nodes)
        dy_deta = np.dot(dN_deta, y_nodes)

        detJ = dx_dxi * dy_deta - dx_deta * dy_dxi

        if abs(detJ) < 1e-14:
            raise ValueError("Jacobian 行列式接近零，单元退化")

        inv_detJ = 1.0 / detJ
        dN_dx = inv_detJ * (dN_dxi * dy_deta - dN_deta * dy_dxi)
        dN_dy = inv_detJ * (-dN_dxi * dx_deta + dN_deta * dx_dxi)

        return detJ, dN_dx, dN_dy


class FEM2DSerene:

    def __init__(self, nx, ny, x_coords, y_coords):
        if nx % 2 == 0 or ny % 2 == 0:
            raise ValueError("nx 和 ny 必须为奇数")
        if nx < 3 or ny < 3:
            raise ValueError("nx 和 ny 必须至少为 3")

        self.nx = nx
        self.ny = ny
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.element = SerendipityElement()


        self.ex_num = (nx - 1) // 2
        self.ey_num = (ny - 1) // 2


        self.num_nodes = self._compute_node_num()

    def _compute_node_num(self):



        return self.nx * self.ny

    def _global_to_local_node_map(self, ex, ey):
        i0 = 2 * ex
        j0 = 2 * ey


        local_indices = [
            (i0 + 2, j0 + 2),
            (i0 + 1, j0 + 2),
            (i0, j0 + 2),
            (i0, j0 + 1),
            (i0, j0),
            (i0 + 1, j0),
            (i0 + 2, j0),
            (i0 + 2, j0 + 1),
        ]


        node_ids = []
        for i, j in local_indices:
            if 0 <= i < self.nx and 0 <= j < self.ny:
                node_ids.append(i * self.ny + j)
            else:
                raise IndexError("节点索引越界")

        return np.array(node_ids, dtype=int)

    def assemble_system(self, a_func, c_func, f_func):
        K = np.zeros((self.num_nodes, self.num_nodes))
        F = np.zeros(self.num_nodes)

        for ey in range(self.ey_num):
            for ex in range(self.ex_num):

                node_ids = self._global_to_local_node_map(ex, ey)
                x_nodes = np.array([self.x_coords[i // self.ny] for i in node_ids])
                y_nodes = np.array([self.y_coords[i % self.ny] for i in node_ids])


                Ke = np.zeros((8, 8))
                Fe = np.zeros(8)

                for qx in range(self.element.quad_num):
                    xi = self.element.quad_xi[qx]
                    wx = self.element.quad_w[qx]
                    for qy in range(self.element.quad_num):
                        eta = self.element.quad_xi[qy]
                        wy = self.element.quad_w[qy]

                        detJ, dN_dx, dN_dy = self.element.jacobian_and_derivatives(
                            xi, eta, x_nodes, y_nodes
                        )
                        N = self.element.basis_functions(xi, eta)


                        xq = np.dot(N, x_nodes)
                        yq = np.dot(N, y_nodes)

                        aq = a_func(xq, yq)
                        cq = c_func(xq, yq)
                        fq = f_func(xq, yq)

                        wq = wx * wy * abs(detJ)

                        for i in range(8):
                            for j in range(8):
                                Ke[i, j] += wq * (
                                    aq * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]) +
                                    cq * N[i] * N[j]
                                )
                            Fe[i] += wq * fq * N[i]


                for i in range(8):
                    ii = node_ids[i]
                    for j in range(8):
                        jj = node_ids[j]
                        K[ii, jj] += Ke[i, j]
                    F[ii] += Fe[i]

        return K, F

    def apply_dirichlet_bc(self, K, F, bc_nodes, bc_values):
        K = K.copy()
        F = F.copy()

        for node, val in zip(bc_nodes, bc_values):
            K[node, :] = 0.0
            K[:, node] = 0.0
            K[node, node] = 1.0
            F[node] = val

        return K, F

    def solve(self, K, F):
        return np.linalg.solve(K, F)

    def compute_l2_error(self, u, exact_func):
        error_sq = 0.0

        for ey in range(self.ey_num):
            for ex in range(self.ex_num):
                node_ids = self._global_to_local_node_map(ex, ey)
                x_nodes = np.array([self.x_coords[i // self.ny] for i in node_ids])
                y_nodes = np.array([self.y_coords[i % self.ny] for i in node_ids])
                u_nodes = u[node_ids]

                for qx in range(self.element.quad_num):
                    xi = self.element.quad_xi[qx]
                    wx = self.element.quad_w[qx]
                    for qy in range(self.element.quad_num):
                        eta = self.element.quad_xi[qy]
                        wy = self.element.quad_w[qy]

                        detJ, _, _ = self.element.jacobian_and_derivatives(
                            xi, eta, x_nodes, y_nodes
                        )
                        N = self.element.basis_functions(xi, eta)

                        xq = np.dot(N, x_nodes)
                        yq = np.dot(N, y_nodes)
                        uq = np.dot(N, u_nodes)

                        eq = exact_func(xq, yq)
                        wq = wx * wy * abs(detJ)
                        error_sq += wq * (uq - eq) ** 2

        return np.sqrt(error_sq)
