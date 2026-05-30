
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class PoissonFEM2D:

    def __init__(self, nx, ny, lx, ly, solid_mask=None):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)

        if solid_mask is None:
            solid_mask = np.zeros((ny, nx), dtype=bool)
        self.solid_mask = solid_mask


        self.n_nodes = nx * ny

        self.id_map = np.arange(self.n_nodes).reshape(ny, nx)


        self.is_free = ~self.solid_mask.flatten()
        self.free_dofs = np.where(self.is_free)[0]
        self.n_free = len(self.free_dofs)


        self._build_mesh_connectivity()
        self._assemble_stiffness_matrix()

    def _build_mesh_connectivity(self):
        nx, ny = self.nx, self.ny
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = self.id_map[j, i]
                n2 = self.id_map[j, i + 1]
                n3 = self.id_map[j + 1, i]
                n4 = self.id_map[j + 1, i + 1]
                elements.append([n1, n2, n3])
                elements.append([n4, n3, n2])
        self.elements = np.array(elements, dtype=int)
        self.n_elements = len(self.elements)


        x = np.linspace(0.0, self.lx, nx)
        y = np.linspace(0.0, self.ly, ny)
        X, Y = np.meshgrid(x, y)
        self.coords = np.column_stack((X.flatten(), Y.flatten()))

    def _local_stiffness(self, x, y):
        B = np.array([
            [x[1] - x[0], x[2] - x[0]],
            [y[1] - y[0], y[2] - y[0]]
        ])
        area = 0.5 * abs(np.linalg.det(B))
        if area < 1e-16:
            return np.zeros((3, 3))


        dN_dxi = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        inv_B_T = np.linalg.inv(B).T
        grad_N = inv_B_T @ dN_dxi

        k_local = area * (grad_N.T @ grad_N)
        return k_local

    def _assemble_stiffness_matrix(self):
        rows = []
        cols = []
        data = []

        for elem in self.elements:
            nodes = elem
            x = self.coords[nodes, 0]
            y = self.coords[nodes, 1]
            k_loc = self._local_stiffness(x, y)

            for a in range(3):
                for b in range(3):
                    rows.append(nodes[a])
                    cols.append(nodes[b])
                    data.append(k_loc[a, b])

        self.K = csr_matrix(
            (data, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        )

    def solve(self, rhs_field, dirichlet_mask, dirichlet_values,
              neumann_edges=None, neumann_flux=None):
        rhs_flat = rhs_field.flatten()
        F = np.zeros(self.n_nodes)





        node_area = np.zeros(self.n_nodes)
        for elem in self.elements:
            nodes = elem
            x = self.coords[nodes, 0]
            y = self.coords[nodes, 1]
            area = 0.5 * abs(
                (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            )
            for n in nodes:
                node_area[n] += area / 3.0

        F = rhs_flat * node_area


        if neumann_edges is not None and neumann_flux is not None:
            for idx, (na, nb) in enumerate(neumann_edges):
                length = np.linalg.norm(self.coords[nb] - self.coords[na])
                flux = neumann_flux[idx] if hasattr(neumann_flux, '__len__') else neumann_flux
                F[na] += 0.5 * flux * length
                F[nb] += 0.5 * flux * length


        d_mask_flat = dirichlet_mask.flatten()
        d_values_flat = dirichlet_values.flatten()
        is_dirichlet = d_mask_flat & (~self.solid_mask.flatten())


        free = self.is_free & (~is_dirichlet)
        free_ids = np.where(free)[0]

        if len(free_ids) == 0:

            psi_flat = np.zeros(self.n_nodes)
            psi_flat[is_dirichlet] = d_values_flat[is_dirichlet]
            psi_flat[self.solid_mask.flatten()] = 0.0
            return psi_flat.reshape(self.ny, self.nx)

        K_ff = self.K[np.ix_(free_ids, free_ids)]
        F_f = F[free_ids].copy()


        dir_ids = np.where(is_dirichlet)[0]
        if len(dir_ids) > 0:
            K_fd = self.K[np.ix_(free_ids, dir_ids)]
            F_f -= K_fd @ d_values_flat[dir_ids]


        try:
            psi_free = spsolve(K_ff, F_f)
        except Exception as e:

            psi_free = np.linalg.lstsq(K_ff.toarray(), F_f, rcond=None)[0]

        psi_flat = np.zeros(self.n_nodes)
        psi_flat[free_ids] = psi_free
        psi_flat[is_dirichlet] = d_values_flat[is_dirichlet]
        psi_flat[self.solid_mask.flatten()] = 0.0

        return psi_flat.reshape(self.ny, self.nx)
