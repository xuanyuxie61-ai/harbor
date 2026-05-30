
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
from utils import compute_triangle_area, check_bounds


class FEMSystem:
    def __init__(self, nodes: np.ndarray, triangles: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_tri = self.triangles.shape[0]


        self.boundary_nodes = self._detect_boundary_nodes()

    def _detect_boundary_nodes(self) -> np.ndarray:
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
        p = np.asarray(p, dtype=float)
        nodes = self.triangles[tri_idx] - 1
        t = self.nodes[nodes]

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
        row_ind = []
        col_ind = []
        data = []

        for e in range(self.n_tri):
            nodes = self.triangles[e] - 1
            t = self.nodes[nodes]
            area = abs(compute_triangle_area(t[0], t[1], t[2]))
            if area < 1e-14:
                continue


            dphi = np.zeros((3, 2))
            dphi[0, 0] = (t[1, 1] - t[2, 1]) / (2.0 * area)
            dphi[0, 1] = (t[2, 0] - t[1, 0]) / (2.0 * area)
            dphi[1, 0] = (t[2, 1] - t[0, 1]) / (2.0 * area)
            dphi[1, 1] = (t[0, 0] - t[2, 0]) / (2.0 * area)
            dphi[2, 0] = (t[0, 1] - t[1, 1]) / (2.0 * area)
            dphi[2, 1] = (t[1, 0] - t[0, 0]) / (2.0 * area)






            raise NotImplementedError("Hole_2: fem_solver.py stiffness matrix assembly 待实现")

        A = csr_matrix((data, (row_ind, col_ind)), shape=(self.n_nodes, self.n_nodes))
        return A

    def assemble_mass_matrix(self) -> csr_matrix:
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
        M = self.assemble_mass_matrix()
        b = M @ f_values

        u = spsolve(M, b)
        return u

    def solve_poisson(self, rhs: np.ndarray,
                      bc_values: Optional[np.ndarray] = None) -> np.ndarray:
        A = self.assemble_stiffness_matrix()
        M = self.assemble_mass_matrix()
        b = M @ rhs

        if bc_values is None:
            bc_values = np.zeros(len(self.boundary_nodes))

        u = np.zeros(self.n_nodes, dtype=float)


        interior = np.setdiff1d(np.arange(self.n_nodes), self.boundary_nodes)
        A_int = A[interior][:, interior]
        b_int = b[interior] - A[interior][:, self.boundary_nodes] @ bc_values

        u_interior = spsolve(A_int, b_int)
        u[interior] = u_interior
        u[self.boundary_nodes] = bc_values

        return u

    def interpolate_to_points(self, u: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        m = points.shape[0]
        result = np.zeros(m, dtype=float)

        for pi in range(m):
            px, py = points[pi]
            found = False
            for e in range(self.n_tri):
                nodes = self.triangles[e] - 1
                t = self.nodes[nodes]

                area = compute_triangle_area(t[0], t[1], t[2])
                if abs(area) < 1e-14:
                    continue

                a1 = compute_triangle_area(np.array([px, py]), t[1], t[2])
                a2 = compute_triangle_area(t[0], np.array([px, py]), t[2])
                a3 = compute_triangle_area(t[0], t[1], np.array([px, py]))


                L1 = a1 / area
                L2 = a2 / area
                L3 = a3 / area


                if L1 >= -1e-10 and L2 >= -1e-10 and L3 >= -1e-10 and abs(L1 + L2 + L3 - 1.0) < 1e-8:
                    result[pi] = L1 * u[nodes[0]] + L2 * u[nodes[1]] + L3 * u[nodes[2]]
                    found = True
                    break
            if not found:

                dists = np.sum((self.nodes - np.array([px, py])) ** 2, axis=1)
                nearest = np.argmin(dists)
                result[pi] = u[nearest]

        return result
