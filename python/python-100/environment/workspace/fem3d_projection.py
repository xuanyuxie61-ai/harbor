
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM3DProjection:

    def __init__(self, nodes_2d: np.ndarray, elements_2d: np.ndarray, axial_length: float = 0.2):
        self.nodes_2d = np.asarray(nodes_2d, dtype=float)
        self.elements_2d = np.asarray(elements_2d, dtype=int)
        self.axial_length = float(axial_length)
        self.n_node_2d = self.nodes_2d.shape[0]
        self.n_elem_2d = self.elements_2d.shape[0]


        self._build_extruded_mesh()

    def _build_extruded_mesh(self):
        n2 = self.n_node_2d
        L = self.axial_length


        self.nodes_3d = np.zeros((2 * n2, 3))
        self.nodes_3d[:n2, :2] = self.nodes_2d
        self.nodes_3d[n2:, :2] = self.nodes_2d
        self.nodes_3d[n2:, 2] = L


        tetrahedra = []
        for e in range(self.n_elem_2d):
            v = self.elements_2d[e]
            v_bot = v
            v_top = v + n2


            tetrahedra.append([v_bot[0], v_bot[1], v_bot[2], v_top[0]])
            tetrahedra.append([v_bot[1], v_bot[2], v_top[0], v_top[1]])
            tetrahedra.append([v_bot[2], v_top[0], v_top[1], v_top[2]])

        self.elements_3d = np.array(tetrahedra, dtype=int)
        self.n_node_3d = 2 * n2
        self.n_elem_3d = len(tetrahedra)

    def _tetrahedron_volume(self, elem_idx: int) -> float:
        v = self.elements_3d[elem_idx]
        p = self.nodes_3d[v]

        mat = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        return abs(np.linalg.det(mat)) / 6.0

    def assemble_mass_matrix(self) -> csr_matrix:
        row_idx = []
        col_idx = []
        data = []

        for e in range(self.n_elem_3d):
            v = self.elements_3d[e]
            vol = self._tetrahedron_volume(e)
            if vol < 1.0e-18:
                continue

            coeff = vol / 20.0
            for i in range(4):
                for j in range(4):
                    row_idx.append(v[i])
                    col_idx.append(v[j])
                    data.append(coeff * (2.0 if i == j else 1.0))

        M = csr_matrix((data, (row_idx, col_idx)), shape=(self.n_node_3d, self.n_node_3d))
        return M

    def project_2d_to_3d(self, A_2d: np.ndarray) -> np.ndarray:
        if len(A_2d) != self.n_node_2d:
            raise ValueError("2D场节点数与网格不匹配")

        M = self.assemble_mass_matrix()
        b = np.zeros(self.n_node_3d)


        for e in range(self.n_elem_3d):
            v = self.elements_3d[e]
            vol = self._tetrahedron_volume(e)
            if vol < 1.0e-18:
                continue


            p_xy = np.mean(self.nodes_3d[v, :2], axis=0)

            dists = np.sum((self.nodes_2d - p_xy) ** 2, axis=1)
            nearest = np.argmin(dists)
            A_val = A_2d[nearest]


            for i in range(4):
                b[v[i]] += A_val * vol / 4.0


        A_3d = spsolve(M, b)
        if A_3d is None:

            A_3d = np.concatenate([A_2d, A_2d])
        else:
            A_3d = np.asarray(A_3d)

        return A_3d

    def compute_3d_magnetic_energy(self, A_3d: np.ndarray, nu_3d_func) -> float:
        W = 0.0
        for e in range(self.n_elem_3d):
            v = self.elements_3d[e]
            p = self.nodes_3d[v]
            vol = self._tetrahedron_volume(e)
            if vol < 1.0e-18:
                continue


            grads = self._tet4_gradients(e)
            centroid = np.mean(p, axis=0)
            nu_val = nu_3d_func(centroid[0], centroid[1], centroid[2])

            dAdx = sum(A_3d[v[j]] * grads[j, 0] for j in range(4))
            dAdy = sum(A_3d[v[j]] * grads[j, 1] for j in range(4))
            B2 = dAdx * dAdx + dAdy * dAdy
            W += 0.5 * nu_val * B2 * vol

        return W

    def _tet4_gradients(self, elem_idx: int) -> np.ndarray:
        v = self.elements_3d[elem_idx]
        p = self.nodes_3d[v]

        mat = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = np.linalg.det(mat) / 6.0
        if abs(vol) < 1.0e-18:
            return np.zeros((4, 3))

        grads = np.zeros((4, 3))
        for i in range(4):

            idx = [j for j in range(4) if j != i]
            a = p[idx[1]] - p[idx[2]]
            b = p[idx[0]] - p[idx[2]]

            cross = np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])
            grads[i] = cross / (6.0 * vol)


        return grads

    def compute_axial_force_end_effects(self, A_3d: np.ndarray) -> float:
        n2 = self.n_node_2d
        L = self.axial_length


        A_bottom = A_3d[:n2]
        A_top = A_3d[n2:]


        dA_dz = (A_top - A_bottom) / L
        Bz = -dA_dz


        Fz = 0.0
        for e in range(self.n_elem_2d):
            v = self.elements_2d[e]
            p = self.nodes_2d[v]
            area = 0.5 * abs(
                (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
            )
            Bz_avg = np.mean(np.abs(Bz[v]))
            Fz += 0.5 * Bz_avg ** 2 / (4.0 * np.pi * 1.0e-7) * area

        return Fz
