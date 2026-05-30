
import numpy as np
from typing import List, Tuple, Callable


class Tetrahedron:

    def __init__(self, vertices: np.ndarray):
        self.v = np.asarray(vertices, dtype=float)
        if self.v.shape != (4, 3):
            raise ValueError("Tetrahedron vertices must be 4x3 array")
        self._volume = None
        self._jacobian = None
        self._inv_jacobian = None

    @property
    def volume(self) -> float:
        if self._volume is None:

            mat = np.column_stack([
                self.v[1] - self.v[0],
                self.v[2] - self.v[0],
                self.v[3] - self.v[0]
            ])
            det = np.linalg.det(mat)
            self._volume = abs(det) / 6.0
        return self._volume

    @property
    def jacobian_matrix(self) -> np.ndarray:
        if self._jacobian is None:
            self._jacobian = np.column_stack([
                self.v[1] - self.v[0],
                self.v[2] - self.v[0],
                self.v[3] - self.v[0]
            ])
        return self._jacobian

    @property
    def inv_jacobian_matrix(self) -> np.ndarray:
        if self._inv_jacobian is None:
            J = self.jacobian_matrix
            if abs(np.linalg.det(J)) < 1e-30:
                self._inv_jacobian = np.eye(3)
            else:
                self._inv_jacobian = np.linalg.inv(J)
        return self._inv_jacobian

    def reference_to_physical(self, ref_points: np.ndarray) -> np.ndarray:
        ref = np.asarray(ref_points)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        return self.v[0] + ref @ self.jacobian_matrix.T

    def integrate(self, func: Callable, order: int = 1) -> float:
        if order == 1:
            pts_ref = np.array([[0.25, 0.25, 0.25]])
            w = np.array([1.0])
        elif order == 2:
            a = 0.58541020
            b = 0.13819660
            pts_ref = np.array([
                [a, b, b],
                [b, a, b],
                [b, b, a],
                [b, b, b]
            ])
            w = np.array([0.25, 0.25, 0.25, 0.25])
        elif order == 3:

            pts_ref = np.array([
                [0.25, 0.25, 0.25],
                [0.5, 1.0/6.0, 1.0/6.0],
                [1.0/6.0, 0.5, 1.0/6.0],
                [1.0/6.0, 1.0/6.0, 0.5],
                [1.0/6.0, 1.0/6.0, 1.0/6.0]
            ])
            w = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])
        else:

            pts_ref = np.array([[0.25, 0.25, 0.25]])
            w = np.array([1.0])

        pts_phys = self.reference_to_physical(pts_ref)
        vals = np.array([func(p[0], p[1], p[2]) for p in pts_phys])
        return self.volume * np.dot(w, vals)

    def shape_function_gradients(self) -> np.ndarray:

        grad_ref = np.array([
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ])
        return grad_ref @ self.inv_jacobian_matrix.T

    def quality_measure(self) -> float:

        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(self.v[i] - self.v[j]))
        edges = np.array(edges)
        

        vol = self.volume
        if vol < 1e-30:
            return 0.0
        

        faces = [
            [self.v[1], self.v[2], self.v[3]],
            [self.v[0], self.v[2], self.v[3]],
            [self.v[0], self.v[1], self.v[3]],
            [self.v[0], self.v[1], self.v[2]]
        ]
        area_sum = 0.0
        for f in faces:
            a = np.linalg.norm(f[1] - f[0])
            b = np.linalg.norm(f[2] - f[1])
            c = np.linalg.norm(f[0] - f[2])
            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
            area_sum += area
        

        r_in = 3.0 * vol / area_sum if area_sum > 1e-30 else 0.0
        


        try:
            M = np.zeros((4, 4))
            for i in range(4):
                M[i, :3] = self.v[i]
                M[i, 3] = np.sum(self.v[i] ** 2)
            det_M = np.linalg.det(M)
            
            a_mat = np.ones((4, 4))
            a_mat[:, :3] = self.v
            det_a = np.linalg.det(a_mat)
            
            if abs(det_a) < 1e-30:
                r_circ = 1e10
            else:
                r_circ = abs(det_M) / (6.0 * vol)
        except np.linalg.LinAlgError:
            r_circ = 1e10
        
        if r_circ < 1e-30:
            return 0.0
        return r_in / r_circ


class TetrahedralMesh:

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self._tets = None

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    def get_tetrahedron(self, elem_idx: int) -> Tetrahedron:
        idx = self.elements[elem_idx]
        verts = self.nodes[idx]
        return Tetrahedron(verts)

    def integrate_over_mesh(self, func: Callable, order: int = 1) -> Tuple[float, float]:
        total = 0.0
        vol_total = 0.0
        for e in range(self.n_elements):
            tet = self.get_tetrahedron(e)
            vol_total += tet.volume
            total += tet.integrate(func, order)
        return total, vol_total

    def mesh_quality_stats(self) -> dict:
        qualities = []
        for e in range(self.n_elements):
            tet = self.get_tetrahedron(e)
            q = tet.quality_measure()
            qualities.append(q)
        qualities = np.array(qualities)
        if len(qualities) == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'var': 0.0}
        return {
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'mean': float(np.mean(qualities)),
            'var': float(np.var(qualities))
        }

    @staticmethod
    def generate_uniform_box_mesh(nx: int = 4, ny: int = 4, nz: int = 4,
                                   xlim=(0.0, 1.0),
                                   ylim=(0.0, 1.0),
                                   zlim=(0.0, 1.0)) -> 'TetrahedralMesh':

        xs = np.linspace(xlim[0], xlim[1], nx + 1)
        ys = np.linspace(ylim[0], ylim[1], ny + 1)
        zs = np.linspace(zlim[0], zlim[1], nz + 1)

        nodes = []
        node_index = {}
        idx = 0
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    nodes.append([xs[i], ys[j], zs[k]])
                    node_index[(i, j, k)] = idx
                    idx += 1
        nodes = np.array(nodes)


        elements = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):

                    p000 = node_index[(i, j, k)]
                    p100 = node_index[(i + 1, j, k)]
                    p010 = node_index[(i, j + 1, k)]
                    p110 = node_index[(i + 1, j + 1, k)]
                    p001 = node_index[(i, j, k + 1)]
                    p101 = node_index[(i + 1, j, k + 1)]
                    p011 = node_index[(i, j + 1, k + 1)]
                    p111 = node_index[(i + 1, j + 1, k + 1)]


                    tets = [
                        [p000, p100, p110, p111],
                        [p000, p100, p111, p101],
                        [p000, p101, p111, p001],
                        [p000, p111, p011, p001],
                        [p000, p011, p111, p010],
                        [p000, p010, p111, p110],
                    ]
                    elements.extend(tets)

        elements = np.array(elements)
        return TetrahedralMesh(nodes, elements)


def compute_local_density_field(positions: np.ndarray,
                                mesh: TetrahedralMesh,
                                smoothing_width: float = 0.1) -> np.ndarray:
    n_nodes = mesh.n_nodes
    density = np.zeros(n_nodes)
    sigma_sq = 2.0 * smoothing_width ** 2
    if sigma_sq < 1e-30:
        sigma_sq = 1e-30

    for n in range(n_nodes):
        xn = mesh.nodes[n]
        for r in positions:
            dr = xn - r
            dist_sq = np.dot(dr, dr)
            density[n] += np.exp(-dist_sq / sigma_sq)
    return density
