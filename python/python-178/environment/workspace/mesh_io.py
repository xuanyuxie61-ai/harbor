
import numpy as np
from typing import Tuple, Optional






class TetrahedralMesh:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_nodes = self.nodes.shape[0]
        self.n_elem = self.elements.shape[0]
        if self.nodes.shape[1] != 3:
            raise ValueError("Nodes must have 3 coordinates.")
        if self.elements.shape[1] != 4:
            raise ValueError("Elements must have 4 vertices.")
        self._validate()
        self._compute_neighbors()

    def _validate(self):
        min_idx = self.elements.min()
        max_idx = self.elements.max()
        if min_idx < 0 or max_idx >= self.n_nodes:
            raise ValueError("Element indices out of bounds.")

    def _compute_neighbors(self):

        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ], dtype=np.int64)

        face_map = {}
        self.face_elements = -np.ones((self.n_elem, 4), dtype=np.int64)
        self.face_local_face = -np.ones((self.n_elem, 4), dtype=np.int64)
        for e in range(self.n_elem):
            for f in range(4):
                face_nodes = tuple(sorted(self.elements[e, faces[f]]))
                if face_nodes in face_map:
                    e2, f2 = face_map[face_nodes]
                    self.face_elements[e, f] = e2
                    self.face_local_face[e, f] = f2
                    self.face_elements[e2, f2] = e
                    self.face_local_face[e2, f2] = f
                else:
                    face_map[face_nodes] = (e, f)

        self.boundary_faces = []
        for e in range(self.n_elem):
            for f in range(4):
                if self.face_elements[e, f] < 0:
                    face_nodes = self.elements[e, faces[f]]
                    self.boundary_faces.append((e, f, face_nodes))
        self.n_boundary_faces = len(self.boundary_faces)

    def element_volume(self, e: int) -> float:
        v = self.elements[e]
        M = np.vstack([
            self.nodes[v[1]] - self.nodes[v[0]],
            self.nodes[v[2]] - self.nodes[v[0]],
            self.nodes[v[3]] - self.nodes[v[0]]
        ])
        return abs(np.linalg.det(M)) / 6.0

    def face_normal_and_area(self, e: int, f: int) -> Tuple[np.ndarray, float]:
        faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int64)
        v = self.elements[e, faces[f]]
        p0, p1, p2 = self.nodes[v[0]], self.nodes[v[1]], self.nodes[v[2]]
        u = p1 - p0
        w = p2 - p0
        cross = np.cross(u, w)
        area = 0.5 * np.linalg.norm(cross)
        normal = cross / (2.0 * area + 1e-30)

        centroid = self.nodes[self.elements[e]].mean(axis=0)
        face_center = (p0 + p1 + p2) / 3.0
        if np.dot(normal, face_center - centroid) < 0:
            normal = -normal
        return normal, area


def generate_unit_cube_tetrahedra() -> TetrahedralMesh:
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    elements = np.array([
        [0, 1, 2, 6],
        [0, 1, 6, 5],
        [0, 5, 6, 4],
        [0, 2, 3, 6],
        [0, 3, 6, 7],
        [0, 4, 6, 7],
    ], dtype=np.int64)
    return TetrahedralMesh(nodes, elements)


def generate_refined_mesh(nx: int = 2, ny: int = 2, nz: int = 2) -> TetrahedralMesh:
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("Grid dimensions must be at least 1.")
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    z = np.linspace(0.0, 1.0, nz + 1)
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    idx = 0
    node_index = {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[idx] = [x[i], y[j], z[k]]
                node_index[(i, j, k)] = idx
                idx += 1
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c000 = node_index[(i, j, k)]
                c100 = node_index[(i + 1, j, k)]
                c110 = node_index[(i + 1, j + 1, k)]
                c010 = node_index[(i, j + 1, k)]
                c001 = node_index[(i, j, k + 1)]
                c101 = node_index[(i + 1, j, k + 1)]
                c111 = node_index[(i + 1, j + 1, k + 1)]
                c011 = node_index[(i, j + 1, k + 1)]

                elements.append([c000, c100, c110, c111])
                elements.append([c000, c100, c111, c101])
                elements.append([c000, c101, c111, c001])
                elements.append([c000, c110, c010, c111])
                elements.append([c000, c010, c111, c011])
                elements.append([c000, c011, c111, c001])
    elements = np.array(elements, dtype=np.int64)
    return TetrahedralMesh(nodes, elements)






class SparseMatrixCOO:
    def __init__(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                 shape: Optional[Tuple[int, int]] = None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.vals = np.asarray(vals, dtype=np.float64)
        self.nnz = len(self.vals)
        if shape is None:
            n = max(self.rows.max(), self.cols.max()) + 1
            self.shape = (n, n)
        else:
            self.shape = shape

    def to_crs(self) -> 'SparseMatrixCRS':
        n_rows = self.shape[0]

        order = np.lexsort((self.cols, self.rows))
        rows = self.rows[order]
        cols = self.cols[order]
        vals = self.vals[order]
        row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
        for i in range(self.nnz):
            row_ptr[rows[i] + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        return SparseMatrixCRS(vals, cols, row_ptr, self.shape)


class SparseMatrixCRS:
    def __init__(self, vals: np.ndarray, cols: np.ndarray,
                 row_ptr: np.ndarray, shape: Tuple[int, int]):
        self.vals = np.asarray(vals, dtype=np.float64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.row_ptr = np.asarray(row_ptr, dtype=np.int64)
        self.shape = shape

    def mv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.shape[1]:
            raise ValueError("Dimension mismatch in CRS matrix-vector product.")
        y = np.zeros(self.shape[0], dtype=np.float64)
        for i in range(self.shape[0]):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                y[i] += self.vals[k] * x[self.cols[k]]
        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.shape[0]:
            raise ValueError("Dimension mismatch in CRS transposed product.")
        y = np.zeros(self.shape[1], dtype=np.float64)
        for i in range(self.shape[0]):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                y[self.cols[k]] += self.vals[k] * x[i]
        return y

    def to_dense(self) -> np.ndarray:
        A = np.zeros(self.shape, dtype=np.float64)
        for i in range(self.shape[0]):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.cols[k]] = self.vals[k]
        return A


def build_sparse_laplacian_1d(n: int) -> SparseMatrixCRS:
    if n < 2:
        raise ValueError("n must be at least 2.")
    rows, cols, vals = [], [], []
    for i in range(n):
        if i > 0:
            rows.append(i); cols.append(i - 1); vals.append(-1.0)
        rows.append(i); cols.append(i); vals.append(2.0)
        if i < n - 1:
            rows.append(i); cols.append(i + 1); vals.append(-1.0)
    coo = SparseMatrixCOO(np.array(rows), np.array(cols), np.array(vals), (n, n))
    return coo.to_crs()


def write_matrix_market(coo: SparseMatrixCOO, filename: str):
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{coo.shape[0]} {coo.shape[1]} {coo.nnz}\n")
        for i in range(coo.nnz):
            f.write(f"{coo.rows[i] + 1} {coo.cols[i] + 1} {coo.vals[i]:.16e}\n")
