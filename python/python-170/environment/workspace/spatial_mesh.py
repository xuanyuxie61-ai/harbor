
import numpy as np


class TetMesh:

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        if self.elements.shape[1] != 4:
            raise ValueError("Only linear tetrahedra (4 nodes) supported.")
        self._edge_map = {}
        self._build_edge_map()

    def _build_edge_map(self):
        self._edge_map = {}
        edge_idx = 0
        for elem in self.elements:
            edges = [(elem[i], elem[j]) for i in range(4) for j in range(i + 1, 4)]
            for e in edges:
                key = tuple(sorted(e))
                if key not in self._edge_map:
                    self._edge_map[key] = edge_idx
                    edge_idx += 1
        self.num_edges = edge_idx

    def refine(self):
        Nn = self.nodes.shape[0]
        Ne = self.elements.shape[0]


        edge_nodes = {}
        new_nodes = [self.nodes.copy()]
        for (i, j), eidx in self._edge_map.items():
            mid = 0.5 * (self.nodes[i] + self.nodes[j])
            edge_nodes[(i, j)] = Nn + eidx
            edge_nodes[(j, i)] = Nn + eidx
        new_nodes.append(np.array([0.5 * (self.nodes[i] + self.nodes[j])
                                    for (i, j), eidx in sorted(self._edge_map.items(), key=lambda kv: kv[1])]))
        all_nodes = np.vstack(new_nodes)




        new_elements = []
        for elem in self.elements:
            v = list(elem)
            m = [edge_nodes[tuple(sorted((v[i], v[j])))]
                 for i in range(4) for j in range(i + 1, 4)]
            m01, m02, m03, m12, m13, m23 = m

            new_elements.append([v[0], m01, m02, m03])
            new_elements.append([m01, v[1], m12, m13])
            new_elements.append([m02, m12, v[2], m23])
            new_elements.append([m03, m13, m23, v[3]])
            new_elements.append([m01, m02, m03, m13])
            new_elements.append([m01, m02, m12, m13])
            new_elements.append([m02, m03, m13, m23])
            new_elements.append([m02, m12, m13, m23])

        new_elements = np.array(new_elements, dtype=int)
        return TetMesh(all_nodes, new_elements)

    def point_in_tet(self, p: np.ndarray, tet_idx: int, tol: float = 1e-10):
        elem = self.elements[tet_idx]
        v0 = self.nodes[elem[0]]
        v1 = self.nodes[elem[1]]
        v2 = self.nodes[elem[2]]
        v3 = self.nodes[elem[3]]

        M = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        try:
            lamb = np.linalg.solve(M, p - v0)
        except np.linalg.LinAlgError:
            return False, np.zeros(4)
        bary = np.empty(4, dtype=float)
        bary[1:4] = lamb
        bary[0] = 1.0 - np.sum(lamb)
        inside = np.all(bary >= -tol)
        return inside, bary

    def locate_point(self, p: np.ndarray, tol: float = 1e-10):
        p = np.asarray(p, dtype=float)
        for idx in range(self.elements.shape[0]):
            inside, bary = self.point_in_tet(p, idx, tol)
            if inside:
                return idx, bary
        return None, None

    def interpolate_nodal_field(self, p: np.ndarray, field_values: np.ndarray):
        tet_idx, bary = self.locate_point(p)
        if tet_idx is None:
            return np.nan
        elem = self.elements[tet_idx]
        return float(np.dot(bary, field_values[elem]))


def generate_simple_tet_mesh(scale: float = 1.5):
    s = scale
    nodes = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s]
    ], dtype=float)
    elements = np.array([
        [0, 1, 2, 5],
        [0, 2, 7, 5],
        [0, 2, 3, 7],
        [0, 5, 7, 4],
        [2, 5, 7, 6],
        [0, 1, 5, 2]
    ], dtype=int)
    return TetMesh(nodes, elements)


def read_mesh_medit(filename: str):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    vertices = []
    tetrahedra = []
    mode = None
    count = 0
    read_so_far = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        lower = line.lower()
        if lower in ('vertices', 'tetrahedra', 'tetrahedrons', 'triangles', 'edges'):
            mode = lower
            read_so_far = 0
            continue
        if lower == 'end':
            mode = None
            continue
        if mode in ('vertices',):
            if read_so_far == 0:
                count = int(line)
                read_so_far = 0
                continue
            if read_so_far < count:
                parts = line.split()
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                read_so_far += 1
        elif mode in ('tetrahedra', 'tetrahedrons'):
            if read_so_far == 0:
                count = int(line)
                read_so_far = 0
                continue
            if read_so_far < count:
                parts = line.split()
                if len(parts) >= 4:

                    tetrahedra.append([int(parts[0]) - 1, int(parts[1]) - 1,
                                       int(parts[2]) - 1, int(parts[3]) - 1])
                read_so_far += 1

    if len(vertices) == 0 or len(tetrahedra) == 0:
        return None
    return TetMesh(np.array(vertices, dtype=float), np.array(tetrahedra, dtype=int))
