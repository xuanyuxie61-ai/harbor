
import numpy as np


class TetrahedralMesh:

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self._volumes = None
        self._barycenters = None

    @staticmethod
    def generate_uniform_box(nx: int = 5, ny: int = 5, nz: int = 5,
                              xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0)):
        if nx < 2 or ny < 2 or nz < 2:
            raise ValueError("nx, ny, nz must be >= 2")
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        z = np.linspace(zlim[0], zlim[1], nz)
        nodes = []
        idx = {}
        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    nodes.append([x[i], y[j], z[k]])
                    idx[(i, j, k)] = count
                    count += 1
        nodes = np.array(nodes)
        elements = []
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    c000 = idx[(i, j, k)]
                    c100 = idx[(i + 1, j, k)]
                    c010 = idx[(i, j + 1, k)]
                    c110 = idx[(i + 1, j + 1, k)]
                    c001 = idx[(i, j, k + 1)]
                    c101 = idx[(i + 1, j, k + 1)]
                    c011 = idx[(i, j + 1, k + 1)]
                    c111 = idx[(i + 1, j + 1, k + 1)]

                    elements.append([c000, c100, c110, c111])
                    elements.append([c000, c100, c111, c101])
                    elements.append([c000, c010, c110, c111])
                    elements.append([c000, c010, c111, c011])
                    elements.append([c000, c001, c101, c111])
                    elements.append([c000, c001, c111, c011])
        elements = np.array(elements, dtype=int)
        return TetrahedralMesh(nodes, elements)

    def compute_volumes(self) -> np.ndarray:


        raise NotImplementedError("compute_volumes not implemented (Hole 3)")

    def barycenters(self) -> np.ndarray:
        if self._barycenters is not None:
            return self._barycenters
        v = self.nodes[self.elements]
        self._barycenters = np.mean(v, axis=1)
        return self._barycenters

    def bounding_box(self):
        return (
            self.nodes[:, 0].min(), self.nodes[:, 0].max(),
            self.nodes[:, 1].min(), self.nodes[:, 1].max(),
            self.nodes[:, 2].min(), self.nodes[:, 2].max(),
        )

    def element_diameter(self) -> float:
        max_d = 0.0
        for e in self.elements:
            pts = self.nodes[e]
            for i in range(4):
                for j in range(i + 1, 4):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d > max_d:
                        max_d = d
        return max_d
