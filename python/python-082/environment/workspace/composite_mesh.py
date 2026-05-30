
import numpy as np


class CompositeMesh:

    def __init__(self, nodes, elements, ply_ids, fiber_angles):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.ply_ids = np.asarray(ply_ids, dtype=int)
        self.fiber_angles = np.asarray(fiber_angles, dtype=float)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        self.n_plys = len(self.fiber_angles)
        self._validate()

    def _validate(self):
        if self.nodes.shape[1] != 3:
            raise ValueError("Nodes must have 3 columns (x, y, z).")
        if self.elements.shape[1] != 4:
            raise ValueError("Elements must be quadrilaterals (4 nodes).")
        if self.ply_ids.min() < 0 or self.ply_ids.max() >= self.n_plys:
            raise ValueError("ply_ids out of range.")
        if np.any(np.isnan(self.nodes)) or np.any(np.isinf(self.nodes)):
            raise ValueError("Mesh nodes contain NaN or Inf.")

    @classmethod
    def generate_laminate(cls, length=1.0, width=1.0, n_x=10, n_y=10,
                          ply_thickness=0.125e-3, n_plys=8,
                          fiber_angles=None):
        if fiber_angles is None:
            if n_plys == 8:
                fiber_angles = [0.0, 90.0, 45.0, -45.0,
                                -45.0, 45.0, 90.0, 0.0]
            else:
                fiber_angles = [0.0] * n_plys
        if len(fiber_angles) != n_plys:
            raise ValueError("Length of fiber_angles must equal n_plys.")


        nx_nodes = n_x + 1
        ny_nodes = n_y + 1
        n_nodes_per_layer = nx_nodes * ny_nodes
        total_nodes = n_nodes_per_layer * (n_plys + 1)

        nodes = np.zeros((total_nodes, 3), dtype=float)
        dx = length / n_x
        dy = width / n_y

        node_id = 0
        for k in range(n_plys + 1):
            z = k * ply_thickness
            for j in range(ny_nodes):
                y = j * dy
                for i in range(nx_nodes):
                    x = i * dx
                    nodes[node_id] = [x, y, z]
                    node_id += 1


        elements = []
        ply_ids = []
        for k in range(n_plys):
            base = k * n_nodes_per_layer
            for j in range(n_y):
                for i in range(n_x):
                    n0 = base + j * nx_nodes + i
                    n1 = base + j * nx_nodes + (i + 1)
                    n2 = base + (j + 1) * nx_nodes + (i + 1)
                    n3 = base + (j + 1) * nx_nodes + i
                    elements.append([n0, n1, n2, n3])
                    ply_ids.append(k)

        return cls(nodes, elements, ply_ids, fiber_angles)

    def element_centroids(self):
        centroids = np.zeros((self.n_elements, 3))
        for e in range(self.n_elements):
            elem_nodes = self.elements[e]
            centroids[e] = np.mean(self.nodes[elem_nodes], axis=0)
        return centroids

    def element_area(self, elem_idx):
        coords = self.nodes[self.elements[elem_idx]]

        v0 = coords[1] - coords[0]
        v1 = coords[2] - coords[0]
        v2 = coords[3] - coords[0]
        n1 = np.cross(v0, v1)
        n2 = np.cross(v1, v2)
        area = 0.5 * (np.linalg.norm(n1) + np.linalg.norm(n2))
        return area

    def element_normal(self, elem_idx):
        coords = self.nodes[self.elements[elem_idx]]
        v0 = coords[1] - coords[0]
        v1 = coords[3] - coords[0]
        n = np.cross(v0, v1)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            return np.array([0.0, 0.0, 1.0])
        return n / norm

    def total_surface_area(self):
        return sum(self.element_area(e) for e in range(self.n_elements))

    def rotation_matrix(self, ply_idx, angle_deg=None):
        if angle_deg is None:
            angle_deg = self.fiber_angles[ply_idx]
        theta = np.deg2rad(angle_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return R

    def reuter_matrix(self, ply_idx, angle_deg=None):
        if angle_deg is None:
            angle_deg = self.fiber_angles[ply_idx]
        theta = np.deg2rad(angle_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        T = np.array([
            [c * c, s * s, 2.0 * s * c],
            [s * s, c * c, -2.0 * s * c],
            [-s * c, s * c, c * c - s * s]
        ])
        return T

    def summary(self):
        print("=" * 60)
        print("Composite Mesh Summary")
        print("=" * 60)
        print(f"  Nodes          : {self.n_nodes}")
        print(f"  Elements       : {self.n_elements}")
        print(f"  Plys           : {self.n_plys}")
        print(f"  Total area     : {self.total_surface_area():.6e} m^2")
        print(f"  Fiber angles   : {self.fiber_angles}")
        print("=" * 60)
