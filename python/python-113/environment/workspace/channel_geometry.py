
import numpy as np


class Triangulation:
    def __init__(self, nodes, elements, neighbors=None):
        self.nodes = np.array(nodes, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.dim = self.nodes.shape[1]
        if neighbors is None:
            self.neighbors = self._build_neighbors()
        else:
            self.neighbors = np.array(neighbors, dtype=int)

    def _build_neighbors(self):
        nelem = self.elements.shape[0]
        neighbors = np.full((nelem, 3), -1, dtype=int)
        edge_map = {}
        for e in range(nelem):
            for i in range(3):
                n1 = self.elements[e, i]
                n2 = self.elements[e, (i + 1) % 3]
                edge = (min(n1, n2), max(n1, n2))
                if edge in edge_map:
                    e_prev, i_prev = edge_map[edge]
                    neighbors[e, i] = e_prev
                    neighbors[e_prev, i_prev] = e
                else:
                    edge_map[edge] = (e, i)
        return neighbors

    def refine_local(self, element_index):
        elem = int(element_index)
        if elem < 0 or elem >= self.elements.shape[0]:
            raise IndexError("单元索引越界")

        n1, n2, n3 = self.elements[elem]

        n12 = self.nodes.shape[0]
        n23 = n12 + 1
        n31 = n12 + 2


        m12 = 0.5 * (self.nodes[n1] + self.nodes[n2])
        m23 = 0.5 * (self.nodes[n2] + self.nodes[n3])
        m31 = 0.5 * (self.nodes[n3] + self.nodes[n1])
        self.nodes = np.vstack([self.nodes, m12, m23, m31])


        ea1 = self.neighbors[elem, 0]
        eb1 = self.neighbors[elem, 1]
        ec1 = self.neighbors[elem, 2]

        old_nelem = self.elements.shape[0]

        e1 = old_nelem
        e2 = old_nelem + 1
        e3 = old_nelem + 2
        new_elements = [self.elements.copy()]
        new_neighbors = [self.neighbors.copy()]


        new_elem0 = np.array([n23, n31, n12])
        new_elements[0][elem] = new_elem0


        added = np.array([
            [n1, n12, n31],
            [n2, n23, n12],
            [n3, n31, n23]
        ])
        new_elements.append(added)


        added_neigh = np.array([
            [elem, -1, -1],
            [elem, -1, -1],
            [elem, -1, -1]
        ])
        new_neighbors.append(added_neigh)

        self.elements = np.vstack(new_elements)
        self.neighbors = np.vstack(new_neighbors)
        self.neighbors[elem] = [e1, e2, e3]


        return self

    def element_centers(self):
        return np.mean(self.nodes[self.elements], axis=1)

    def in_selectivity_filter(self, points, radius=0.15):
        if self.dim == 2:

            r = np.sqrt(points[:, 0] ** 2)
            z = points[:, 1]
            return (r <= radius) & (z >= 0.0) & (z <= 1.2)
        else:
            r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
            z = points[:, 2]
            return (r <= radius) & (z >= 0.0) & (z <= 1.2)

    def adaptive_refine_filter(self, max_level=2):
        for level in range(max_level):
            centers = self.element_centers()
            mask = self.in_selectivity_filter(centers)
            to_refine = np.where(mask)[0]

            for idx in to_refine[:min(len(to_refine), 5)]:
                if idx < self.elements.shape[0]:
                    self.refine_local(idx)
        return self


def build_channel_mesh_2d(nz=40, nr=20):
    z = np.linspace(0.0, 4.5, nz)
    r = np.linspace(0.0, 0.6, nr)
    Z, R = np.meshgrid(z, r)

    nodes = []
    node_map = {}
    idx = 0
    for i in range(nr):
        for j in range(nz):

            rr = R[i, j]
            zz = Z[i, j]

            if 1.5 <= zz <= 2.7 and rr > 0.15:
                continue

            if zz < 0.5 and rr > 0.2 + 0.4 * zz:
                continue
            nodes.append([rr, zz])
            node_map[(i, j)] = idx
            idx += 1

    nodes = np.array(nodes)


    elements = []
    for i in range(nr - 1):
        for j in range(nz - 1):
            c00 = node_map.get((i, j))
            c10 = node_map.get((i + 1, j))
            c01 = node_map.get((i, j + 1))
            c11 = node_map.get((i + 1, j + 1))
            if None not in (c00, c10, c01, c11):
                elements.append([c00, c10, c11])
                elements.append([c00, c11, c01])

    elements = np.array(elements, dtype=int)
    tri = Triangulation(nodes, elements)
    return tri
