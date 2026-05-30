
import numpy as np


class TetMesh:
    def __init__(self, nodes, elements):
        self.nodes = np.array(nodes, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

    def compute_volumes(self):
        vols = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            idx = self.elements[e]
            x1, x2, x3, x4 = self.nodes[idx]
            J = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
            vols[e] = np.linalg.det(J) / 6.0
        return vols

    def check_jacobian_positive(self, tol=1e-12):
        vols = self.compute_volumes()
        min_vol = np.min(vols)
        neg_count = np.sum(vols < tol)
        return min_vol, neg_count, vols


def tetrahedron_grid_count(n):
    return ((n + 1) * (n + 2) * (n + 3)) // 6


def generate_tetrahedron_grid(n, tet_vertices):
    tet_vertices = np.array(tet_vertices, dtype=float)
    ng = tetrahedron_grid_count(n)
    points = np.zeros((ng, 3))
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                points[p] = (i * tet_vertices[0] +
                             j * tet_vertices[1] +
                             k * tet_vertices[2] +
                             l * tet_vertices[3]) / n
                p += 1
    return points


def generate_cube_tet_mesh(nx, ny, nz, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0)):
    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny
    dz = (zlim[1] - zlim[0]) / nz

    npx, npy, npz = nx + 1, ny + 1, nz + 1
    n_nodes = npx * npy * npz
    nodes = np.zeros((n_nodes, 3))

    for k in range(npz):
        for j in range(npy):
            for i in range(npx):
                idx = i + j * npx + k * npx * npy
                nodes[idx] = [xlim[0] + i * dx, ylim[0] + j * dy, zlim[0] + k * dz]

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):

                n000 = i + j * npx + k * npx * npy
                n100 = (i + 1) + j * npx + k * npx * npy
                n110 = (i + 1) + (j + 1) * npx + k * npx * npy
                n010 = i + (j + 1) * npx + k * npx * npy
                n001 = i + j * npx + (k + 1) * npx * npy
                n101 = (i + 1) + j * npx + (k + 1) * npx * npy
                n111 = (i + 1) + (j + 1) * npx + (k + 1) * npx * npy
                n011 = i + (j + 1) * npx + (k + 1) * npx * npy



                tets = [
                    [n000, n101, n100, n111],
                    [n000, n001, n101, n111],
                    [n000, n011, n001, n111],
                    [n000, n010, n011, n111],
                    [n000, n110, n010, n111],
                    [n000, n100, n110, n111],
                ]
                elements.extend(tets)

    return TetMesh(nodes, elements)


def refine_tet_mesh(mesh):
    old_nodes = mesh.nodes.copy()
    old_elements = mesh.elements.copy()
    n_old = old_nodes.shape[0]


    edge_to_mid = {}
    new_elements = []


    for elem in old_elements:
        edges = [(elem[0], elem[1]), (elem[0], elem[2]), (elem[0], elem[3]),
                 (elem[1], elem[2]), (elem[1], elem[3]), (elem[2], elem[3])]
        for e in edges:
            key = tuple(sorted(e))
            if key not in edge_to_mid:
                edge_to_mid[key] = None


    mid_indices = {}
    new_node_list = list(old_nodes)
    for key in edge_to_mid:
        i, j = key
        mid_idx = len(new_node_list)
        mid_indices[key] = mid_idx
        new_node_list.append((old_nodes[i] + old_nodes[j]) * 0.5)

    new_nodes = np.array(new_node_list)


    for elem in old_elements:
        n1, n2, n3, n4 = elem
        e12 = mid_indices[tuple(sorted((n1, n2)))]
        e13 = mid_indices[tuple(sorted((n1, n3)))]
        e14 = mid_indices[tuple(sorted((n1, n4)))]
        e23 = mid_indices[tuple(sorted((n2, n3)))]
        e24 = mid_indices[tuple(sorted((n2, n4)))]
        e34 = mid_indices[tuple(sorted((n3, n4)))]

        sub_tets = [
            [n1, e12, e13, e14],
            [n2, e12, e23, e24],
            [n3, e13, e23, e34],
            [n4, e14, e24, e34],
            [e12, e13, e14, e24],
            [e12, e13, e23, e24],
            [e13, e14, e24, e34],
            [e13, e23, e24, e34],
        ]
        new_elements.extend(sub_tets)

    return TetMesh(new_nodes, np.array(new_elements, dtype=int))
