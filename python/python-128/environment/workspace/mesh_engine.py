
import numpy as np


class TetrahedralMesh:

    def __init__(self, nodes=None, elements=None):
        self.nodes = np.zeros((0, 3), dtype=float) if nodes is None else np.asarray(nodes, dtype=float)
        self.elements = np.zeros((0, 4), dtype=int) if elements is None else np.asarray(elements, dtype=int)
        self._validate()

    def _validate(self):
        if self.nodes.size == 0 or self.elements.size == 0:
            return
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("TetrahedralMesh: nodes 必须为 (n, 3) 数组")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("TetrahedralMesh: elements 必须为 (m, 4) 数组")
        n_nodes = self.nodes.shape[0]
        emin = self.elements.min()
        emax = self.elements.max()
        if emin < 0 or emax >= n_nodes:

            if emin == 1 and emax == n_nodes:
                self.elements = self.elements - 1
            else:
                raise ValueError(
                    "TetrahedralMesh: 单元节点索引越界 (min=%d, max=%d, n_nodes=%d)" % (emin, emax, n_nodes)
                )

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]

    def element_volume(self, elem_idx: int):
        idx = self.elements[elem_idx]
        a, b, c, d = self.nodes[idx[0]], self.nodes[idx[1]], self.nodes[idx[2]], self.nodes[idx[3]]
        vol = np.dot(a - d, np.cross(b - d, c - d)) / 6.0
        return abs(vol)

    def total_volume(self):
        vol = 0.0
        for e in range(self.n_elements):
            vol += self.element_volume(e)
        return vol

    def refine_uniform(self):
        if self.n_elements == 0:
            return TetrahedralMesh()

        n_nodes_old = self.n_nodes

        edge_to_mid = {}
        new_nodes = [self.nodes.copy()]

        def get_mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in edge_to_mid:
                mid_idx = n_nodes_old + len(edge_to_mid)
                edge_to_mid[key] = mid_idx
                new_nodes.append(((self.nodes[i] + self.nodes[j]) / 2.0).reshape(1, 3))
            return edge_to_mid[key]

        new_elements = []
        for e in range(self.n_elements):
            v0, v1, v2, v3 = self.elements[e]
            m01 = get_mid(v0, v1)
            m02 = get_mid(v0, v2)
            m03 = get_mid(v0, v3)
            m12 = get_mid(v1, v2)
            m13 = get_mid(v1, v3)
            m23 = get_mid(v2, v3)


            subs = [
                [v0, m01, m02, m03],
                [m01, v1, m12, m13],
                [m02, m12, v2, m23],
                [m03, m13, m23, v3],
                [m01, m02, m03, m13],
                [m01, m02, m12, m13],
                [m02, m03, m13, m23],
                [m02, m12, m13, m23],
            ]
            new_elements.extend(subs)

        all_nodes = np.vstack(new_nodes) if len(new_nodes) > 1 else new_nodes[0]
        all_elements = np.array(new_elements, dtype=int)
        return TetrahedralMesh(all_nodes, all_elements)

    def compute_centroids(self):
        c = np.zeros((self.n_elements, 3), dtype=float)
        for e in range(self.n_elements):
            idx = self.elements[e]
            c[e] = self.nodes[idx].mean(axis=0)
        return c

    def compute_boundary_faces(self):
        face_count = {}
        for e in range(self.n_elements):
            idx = list(self.elements[e])
            faces = [
                tuple(sorted([idx[0], idx[1], idx[2]])),
                tuple(sorted([idx[0], idx[1], idx[3]])),
                tuple(sorted([idx[0], idx[2], idx[3]])),
                tuple(sorted([idx[1], idx[2], idx[3]])),
            ]
            for f in faces:
                face_count[f] = face_count.get(f, 0) + 1

        boundary_faces = [f for f, cnt in face_count.items() if cnt == 1]
        return np.array(boundary_faces, dtype=int)


def generate_uniform_box_mesh(xlim=(-1.0, 1.0),
                              ylim=(-1.0, 1.0),
                              zlim=(-1.0, 1.0),
                              nx=4, ny=4, nz=4):
    nx = max(2, int(nx))
    ny = max(2, int(ny))
    nz = max(2, int(nz))

    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)

    nodes = []
    node_index = {}
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k]])
                node_index[(i, j, k)] = idx
                idx += 1

    nodes = np.array(nodes, dtype=float)
    elements = []

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                c = [
                    node_index[(i, j, k)],
                    node_index[(i + 1, j, k)],
                    node_index[(i + 1, j + 1, k)],
                    node_index[(i, j + 1, k)],
                    node_index[(i, j, k + 1)],
                    node_index[(i + 1, j, k + 1)],
                    node_index[(i + 1, j + 1, k + 1)],
                    node_index[(i, j + 1, k + 1)],
                ]


                tets = [
                    [c[0], c[1], c[3], c[4]],
                    [c[1], c[3], c[4], c[5]],
                    [c[1], c[2], c[3], c[5]],
                    [c[3], c[4], c[5], c[7]],
                    [c[3], c[5], c[6], c[7]],
                    [c[2], c[3], c[5], c[6]],
                ]
                elements.extend(tets)

    elements = np.array(elements, dtype=int)
    return TetrahedralMesh(nodes, elements)


def gmsh_format_string(mesh: TetrahedralMesh):
    lines = []
    lines.append("$MeshFormat")
    lines.append("2.2 0 8")
    lines.append("$EndMeshFormat")
    lines.append("$Nodes")
    lines.append("%d" % mesh.n_nodes)
    for i in range(mesh.n_nodes):
        lines.append("  %d %.16g %.16g %.16g" % (i + 1, mesh.nodes[i, 0], mesh.nodes[i, 1], mesh.nodes[i, 2]))
    lines.append("$EndNodes")
    lines.append("$Elements")
    lines.append("%d" % mesh.n_elements)
    for e in range(mesh.n_elements):

        nodes_str = " ".join("%d" % (mesh.elements[e, v] + 1) for v in range(4))
        lines.append("  %d 4 2 0 %d %s" % (e + 1, e + 1, nodes_str))
    lines.append("$EndElements")
    return "\n".join(lines)


def parse_stl_like_surface(nodes_xyz, face_nodes):
    nodes = np.asarray(nodes_xyz, dtype=float)
    faces = np.asarray(face_nodes, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("parse_stl_like_surface: faces 必须为 (m, 3) 数组")
    return nodes, faces
