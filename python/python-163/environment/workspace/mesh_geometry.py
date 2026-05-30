
import numpy as np


class Triangle:

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if self.vertices.shape != (3, 2):
            raise ValueError("Triangle requires exactly 3 vertices in 2D.")
        self._compute_properties()

    def _compute_properties(self):
        A = self.vertices[0]
        B = self.vertices[1]
        C = self.vertices[2]


        self.a = np.linalg.norm(B - C)
        self.b = np.linalg.norm(A - C)
        self.c = np.linalg.norm(A - B)


        self.s = 0.5 * (self.a + self.b + self.c)
        area_sq = self.s * (self.s - self.a) * (self.s - self.b) * (self.s - self.c)
        if area_sq <= 0:
            self.area = 0.0
            self.degenerate = True
        else:
            self.area = np.sqrt(area_sq)
            self.degenerate = False


        self.angles = np.zeros(3)
        if not self.degenerate:
            self.angles[0] = np.arccos(np.clip((self.b**2 + self.c**2 - self.a**2)
                                               / (2.0 * self.b * self.c), -1.0, 1.0))
            self.angles[1] = np.arccos(np.clip((self.a**2 + self.c**2 - self.b**2)
                                               / (2.0 * self.a * self.c), -1.0, 1.0))
            self.angles[2] = np.arccos(np.clip((self.a**2 + self.b**2 - self.c**2)
                                               / (2.0 * self.a * self.b), -1.0, 1.0))


        self.centroid = (A + B + C) / 3.0


        if not self.degenerate:
            D = 2.0 * (A[0] * (B[1] - C[1])
                       + B[0] * (C[1] - A[1])
                       + C[0] * (A[1] - B[1]))
            if abs(D) > 1.0e-14:
                ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1])
                      + (B[0]**2 + B[1]**2) * (C[1] - A[1])
                      + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
                uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0])
                      + (B[0]**2 + B[1]**2) * (A[0] - C[0])
                      + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
                self.circum_center = np.array([ux, uy])
                self.circum_radius = np.linalg.norm(A - self.circum_center)
            else:
                self.circum_center = np.zeros(2)
                self.circum_radius = 0.0
        else:
            self.circum_center = np.zeros(2)
            self.circum_radius = 0.0


        if not self.degenerate and self.s > 0:
            self.in_radius = self.area / self.s
            self.in_center = (self.a * A + self.b * B + self.c * C) / (self.a + self.b + self.c)
        else:
            self.in_radius = 0.0
            self.in_center = self.centroid.copy()


        if not self.degenerate and (self.a**2 + self.b**2 + self.c**2) > 0:
            self.quality = 4.0 * np.sqrt(3.0) * self.area / (self.a**2 + self.b**2 + self.c**2)
        else:
            self.quality = 0.0

    def is_well_formed(self, min_angle_deg=15.0, max_angle_deg=120.0):
        if self.degenerate:
            return False
        angles_deg = np.degrees(self.angles)
        return (np.all(angles_deg >= min_angle_deg)
                and np.all(angles_deg <= max_angle_deg)
                and self.quality > 0.3)


class Tetrahedron:

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if self.vertices.shape != (4, 3):
            raise ValueError("Tetrahedron requires exactly 4 vertices in 3D.")
        self._compute_properties()

    def _compute_properties(self):
        v0, v1, v2, v3 = self.vertices
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0


        self.volume = abs(np.dot(e1, np.cross(e2, e3))) / 6.0
        self.degenerate = (self.volume < 1.0e-14)


        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(self.vertices[i] - self.vertices[j]))
        self.edge_lengths = np.array(edges)


        if not self.degenerate:

            self.quality = (216.0 * np.sqrt(3.0) * self.volume
                            / np.sum(self.edge_lengths ** 3))
        else:
            self.quality = 0.0


        self.centroid = np.mean(self.vertices, axis=0)


def triangle_mesh_quality(triangles):
    qualities = []
    areas = []
    min_angles = []
    max_angles = []

    for tri_verts in triangles:
        tri = Triangle(tri_verts)
        qualities.append(tri.quality)
        areas.append(tri.area)
        if not tri.degenerate:
            min_angles.append(np.min(np.degrees(tri.angles)))
            max_angles.append(np.max(np.degrees(tri.angles)))

    stats = {
        "num_triangles": len(triangles),
        "min_quality": float(np.min(qualities)) if qualities else 0.0,
        "max_quality": float(np.max(qualities)) if qualities else 0.0,
        "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
        "total_area": float(np.sum(areas)),
        "min_angle_deg": float(np.min(min_angles)) if min_angles else 0.0,
        "max_angle_deg": float(np.max(max_angles)) if max_angles else 0.0,
    }
    return stats


def reservoir_boundary_polygon():

    xy = np.array([
        [0.0, 0.0],
        [500.0, 0.0],
        [500.0, 200.0],
        [250.0, 250.0],
        [0.0, 200.0]
    ], dtype=np.float64)
    return xy


def triangulate_polygon_simple(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    n = polygon.shape[0]
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices.")

    triangles = []
    for i in range(1, n - 1):
        triangles.append(np.array([polygon[0], polygon[i], polygon[i + 1]]))
    return triangles


def generate_structured_hex_mesh(Lx, Ly, Lz, nx, ny, nz):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    nodes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k]])
    nodes = np.array(nodes, dtype=np.float64)

    def node_index(i, j, k):
        return k * nx * ny + j * nx + i

    elements = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                elem = [
                    node_index(i, j, k),
                    node_index(i + 1, j, k),
                    node_index(i + 1, j + 1, k),
                    node_index(i, j + 1, k),
                    node_index(i, j, k + 1),
                    node_index(i + 1, j, k + 1),
                    node_index(i + 1, j + 1, k + 1),
                    node_index(i, j + 1, k + 1),
                ]
                elements.append(elem)

    elements = np.array(elements, dtype=np.int64)
    return nodes, elements


def convert_quadratic_tet_to_linear(quadratic_elements):
    quadratic_elements = np.asarray(quadratic_elements, dtype=np.int64)
    if quadratic_elements.shape[0] != 10:
        raise ValueError("Quadratic tetrahedra must have 10 nodes.")
    n_tets = quadratic_elements.shape[1]
    n_linear = 8 * n_tets
    linear_elements = np.zeros((4, n_linear), dtype=np.int64)

    tetra2 = 0
    for tetra1 in range(n_tets):
        n1 = quadratic_elements[0, tetra1]
        n2 = quadratic_elements[1, tetra1]
        n3 = quadratic_elements[2, tetra1]
        n4 = quadratic_elements[3, tetra1]
        n5 = quadratic_elements[4, tetra1]
        n6 = quadratic_elements[5, tetra1]
        n7 = quadratic_elements[6, tetra1]
        n8 = quadratic_elements[7, tetra1]
        n9 = quadratic_elements[8, tetra1]
        nx = quadratic_elements[9, tetra1]

        linear_elements[:, tetra2] = [n1, n5, n6, n7]
        tetra2 += 1
        linear_elements[:, tetra2] = [n2, n5, n8, n9]
        tetra2 += 1
        linear_elements[:, tetra2] = [n3, n6, n8, nx]
        tetra2 += 1
        linear_elements[:, tetra2] = [n4, n7, n9, nx]
        tetra2 += 1
        linear_elements[:, tetra2] = [n5, n6, n7, n9]
        tetra2 += 1
        linear_elements[:, tetra2] = [n5, n6, n8, n9]
        tetra2 += 1
        linear_elements[:, tetra2] = [n6, n7, n9, nx]
        tetra2 += 1
        linear_elements[:, tetra2] = [n6, n8, n9, nx]
        tetra2 += 1

    return linear_elements


def reservoir_tetrahedral_mesh(params):
    Lx = params.reservoir_length
    Ly = params.reservoir_width
    Lz = params.reservoir_height
    nx, nz, ny = params.grid_shape()

    nodes, hex_elements = generate_structured_hex_mesh(Lx, Ly, Lz, nx, ny, nz)


    n_hex = hex_elements.shape[0]
    n_tets = 6 * n_hex
    tetra_elements = np.zeros((4, n_tets), dtype=np.int64)

    idx = 0
    for h in range(n_hex):
        e = hex_elements[h]
        v = [nodes[e[i]] for i in range(8)]


        tets = [
            [e[0], e[1], e[3], e[4]],
            [e[1], e[2], e[3], e[6]],
            [e[1], e[3], e[4], e[6]],
            [e[1], e[4], e[5], e[6]],
            [e[3], e[4], e[6], e[7]],
            [e[2], e[3], e[6], e[7]],
        ]
        for t in tets:
            tetra_elements[:, idx] = t
            idx += 1

    return nodes, tetra_elements
