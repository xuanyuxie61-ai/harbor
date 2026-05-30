
import numpy as np


def triangle_area(p1, p2, p3):
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross)


def refine_triangle_midpoint(p1, p2, p3):
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    n12 = 0.5 * (p1 + p2)
    n23 = 0.5 * (p2 + p3)
    n31 = 0.5 * (p3 + p1)

    points = [p1, p2, p3, n12, n23, n31]
    triangles = [
        (3, 4, 5),
        (0, 3, 5),
        (1, 4, 3),
        (2, 5, 4),
    ]
    return points, triangles


def refine_triangle_local(node_xy, element_node, target_element_idx, element_neighbors=None):
    node_xy = np.asarray(node_xy, dtype=float)
    if element_neighbors is None:
        element_neighbors = [(-1, -1, -1)] * len(element_node)

    n1, n2, n3 = element_node[target_element_idx]
    n1, n2, n3 = int(n1), int(n2), int(n3)


    n12_idx = node_xy.shape[0]
    n23_idx = node_xy.shape[0] + 1
    n31_idx = node_xy.shape[0] + 2

    new_xy = np.vstack([
        node_xy,
        0.5 * (node_xy[n1] + node_xy[n2]),
        0.5 * (node_xy[n2] + node_xy[n3]),
        0.5 * (node_xy[n3] + node_xy[n1])
    ])

    new_elements = []

    for e_idx, tri in enumerate(element_node):
        if e_idx != target_element_idx:
            new_elements.append(tri)


    new_elements.append((n23_idx, n31_idx, n12_idx))
    new_elements.append((n1, n12_idx, n31_idx))
    new_elements.append((n2, n23_idx, n12_idx))
    new_elements.append((n3, n31_idx, n23_idx))

    return new_xy, new_elements


class AdaptiveTriMesh:

    def __init__(self, points, elements):
        self.points = np.asarray(points, dtype=float)
        self.elements = [tuple(int(x) for x in e) for e in elements]

    def element_area(self, e_idx):
        i1, i2, i3 = self.elements[e_idx]
        return triangle_area(self.points[i1], self.points[i2], self.points[i3])

    def element_diameter(self, e_idx):
        i1, i2, i3 = self.elements[e_idx]
        p1, p2, p3 = self.points[i1], self.points[i2], self.points[i3]
        d12 = np.linalg.norm(p1 - p2)
        d23 = np.linalg.norm(p2 - p3)
        d31 = np.linalg.norm(p3 - p1)
        return max(d12, d23, d31)

    def refine_by_indicator(self, indicators, theta=0.7):
        indicators = np.asarray(indicators)
        if len(indicators) != len(self.elements):
            raise ValueError("指示子数量必须等于单元数")
        max_eta = np.max(indicators)
        if max_eta < 1e-15:
            return AdaptiveTriMesh(self.points.copy(), self.elements.copy())

        threshold = theta * max_eta
        points = self.points.copy()
        elements = list(self.elements)


        sorted_idx = np.argsort(-indicators)
        for e_idx in sorted_idx:
            if indicators[e_idx] < threshold:
                break
            points, elements = refine_triangle_local(points, elements, e_idx)

            break

        return AdaptiveTriMesh(points, elements)

    def compute_error_indicator_fmm(self, element_potentials, element_direct):
        diff = np.abs(element_potentials - element_direct)
        denom = np.abs(element_direct) + 1e-15
        return diff / denom

    def to_mesh_data(self):
        n_vertices = self.points.shape[0]
        n_triangles = len(self.elements)
        vertex_labels = np.zeros(n_vertices, dtype=int)
        triangle_labels = np.zeros(n_triangles, dtype=int)

        return {
            "dim": 2,
            "vertices": n_vertices,
            "triangles": n_triangles,
            "vertex_coordinate": self.points.T,
            "vertex_label": vertex_labels,
            "triangle_vertex": np.array(self.elements).T,
            "triangle_label": triangle_labels
        }


def project_3d_to_2d(points_3d, normal=None):
    points_3d = np.asarray(points_3d)
    if normal is None:
        return points_3d[:, :2]
    normal = np.asarray(normal)
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-15)
    proj = np.column_stack([
        np.dot(points_3d, u),
        np.dot(points_3d, v)
    ])
    return proj
