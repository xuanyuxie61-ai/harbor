
import numpy as np


def rotation_matrix_2d(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def dilation_matrix_2d(sx, sy):
    return np.array([[sx, 0.0], [0.0, sy]], dtype=float)


def translation_vector_2d(tx, ty, n_points):
    return np.tile(np.array([[tx], [ty]], dtype=float), (1, n_points))


def affine_transform_2d(points, A=None, b=None):
    points = np.array(points, dtype=float)
    if points.shape[0] != 2:
        raise ValueError("points must have shape (2, N)")
    if A is None:
        A = np.eye(2)
    if b is None:
        b = np.zeros((2, points.shape[1]))
    elif b.ndim == 1:
        b = np.tile(b[:, np.newaxis], (1, points.shape[1]))
    return A @ points + b


def transform_mesh(nodes, elements, A=None, b=None):
    new_nodes = affine_transform_2d(nodes, A, b)
    return new_nodes, elements


def polygon_surface_quality(nodes, elements):
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    nelem = elements.shape[1]
    quality = np.zeros(nelem)





    raise NotImplementedError("Hole_3: polygon_surface_quality 质量计算循环待实现")
    return quality, float(np.min(quality)), float(np.mean(quality))


def adaptive_refinement_markers(nodes, elements, gradient_field, threshold_ratio=0.8):
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    gradient_field = np.array(gradient_field, dtype=float)
    nelem = elements.shape[1]
    grad_norm = np.zeros(nelem)
    for e in range(nelem):
        i1, i2, i3 = elements[:, e] - 1
        g1 = np.linalg.norm(gradient_field[:, i1])
        g2 = np.linalg.norm(gradient_field[:, i2])
        g3 = np.linalg.norm(gradient_field[:, i3])
        grad_norm[e] = (g1 + g2 + g3) / 3.0
    sorted_idx = np.argsort(grad_norm)[::-1]
    n_mark = max(1, int(np.ceil(threshold_ratio * nelem)))
    marker = np.zeros(nelem, dtype=bool)
    marker[sorted_idx[:n_mark]] = True
    return marker


def refine_marked_elements(nodes, elements, marker):
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    node_num = nodes.shape[1]
    elem_num = elements.shape[1]


    edge_mid = {}
    new_nodes_list = [nodes[:, i] for i in range(node_num)]

    def get_mid(i, j):
        i, j = int(i), int(j)
        if i > j:
            i, j = j, i
        key = (i, j)
        if key not in edge_mid:
            mid_pt = (nodes[:, i] + nodes[:, j]) / 2.0
            idx = node_num + len(edge_mid)
            edge_mid[key] = idx
            new_nodes_list.append(mid_pt)
        return edge_mid[key]

    new_elements_list = []
    for e in range(elem_num):
        i1, i2, i3 = elements[:, e] - 1
        if marker[e]:
            m12 = get_mid(i1, i2)
            m23 = get_mid(i2, i3)
            m31 = get_mid(i3, i1)
            new_elements_list.append([i1 + 1, m12 + 1, m31 + 1])
            new_elements_list.append([m12 + 1, i2 + 1, m23 + 1])
            new_elements_list.append([m31 + 1, m23 + 1, i3 + 1])
            new_elements_list.append([m12 + 1, m23 + 1, m31 + 1])
        else:
            new_elements_list.append([i1 + 1, i2 + 1, i3 + 1])

    new_nodes = np.column_stack(new_nodes_list)
    new_elements = np.array(new_elements_list, dtype=int).T
    return new_nodes, new_elements
