import numpy as np






def triangle_grid_count(n_subdivisions):
    n = int(n_subdivisions)
    if n < 0:
        raise ValueError("n_subdivisions must be non-negative")
    return (n + 1) * (n + 2) // 2


def triangle_grid(n_subdivisions, vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must be (3, 2) array")

    N = int(n_subdivisions)
    if N < 0:
        raise ValueError("n_subdivisions must be non-negative")

    count = triangle_grid_count(N)
    points = np.zeros((count, 2), dtype=np.float64)

    idx = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            points[idx] = (i * vertices[0] + j * vertices[1] + k * vertices[2]) / N
            idx += 1

    return points


def generate_disk_cross_section_mesh(r_in, r_out, z_max, n_r=20, n_z=10):
    if r_in <= 0 or r_out <= r_in or z_max <= 0:
        raise ValueError("Invalid geometry parameters")


    r_nodes = np.linspace(r_in, r_out, n_r + 1)
    z_nodes = np.linspace(-z_max, z_max, n_z + 1)


    nodes = []
    node_map = {}
    for i, r in enumerate(r_nodes):
        for j, z in enumerate(z_nodes):
            nodes.append([r, z])
            node_map[(i, j)] = len(nodes) - 1

    nodes = np.array(nodes, dtype=np.float64)


    elements = []
    for i in range(n_r):
        for j in range(n_z):
            n1 = node_map[(i, j)]
            n2 = node_map[(i + 1, j)]
            n3 = node_map[(i, j + 1)]
            n4 = node_map[(i + 1, j + 1)]


            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])

    elements = np.array(elements, dtype=np.int64)

    return nodes, elements






def triangulation_mask(nodes, elements, mask_func):
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if elements.shape[1] != 3:
        raise ValueError("Elements must have 3 nodes per triangle")

    n_elements = elements.shape[0]
    keep = np.ones(n_elements, dtype=bool)

    for e in range(n_elements):
        tri_nodes = elements[e]
        if np.any(tri_nodes < 0) or np.any(tri_nodes >= nodes.shape[0]):
            keep[e] = False
            continue

        coords = nodes[tri_nodes]
        if mask_func(coords):
            keep[e] = False


    filtered_elements = elements[keep].copy()


    used_nodes = np.unique(filtered_elements)
    if len(used_nodes) == 0:
        return np.zeros((0, nodes.shape[1])), np.zeros((0, 3), dtype=np.int64)


    new_index = {old: new for new, old in enumerate(used_nodes)}
    for e in range(filtered_elements.shape[0]):
        for k in range(3):
            filtered_elements[e, k] = new_index[filtered_elements[e, k]]

    filtered_nodes = nodes[used_nodes].copy()

    return filtered_nodes, filtered_elements






def mesh_to_fem_format(nodes, elements, attributes=None):
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    result = {
        'nodes': nodes,
        'elements': elements,
        'n_nodes': nodes.shape[0],
        'n_elements': elements.shape[0],
        'node_dim': nodes.shape[1],
        'element_order': elements.shape[1]
    }

    if attributes is not None:
        result['element_attributes'] = np.asarray(attributes, dtype=np.float64)
    else:
        result['element_attributes'] = np.zeros(elements.shape[0], dtype=np.float64)

    return result


def compute_triangle_area(nodes, element):
    coords = nodes[element]
    if coords.shape[1] == 2:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    else:

        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        return 0.5 * np.linalg.norm(np.cross(v1, v2))


def mask_disk_jet_region(nodes, elements, r_jet, z_jet_threshold):
    def is_jet(coords):

        centroid = np.mean(coords, axis=0)
        r_c, z_c = centroid
        return (r_c < r_jet) and (abs(z_c) > z_jet_threshold)

    return triangulation_mask(nodes, elements, is_jet)


def mask_black_hole_horizon(nodes, elements, r_isco):
    def is_inside_horizon(coords):
        centroid = np.mean(coords, axis=0)
        r_c = centroid[0]
        return r_c < r_isco

    return triangulation_mask(nodes, elements, is_inside_horizon)
