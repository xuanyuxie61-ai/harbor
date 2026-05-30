
import numpy as np


def refine_triangulation(nodes, triangles, refine_flags=None):
    n_nodes = nodes.shape[0]
    n_tri = triangles.shape[0]

    if refine_flags is None:
        refine_flags = np.ones(n_tri, dtype=bool)
    else:
        refine_flags = np.array(refine_flags, dtype=bool)


    edge_midpoint = {}
    midpoint_index = {}

    def get_midpoint(i, j):
        key = tuple(sorted([i, j]))
        if key not in midpoint_index:
            mid = (nodes[i] + nodes[j]) / 2.0
            idx = n_nodes + len(edge_midpoint)
            edge_midpoint[key] = mid
            midpoint_index[key] = idx
        return midpoint_index[key]

    new_triangles_list = []
    parent_map_list = []

    for t in range(n_tri):
        v1, v2, v3 = triangles[t]

        if refine_flags[t]:
            m12 = get_midpoint(v1, v2)
            m23 = get_midpoint(v2, v3)
            m31 = get_midpoint(v3, v1)

            m12_idx = midpoint_index[tuple(sorted([v1, v2]))]
            m23_idx = midpoint_index[tuple(sorted([v2, v3]))]
            m31_idx = midpoint_index[tuple(sorted([v3, v1]))]

            new_triangles_list.append([v1, m12_idx, m31_idx])
            new_triangles_list.append([m12_idx, v2, m23_idx])
            new_triangles_list.append([m31_idx, m23_idx, v3])
            new_triangles_list.append([m12_idx, m23_idx, m31_idx])

            parent_map_list.extend([t, t, t, t])
        else:
            new_triangles_list.append([v1, v2, v3])
            parent_map_list.append(t)


    n_new_midpoints = len(edge_midpoint)
    new_nodes = np.zeros((n_nodes + n_new_midpoints, 2))
    new_nodes[:n_nodes] = nodes

    node_level = np.zeros(n_nodes + n_new_midpoints, dtype=int)

    node_level[:n_nodes] = 0
    node_level[n_nodes:] = 1

    for key, idx in midpoint_index.items():
        new_nodes[idx] = edge_midpoint[key]

    new_triangles = np.array(new_triangles_list, dtype=int)
    parent_map = np.array(parent_map_list, dtype=int)

    return new_nodes, new_triangles, parent_map, node_level


def refine_marked_elements(nodes, triangles, element_errors, threshold_ratio=0.5):
    max_error = np.max(element_errors)
    if max_error < 1e-15:

        return nodes.copy(), triangles.copy(), np.arange(len(triangles)), np.zeros(len(nodes), dtype=int), 0

    threshold = threshold_ratio * max_error
    refine_flags = element_errors >= threshold

    refined_count = np.sum(refine_flags)

    new_nodes, new_triangles, parent_map, node_level = refine_triangulation(
        nodes, triangles, refine_flags
    )

    return new_nodes, new_triangles, parent_map, node_level, int(refined_count)


def coarsen_mesh(nodes, triangles, coarsen_flags):
    coarsen_flags = np.array(coarsen_flags, dtype=bool)
    keep_triangles = ~coarsen_flags

    if np.all(keep_triangles):
        return nodes.copy(), triangles.copy(), np.arange(len(nodes))


    node_usage = np.zeros(len(nodes), dtype=int)
    for tri in triangles[keep_triangles]:
        node_usage[tri] += 1

    keep_nodes = node_usage > 0
    node_remap = np.cumsum(keep_nodes) - 1
    node_remap[~keep_nodes] = -1

    new_nodes = nodes[keep_nodes]
    new_triangles = node_remap[triangles[keep_triangles]]


    valid = np.all(new_triangles >= 0, axis=1)
    new_triangles = new_triangles[valid]

    return new_nodes, new_triangles, node_remap


def compute_mesh_quality(nodes, triangles):
    n_tri = len(triangles)
    quality = np.zeros(n_tri)

    for t in range(n_tri):
        p1 = nodes[triangles[t, 0]]
        p2 = nodes[triangles[t, 1]]
        p3 = nodes[triangles[t, 2]]

        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        denom = a ** 2 + b ** 2 + c ** 2
        if denom > 1e-14:
            quality[t] = 4.0 * np.sqrt(3.0) * area / denom
        else:
            quality[t] = 0.0

    return quality
