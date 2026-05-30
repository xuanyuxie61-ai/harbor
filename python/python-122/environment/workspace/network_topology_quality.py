
import numpy as np
import math


def q_measure_triangle(z, triangle_node):
    nt = triangle_node.shape[0]
    if nt < 1:
        return -1.0

    q_min = np.inf
    for tri in range(nt):
        a_idx, b_idx, c_idx = triangle_node[tri]
        a = np.linalg.norm(z[a_idx] - z[b_idx])
        b_len = np.linalg.norm(z[b_idx] - z[c_idx])
        c_len = np.linalg.norm(z[c_idx] - z[a_idx])

        if a * b_len * c_len < 1e-14:
            continue
        q = (b_len + c_len - a) * (c_len + a - b_len) * (a + b_len - c_len) / (a * b_len * c_len)
        q_min = min(q_min, q)

    return q_min if q_min < np.inf else -1.0


def tetrahedron_quality(p, t):
    nt = t.shape[0]
    if nt < 1:
        return -1.0
    q_min = np.inf
    for i in range(nt):
        idx = t[i]
        v0, v1, v2, v3 = p[idx[0]], p[idx[1]], p[idx[2]], p[idx[3]]
        mat = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        vol = abs(np.linalg.det(mat)) / 6.0
        edges = [np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v0), np.linalg.norm(v3 - v0),
                 np.linalg.norm(v2 - v1), np.linalg.norm(v3 - v1), np.linalg.norm(v3 - v2)]
        sum_sq = sum(e ** 2 for e in edges)
        if sum_sq < 1e-14:
            continue
        q = 6.0 * np.sqrt(2.0) * vol / (sum_sq ** 1.5)
        q_min = min(q_min, q)
    return q_min if q_min < np.inf else -1.0


def radius_maximus(dim_num, n, z, walls=True):
    z = np.asarray(z, dtype=float)
    radius = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(z[i] - z[j]) / 2.0
            radius[i] = min(radius[i], dist)
        if walls:

            for d in range(dim_num):
                radius[i] = min(radius[i], z[i, d], 1.0 - z[i, d])
    radius = np.where(np.isinf(radius), 0.0, radius)
    return radius


def sphere_volume_nd(dim_num, r):
    r = np.asarray(r, dtype=float)
    r_safe = np.where(r < 0, 0.0, r)
    return np.pi ** (dim_num / 2.0) * r_safe ** dim_num / math.gamma(dim_num / 2.0 + 1.0)


def sphere_measure(dim_num, n, z, walls=True):
    z = np.asarray(z, dtype=float)
    if z.shape[0] != n or z.shape[1] != dim_num:
        return -1.0
    if np.any(z < 0.0) or np.any(z > 1.0):
        return -1.0

    radius = radius_maximus(dim_num, n, z, walls=walls)
    s = 0.0
    for j in range(n):
        s += sphere_volume_nd(dim_num, radius[j])
    return s


def bandwidth_mesh(triangle_node, n_nodes):
    nt = triangle_node.shape[0]
    bw = 0
    for tri in range(nt):
        nodes = triangle_node[tri]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                diff = abs(nodes[i] - nodes[j])
                bw = max(bw, diff)
    return bw


def alpha_measure(z, triangle_node):
    nt = triangle_node.shape[0]
    if nt < 1:
        return -1.0
    alpha_min = np.inf
    for tri in range(nt):
        a_idx, b_idx, c_idx = triangle_node[tri]
        a = np.linalg.norm(z[b_idx] - z[c_idx])
        b_len = np.linalg.norm(z[a_idx] - z[c_idx])
        c_len = np.linalg.norm(z[a_idx] - z[b_idx])

        cos_a = (b_len ** 2 + c_len ** 2 - a ** 2) / (2.0 * b_len * c_len + 1e-14)
        cos_b = (a ** 2 + c_len ** 2 - b_len ** 2) / (2.0 * a * c_len + 1e-14)
        cos_c = (a ** 2 + b_len ** 2 - c_len ** 2) / (2.0 * a * b_len + 1e-14)
        angles = [np.arccos(np.clip(cos_a, -1.0, 1.0)),
                  np.arccos(np.clip(cos_b, -1.0, 1.0)),
                  np.arccos(np.clip(cos_c, -1.0, 1.0))]
        min_angle = min(angles)
        max_angle = max(angles)
        if max_angle < 1e-14:
            continue
        alpha = min_angle / max_angle
        alpha_min = min(alpha_min, alpha)
    return alpha_min if alpha_min < np.inf else -1.0


def vascular_network_quality_report(nodes, triangles, tetrahedra=None):
    report = {}
    report['q_measure'] = q_measure_triangle(nodes, triangles)
    report['alpha_measure'] = alpha_measure(nodes, triangles)
    report['bandwidth'] = bandwidth_mesh(triangles, nodes.shape[0])
    report['sphere_measure_2d'] = sphere_measure(2, nodes.shape[0],
                                                  (nodes - np.min(nodes, axis=0)) /
                                                  (np.max(nodes, axis=0) - np.min(nodes, axis=0) + 1e-14))
    if tetrahedra is not None:
        report['tetra_quality'] = tetrahedron_quality(nodes, tetrahedra)
    return report
