
import numpy as np


def r8mat_det_4d(a):
    det = (
        a[0, 0] * (
            a[1, 1] * (a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2])
            - a[1, 2] * (a[2, 1] * a[3, 3] - a[2, 3] * a[3, 1])
            + a[1, 3] * (a[2, 1] * a[3, 2] - a[2, 2] * a[3, 1])
        )
        - a[0, 1] * (
            a[1, 0] * (a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2])
            - a[1, 2] * (a[2, 0] * a[3, 3] - a[2, 3] * a[3, 0])
            + a[1, 3] * (a[2, 0] * a[3, 2] - a[2, 2] * a[3, 0])
        )
        + a[0, 2] * (
            a[1, 0] * (a[2, 1] * a[3, 3] - a[2, 3] * a[3, 1])
            - a[1, 1] * (a[2, 0] * a[3, 3] - a[2, 3] * a[3, 0])
            + a[1, 3] * (a[2, 0] * a[3, 1] - a[2, 1] * a[3, 0])
        )
        - a[0, 3] * (
            a[1, 0] * (a[2, 1] * a[3, 2] - a[2, 2] * a[3, 1])
            - a[1, 1] * (a[2, 0] * a[3, 2] - a[2, 2] * a[3, 0])
            + a[1, 2] * (a[2, 0] * a[3, 1] - a[2, 1] * a[3, 0])
        )
    )
    return det


def tetrahedron_volume(p1, p2, p3, p4):
    M = np.array([
        [p2[0] - p1[0], p3[0] - p1[0], p4[0] - p1[0]],
        [p2[1] - p1[1], p3[1] - p1[1], p4[1] - p1[1]],
        [p2[2] - p1[2], p3[2] - p1[2], p4[2] - p1[2]]
    ], dtype=np.float64)
    return abs(np.linalg.det(M)) / 6.0


def tetrahedron_quality_measure_1(p1, p2, p3, p4):
    edges = [
        p2 - p1, p3 - p1, p4 - p1, p3 - p2, p4 - p2, p4 - p3
    ]
    sum_sq = sum(np.dot(e, e) for e in edges)
    if sum_sq < 1e-14:
        return 0.0
    V = tetrahedron_volume(p1, p2, p3, p4)
    Q = 72.0 * np.sqrt(3.0) * V / (sum_sq ** 1.5)
    return Q


def tetrahedron_quality_measure_2(p1, p2, p3, p4):
    V = tetrahedron_volume(p1, p2, p3, p4)
    if V < 1e-14:
        return 0.0


    def triangle_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    A1 = triangle_area(p2, p3, p4)
    A2 = triangle_area(p1, p3, p4)
    A3 = triangle_area(p1, p2, p4)
    A4 = triangle_area(p1, p2, p3)
    S = A1 + A2 + A3 + A4

    if S < 1e-14:
        return 0.0

    r_in = 3.0 * V / S



    A_mat = np.array([
        2.0 * (p2 - p1),
        2.0 * (p3 - p1),
        2.0 * (p4 - p1)
    ], dtype=np.float64)
    b_vec = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1),
        np.dot(p4, p4) - np.dot(p1, p1)
    ], dtype=np.float64)

    try:
        O = np.linalg.solve(A_mat, b_vec)
        r_out = np.linalg.norm(O - p1)
    except np.linalg.LinAlgError:
        return 0.0

    if r_out < 1e-14:
        return 0.0

    return r_in / r_out


def evaluate_mesh_quality(node_xyz, tetra_nodes):
    n_tetra = len(tetra_nodes)
    q1_values = np.zeros(n_tetra, dtype=np.float64)
    q2_values = np.zeros(n_tetra, dtype=np.float64)
    volumes = np.zeros(n_tetra, dtype=np.float64)

    for t in range(n_tetra):
        idx = tetra_nodes[t]

        if np.any(idx < 0) or np.any(idx >= len(node_xyz)):
            q1_values[t] = 0.0
            q2_values[t] = 0.0
            volumes[t] = 0.0
            continue

        p1 = node_xyz[idx[0]]
        p2 = node_xyz[idx[1]]
        p3 = node_xyz[idx[2]]
        p4 = node_xyz[idx[3]]

        q1_values[t] = tetrahedron_quality_measure_1(p1, p2, p3, p4)
        q2_values[t] = tetrahedron_quality_measure_2(p1, p2, p3, p4)
        volumes[t] = tetrahedron_volume(p1, p2, p3, p4)

    return {
        'q1_min': float(np.min(q1_values)),
        'q1_mean': float(np.mean(q1_values)),
        'q1_max': float(np.max(q1_values)),
        'q1_var': float(np.var(q1_values)),
        'q2_min': float(np.min(q2_values)),
        'q2_mean': float(np.mean(q2_values)),
        'q2_max': float(np.max(q2_values)),
        'q2_var': float(np.var(q2_values)),
        'volume_total': float(np.sum(volumes)),
        'volume_min': float(np.min(volumes[volumes > 0]) if np.any(volumes > 0) else 0.0),
        'volume_max': float(np.max(volumes)),
        'n_tetra': n_tetra
    }


def mesh_base_one(node_num, element_order, element_num, element_node):
    en = np.asarray(element_node, dtype=np.int64)
    node_min = np.min(en)
    node_max = np.max(en)

    if node_min == 0 and node_max == node_num - 1:

        en = en + 1
    elif node_min == 1 and node_max == node_num:

        pass
    else:

        if node_min == 0:
            en = en + 1

    return en


def generate_earth_tetrahedral_mesh(n_r=8, n_theta=12, n_phi=12):
    from constants import EARTH_RADIUS_KM

    nodes = []
    node_map = {}


    node_map[(0, 0, 0)] = 0
    nodes.append([0.0, 0.0, 0.0])

    for ir in range(1, n_r):
        r = EARTH_RADIUS_KM * ir / (n_r - 1)
        for it in range(n_theta):
            theta = np.pi * it / (n_theta - 1) if n_theta > 1 else np.pi / 2
            for ip in range(n_phi):
                phi = 2.0 * np.pi * ip / n_phi
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                node_map[(ir, it, ip)] = len(nodes)
                nodes.append([x, y, z])

    nodes = np.array(nodes, dtype=np.float64)


    elements = []

    if n_r > 1:
        for it in range(n_theta - 1):
            for ip in range(n_phi):
                ip_next = (ip + 1) % n_phi

                n0 = 0
                n1 = node_map.get((1, it, ip), 0)
                n2 = node_map.get((1, it, ip_next), 0)
                n3 = node_map.get((1, min(it + 1, n_theta - 1), ip), 0)
                if n1 > 0 and n2 > 0 and n3 > 0:
                    elements.append([n0, n1, n2, n3])

    elements = np.array(elements, dtype=np.int64)
    return nodes, elements


def parse_mesh_data(mesh_text_lines):
    result = {
        'vertices': [],
        'tetrahedrons': []
    }

    mode = 'none'
    count = 0
    read_count = 0

    for line in mesh_text_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.upper() == 'VERTICES':
            mode = 'vertices_count'
            continue
        elif line.upper() == 'TETRAHEDRA':
            mode = 'tetra_count'
            continue
        elif line.upper() == 'END':
            break

        if mode == 'vertices_count':
            parts = line.split()
            count = int(parts[0])
            read_count = 0
            mode = 'vertices_read'
        elif mode == 'vertices_read':
            parts = line.split()
            if len(parts) >= 3:
                result['vertices'].append([
                    float(parts[0]), float(parts[1]), float(parts[2])
                ])
                read_count += 1
                if read_count >= count:
                    mode = 'none'
        elif mode == 'tetra_count':
            parts = line.split()
            count = int(parts[0])
            read_count = 0
            mode = 'tetra_read'
        elif mode == 'tetra_read':
            parts = line.split()
            if len(parts) >= 4:
                result['tetrahedrons'].append([
                    int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                ])
                read_count += 1
                if read_count >= count:
                    mode = 'none'

    result['vertices'] = np.array(result['vertices'], dtype=np.float64)
    result['tetrahedrons'] = np.array(result['tetrahedrons'], dtype=np.int64)
    return result
