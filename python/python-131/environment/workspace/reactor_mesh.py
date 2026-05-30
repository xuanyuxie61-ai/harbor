
import numpy as np






def mesh_boundary_segments(element_node):
    element_node = np.asarray(element_node, dtype=int)
    n_elements, n_vertices = element_node.shape

    n_segments = n_elements * n_vertices
    segments = np.zeros((n_segments, 2), dtype=int)

    s = 0
    for e in range(n_elements):
        j = n_vertices - 1
        for jp1 in range(n_vertices):
            segments[s, 0] = element_node[e, j]
            segments[s, 1] = element_node[e, jp1]
            j = jp1
            s += 1


    segments_sorted = np.sort(segments, axis=1)

    seg_dict = {}
    for i in range(n_segments):
        key = (segments_sorted[i, 0], segments_sorted[i, 1])

        direction = 1 if segments[i, 0] < segments[i, 1] else -1
        if key not in seg_dict:
            seg_dict[key] = [direction]
        else:
            seg_dict[key].append(direction)

    boundary = []
    for key, dirs in seg_dict.items():

        if len(dirs) == 1:
            if dirs[0] == 1:
                boundary.append([key[0], key[1]])
            else:
                boundary.append([key[1], key[0]])

    if not boundary:
        return np.empty((0, 2), dtype=int)

    boundary = np.array(boundary, dtype=int)


    n_b = boundary.shape[0]
    for b1 in range(n_b - 1):
        for b2 in range(b1 + 1, n_b):
            if boundary[b2, 0] == boundary[b1, 1]:
                boundary[[b1 + 1, b2]] = boundary[[b2, b1 + 1]]
                break

    return boundary






def boundary_perturb(p, mu=0.1, seed=42):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    n, d = p.shape

    sig = mu ** 2
    w = mu + sig * rng.standard_normal(n)
    w = np.clip(w, -0.5, 0.5)


    p_prev = np.roll(p, 1, axis=0)
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)

    w_col = w.reshape(-1, 1)
    perturb = (0.5 * (p + p_prev)
               + w_col * (p + p_prev)
               - w_col * (p_next + p_next2))

    perturb = np.roll(perturb, -1, axis=0)

    q = np.zeros((2 * n, d))
    q[0:2 * n:2] = p
    q[1:2 * n:2] = perturb

    return q






def generate_cylindrical_mesh(R, H, Nr, Nz):
    dr = R / Nr
    dz = H / Nz

    n_nodes = (Nr + 1) * (Nz + 1)
    nodes = np.zeros((n_nodes, 2))

    for j in range(Nz + 1):
        for i in range(Nr + 1):
            idx = j * (Nr + 1) + i
            nodes[idx, 0] = i * dr
            nodes[idx, 1] = j * dz

    n_elements = Nr * Nz
    elements = np.zeros((n_elements, 4), dtype=int)

    for j in range(Nz):
        for i in range(Nr):
            e = j * Nr + i
            n1 = j * (Nr + 1) + i
            n2 = n1 + 1
            n4 = (j + 1) * (Nr + 1) + i
            n3 = n4 + 1
            elements[e] = [n1, n2, n3, n4]

    return nodes, elements


def compute_jacobian_2d(nodes, element):
    x = nodes[element, 0]
    y = nodes[element, 1]



    dx1 = x[2] - x[0]
    dy1 = y[2] - y[0]
    dx2 = x[3] - x[1]
    dy2 = y[3] - y[1]

    J = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
    return J


def mesh_quality_report(nodes, elements):
    jac_values = []
    for e in elements:
        jac = compute_jacobian_2d(nodes, e)
        jac_values.append(jac)

    jac_values = np.array(jac_values)
    boundary = mesh_boundary_segments(elements)

    return {
        'jacobian_min': float(np.min(jac_values)),
        'jacobian_mean': float(np.mean(jac_values)),
        'jacobian_negative_count': int(np.sum(jac_values <= 0)),
        'n_elements': elements.shape[0],
        'n_nodes': nodes.shape[0],
        'n_boundary_segments': boundary.shape[0],
    }
