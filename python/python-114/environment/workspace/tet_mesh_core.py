
import numpy as np


def tetrahedron_volume(p1, p2, p3, p4):
    M = np.column_stack((p2 - p1, p3 - p1, p4 - p1))
    vol = np.linalg.det(M) / 6.0
    return abs(vol)


def cayley_menger_volume(p1, p2, p3, p4):
    points = [p1, p2, p3, p4]
    B = np.zeros((5, 5))
    B[0, 1:] = 1.0
    B[1:, 0] = 1.0
    for i in range(4):
        for j in range(4):
            if i == j:
                B[i + 1, j + 1] = 0.0
            else:
                d2 = np.sum((points[i] - points[j]) ** 2)
                B[i + 1, j + 1] = d2
    det_B = np.linalg.det(B)
    if det_B < 0:

        det_B = 0.0
    vol = np.sqrt(det_B / 288.0)
    return vol


def generate_tet_mesh_box(nx=4, ny=4, nz=4, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Grid divisions must be at least 2 in each dimension.")

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
    nodes = np.array(nodes)



    elements = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v000 = node_index[(i, j, k)]
                v100 = node_index[(i + 1, j, k)]
                v010 = node_index[(i, j + 1, k)]
                v110 = node_index[(i + 1, j + 1, k)]
                v001 = node_index[(i, j, k + 1)]
                v101 = node_index[(i + 1, j, k + 1)]
                v011 = node_index[(i, j + 1, k + 1)]
                v111 = node_index[(i + 1, j + 1, k + 1)]


                elements.append([v000, v100, v110, v111])
                elements.append([v000, v100, v111, v101])
                elements.append([v000, v101, v111, v001])
                elements.append([v000, v111, v011, v001])
                elements.append([v000, v011, v111, v010])
                elements.append([v000, v010, v111, v110])

    elements = np.array(elements, dtype=int)
    return nodes, elements


def integrate_over_tet_mesh(nodes, elements, nodal_values):
    if elements.shape[1] not in (4, 10):
        raise ValueError("Only 4-node or 10-node tetrahedra supported.")

    integral = 0.0
    total_volume = 0.0
    for e in range(elements.shape[0]):
        en = elements[e, :4]
        p = nodes[en]

        M = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = abs(np.linalg.det(M)) / 6.0
        if vol < 0:
            vol = 0.0
        avg_val = np.mean(nodal_values[en])
        integral += vol * avg_val
        total_volume += vol

    return integral, total_volume


def integrate_vector_over_tet_mesh(nodes, elements, nodal_values):
    if elements.shape[1] not in (4, 10):
        raise ValueError("Only 4-node or 10-node tetrahedra supported.")

    D = nodal_values.shape[1] if nodal_values.ndim > 1 else 1
    integral = np.zeros(D)
    total_volume = 0.0

    for e in range(elements.shape[0]):
        en = elements[e, :4]
        p = nodes[en]
        M = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = abs(np.linalg.det(M)) / 6.0
        if vol < 0:
            vol = 0.0
        if D == 1:
            avg_val = np.mean(nodal_values[en])
            integral[0] += vol * avg_val
        else:
            avg_val = np.mean(nodal_values[en, :], axis=0)
            integral += vol * avg_val
        total_volume += vol

    return integral, total_volume


def tet_mesh_gradients(nodes, elements, nodal_values):
    M = elements.shape[0]
    gradients = np.zeros((M, 3))
    for e in range(M):
        en = elements[e, :4]
        p = nodes[en]
        X = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        det_X = np.linalg.det(X)
        if abs(det_X) < 1e-14:
            gradients[e] = 0.0
            continue
        du = np.array([
            nodal_values[en[1]] - nodal_values[en[0]],
            nodal_values[en[2]] - nodal_values[en[0]],
            nodal_values[en[3]] - nodal_values[en[0]],
        ])
        try:
            grad = np.linalg.solve(X.T, du)
        except np.linalg.LinAlgError:
            grad = np.zeros(3)
        gradients[e] = grad
    return gradients
