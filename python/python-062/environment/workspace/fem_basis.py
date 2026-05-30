
import numpy as np


def tetrahedron_volume(nodes):
    if nodes.shape != (4, 3):
        raise ValueError("tetrahedron_volume: nodes 形状必须为 (4, 3)")

    M = np.vstack([nodes.T, np.ones(4)])
    volume = abs(np.linalg.det(M)) / 6.0

    if volume < 1e-15:
        raise ValueError("tetrahedron_volume: 四面体体积过小或退化")

    return volume


def basis_mn_tet4(nodes, points):
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        points = points.T

    n = points.shape[0]
    phi = np.zeros((4, n), dtype=np.float64)


    M_full = np.vstack([nodes.T, np.ones(4)])
    vol_signed = np.linalg.det(M_full) / 6.0

    if abs(vol_signed) < 1e-15:
        raise ValueError("basis_mn_tet4: 四面体体积退化")


    for i in range(4):
        for j in range(n):
            sub_nodes = np.copy(nodes)
            sub_nodes[i] = points[j]
            M = np.vstack([sub_nodes.T, np.ones(4)])
            det_i = np.linalg.det(M)
            phi[i, j] = (det_i / 6.0) / vol_signed


    phi = np.clip(phi, 0.0, 1.0)
    col_sum = phi.sum(axis=0)
    col_sum = np.where(np.abs(col_sum) < 1e-15, 1.0, col_sum)
    phi = phi / col_sum

    return phi


def basis_gradient_tet4(nodes):
    volume = tetrahedron_volume(nodes)

    grad_phi = np.zeros((4, 3), dtype=np.float64)

    for i in range(4):

        idx = [k for k in range(4) if k != i]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]


        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)


        grad_phi[i] = normal / (6.0 * volume)


        center = nodes.mean(axis=0)
        to_center = center - nodes[i]
        if np.dot(grad_phi[i], to_center) > 0:
            grad_phi[i] = -grad_phi[i]

    return grad_phi


def fem_laplacian_matrix(nodes, element_nodes, nu_eff):
    n_node = nodes.shape[0]
    n_elem = element_nodes.shape[0]
    L = np.zeros((n_node, n_node), dtype=np.float64)

    for e in range(n_elem):
        en = element_nodes[e]
        elem_nodes = nodes[en]

        vol = tetrahedron_volume(elem_nodes)
        grad = basis_gradient_tet4(elem_nodes)


        for i_loc in range(4):
            for j_loc in range(4):
                i_glob = en[i_loc]
                j_glob = en[j_loc]
                L[i_glob, j_glob] += nu_eff * vol * np.dot(grad[i_loc], grad[j_loc])

    return L
