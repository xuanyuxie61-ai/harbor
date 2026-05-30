
import numpy as np


def assemble_fem_matrices(nodes, elements, boundary_nodes, nu=1.0):
    n_nodes = nodes.shape[0]
    M = np.zeros((n_nodes, n_nodes), dtype=float)
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    B = np.zeros((n_nodes, n_nodes), dtype=float)


    for e in elements:
        i, j, k = e
        p1, p2, p3 = nodes[i], nodes[j], nodes[k]

        b1 = p2[1] - p3[1]
        b2 = p3[1] - p1[1]
        b3 = p1[1] - p2[1]
        c1 = p3[0] - p2[0]
        c2 = p1[0] - p3[0]
        c3 = p2[0] - p1[0]

        area = 0.5 * abs(b1 * c2 - b2 * c1)
        if area < 1.0e-15:
            continue







        raise NotImplementedError("Hole_1: 请实现 FEM 局部质量矩阵、刚度矩阵的组装")






    raise NotImplementedError("Hole_1: 请实现边界控制矩阵 B 的组装")


def assemble_rhs_source(nodes, elements, f_fn, t=0.0):
    n_nodes = nodes.shape[0]
    F = np.zeros(n_nodes, dtype=float)

    for e in elements:
        i, j, k = e
        p1, p2, p3 = nodes[i], nodes[j], nodes[k]
        xc = (p1[0] + p2[0] + p3[0]) / 3.0
        yc = (p1[1] + p2[1] + p3[1]) / 3.0
        f_val = f_fn(xc, yc, t)

        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < 1.0e-15:
            continue



        F[i] += area * f_val / 3.0
        F[j] += area * f_val / 3.0
        F[k] += area * f_val / 3.0

    return F


def evaluate_fem_at_point(nodes, elements, y_coeffs, xq, yq):


    best_bary = None
    best_elem = -1
    min_neg = -1.0e-6

    for idx, e in enumerate(elements):
        p1, p2, p3 = nodes[e]

        denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
        if abs(denom) < 1.0e-15:
            continue
        w1 = ((p2[1] - p3[1]) * (xq - p3[0]) + (p3[0] - p2[0]) * (yq - p3[1])) / denom
        w2 = ((p3[1] - p1[1]) * (xq - p3[0]) + (p1[0] - p3[0]) * (yq - p3[1])) / denom
        w3 = 1.0 - w1 - w2
        if w1 >= min_neg and w2 >= min_neg and w3 >= min_neg:

            w1 = max(w1, 0.0)
            w2 = max(w2, 0.0)
            w3 = max(w3, 0.0)
            s = w1 + w2 + w3
            w1 /= s
            w2 /= s
            w3 /= s
            return w1 * y_coeffs[e[0]] + w2 * y_coeffs[e[1]] + w3 * y_coeffs[e[2]]


    dists = (nodes[:, 0] - xq) ** 2 + (nodes[:, 1] - yq) ** 2
    return y_coeffs[np.argmin(dists)]


def l2_projection(nodes, elements, g_fn, t=0.0):
    M, _, _, _ = assemble_fem_matrices(nodes, elements, [])
    n_nodes = nodes.shape[0]
    b = np.zeros(n_nodes, dtype=float)

    for e in elements:
        i, j, k = e
        p1, p2, p3 = nodes[i], nodes[j], nodes[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < 1.0e-15:
            continue

        xc = (p1[0] + p2[0] + p3[0]) / 3.0
        yc = (p1[1] + p2[1] + p3[1]) / 3.0
        g_val = g_fn(xc, yc, t)

        b[i] += area * g_val / 3.0
        b[j] += area * g_val / 3.0
        b[k] += area * g_val / 3.0

    u = np.linalg.solve(M, b)
    return u


def fem_norm_l2(nodes, elements, y_coeffs):
    M, _, _, _ = assemble_fem_matrices(nodes, elements, [])
    val = np.dot(y_coeffs, M @ y_coeffs)
    return np.sqrt(max(val, 0.0))


def fem_norm_h1(nodes, elements, y_coeffs, nu=1.0):
    M, A, _, _ = assemble_fem_matrices(nodes, elements, [], nu)
    val = np.dot(y_coeffs, M @ y_coeffs) + nu * np.dot(y_coeffs, A @ y_coeffs)
    return np.sqrt(max(val, 0.0))
