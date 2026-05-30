# -*- coding: utf-8 -*-

import numpy as np


def gauss_legendre_2point():
    nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    weights = np.array([1.0, 1.0])
    return nodes, weights


def build_fem_mass_matrix(fem_nodes, elements):
    n_nodes = len(fem_nodes)
    M = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    gp, gw = gauss_legendre_2point()

    for elem in elements:
        i, j = elem
        x_i, x_j = fem_nodes[i], fem_nodes[j]
        h = x_j - x_i
        if abs(h) < 1e-15:
            continue

        x_gp = 0.5 * (x_j + x_i) + 0.5 * h * gp
        w_gp = 0.5 * h * gw

        for q in range(len(gp)):
            xq = x_gp[q]
            wq = w_gp[q]

            phi_i = (x_j - xq) / h
            phi_j = (xq - x_i) / h
            M[i, i] += wq * phi_i * phi_i
            M[i, j] += wq * phi_i * phi_j
            M[j, i] += wq * phi_j * phi_i
            M[j, j] += wq * phi_j * phi_j
    return M


def fem_rhs_projection(fem_nodes, elements, u_func):
    n_nodes = len(fem_nodes)
    b = np.zeros(n_nodes, dtype=np.float64)
    gp, gw = gauss_legendre_2point()

    for elem in elements:
        i, j = elem
        x_i, x_j = fem_nodes[i], fem_nodes[j]
        h = x_j - x_i
        if abs(h) < 1e-15:
            continue
        x_gp = 0.5 * (x_j + x_i) + 0.5 * h * gp
        w_gp = 0.5 * h * gw

        for q in range(len(gp)):
            xq = x_gp[q]
            wq = w_gp[q]
            phi_i = (x_j - xq) / h
            phi_j = (xq - x_i) / h
            uq = u_func(xq)
            b[i] += wq * phi_i * uq
            b[j] += wq * phi_j * uq
    return b


def spectral_to_fem_projection(spectral_nodes, spectral_values,
                                fem_nodes, elements):
    from scipy.interpolate import interp1d

    u_interp = interp1d(spectral_nodes, spectral_values, kind='cubic',
                        fill_value='extrapolate', bounds_error=False)

    M = build_fem_mass_matrix(fem_nodes, elements)
    b = fem_rhs_projection(fem_nodes, elements, u_interp)


    M += 1e-12 * np.eye(len(fem_nodes))
    u_fem = np.linalg.solve(M, b)


    x_test = np.linspace(fem_nodes[0], fem_nodes[-1], 200)
    u_spec_test = u_interp(x_test)
    u_fem_test = np.zeros_like(x_test)
    for elem in elements:
        i, j = elem
        mask = (x_test >= fem_nodes[i]) & (x_test <= fem_nodes[j])
        if np.any(mask):
            h = fem_nodes[j] - fem_nodes[i]
            phi_i = (fem_nodes[j] - x_test[mask]) / h
            phi_j = (x_test[mask] - fem_nodes[i]) / h
            u_fem_test[mask] = phi_i * u_fem[i] + phi_j * u_fem[j]

    l2_error = np.sqrt(np.trapezoid((u_spec_test - u_fem_test) ** 2, x_test))
    return u_fem, l2_error


def build_triangle_neighbors(triangles):
    triangles = np.asarray(triangles, dtype=int)
    n_tri = len(triangles)
    neighbors = np.full((n_tri, 3), -1, dtype=int)


    edge_dict = {}
    for t in range(n_tri):
        verts = triangles[t]
        edges = [(verts[1], verts[2]), (verts[2], verts[0]), (verts[0], verts[1])]
        for e_idx, e in enumerate(edges):
            edge_key = tuple(sorted(e))
            if edge_key in edge_dict:

                other_t, other_e = edge_dict[edge_key]
                neighbors[t, e_idx] = other_t
                neighbors[other_t, other_e] = t
            else:
                edge_dict[edge_key] = (t, e_idx)
    return neighbors


def build_1d_element_neighbors(n_nodes):
    elements = np.zeros((n_nodes - 1, 2), dtype=int)
    for i in range(n_nodes - 1):
        elements[i] = [i, i + 1]
    return elements
