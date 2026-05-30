import numpy as np


def lagrange_basis_1d(x_nodes, x_query):
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    n = len(x_nodes)

    scalar_input = np.isscalar(x_query)
    xq = np.atleast_1d(x_query).astype(np.float64)

    phi = np.zeros((n, len(xq)), dtype=np.float64)

    for k in range(n):

        denom = 1.0
        for j in range(n):
            if j != k:
                diff = x_nodes[k] - x_nodes[j]
                if abs(diff) < 1e-15:
                    diff = 1e-15
                denom *= diff

        num = np.ones(len(xq), dtype=np.float64)
        for j in range(n):
            if j != k:
                num *= (xq - x_nodes[j])

        phi[k, :] = num / denom

    if scalar_input:
        return phi[:, 0]
    return phi


def lagrange_basis_derivative_1d(x_nodes, x_query):
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    n = len(x_nodes)
    xq = np.atleast_1d(x_query).astype(np.float64)

    dphi = np.zeros((n, len(xq)), dtype=np.float64)
    phi = lagrange_basis_1d(x_nodes, xq)

    for k in range(n):
        for j in range(n):
            if j != k:
                diff = xq - x_nodes[j]
                diff = np.where(np.abs(diff) < 1e-15, 1e-15, diff)
                dphi[k, :] += phi[k, :] / diff

    return dphi


def fem1d_mass_matrix(x_nodes_per_element, element_connectivity):
    n_elements = len(element_connectivity)
    n_global = np.max(element_connectivity) + 1

    M = {}


    gauss_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    for e in range(n_elements):
        nodes_e = x_nodes_per_element[e]
        conn = element_connectivity[e]
        n_loc = len(conn)


        x1, x2 = nodes_e[0], nodes_e[-1]
        jac = 0.5 * (x2 - x1)


        x_phys = 0.5 * (x2 - x1) * gauss_pts + 0.5 * (x2 + x1)

        phi = lagrange_basis_1d(nodes_e, x_phys)

        for i_loc in range(n_loc):
            I = conn[i_loc]
            for j_loc in range(n_loc):
                J = conn[j_loc]
                val = np.sum(gauss_wts * phi[i_loc, :] * phi[j_loc, :]) * jac
                key = (min(I, J), max(I, J))
                M[key] = M.get(key, 0.0) + val

    return M


def fem1d_stiffness_matrix(x_nodes_per_element, element_connectivity):
    n_elements = len(element_connectivity)
    n_global = np.max(element_connectivity) + 1

    K = {}

    gauss_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    for e in range(n_elements):
        nodes_e = x_nodes_per_element[e]
        conn = element_connectivity[e]
        n_loc = len(conn)

        x1, x2 = nodes_e[0], nodes_e[-1]
        jac = 0.5 * (x2 - x1)

        x_phys = 0.5 * (x2 - x1) * gauss_pts + 0.5 * (x2 + x1)

        dphi = lagrange_basis_derivative_1d(nodes_e, x_phys)

        for i_loc in range(n_loc):
            I = conn[i_loc]
            for j_loc in range(n_loc):
                J = conn[j_loc]
                val = np.sum(gauss_wts * dphi[i_loc, :] * dphi[j_loc, :]) * jac
                key = (min(I, J), max(I, J))
                K[key] = K.get(key, 0.0) + val

    return K


def solve_fem1d_radial(r_in, r_out, n_elements, order=2,
                       source_func=None, bc_left=None, bc_right=None):
    if source_func is None:
        source_func = lambda r: 0.0

    if bc_left is None:
        bc_left = {'type': 'dirichlet', 'value': 0.0}
    if bc_right is None:
        bc_right = {'type': 'neumann', 'value': 0.0}


    n_nodes = n_elements * order + 1
    r_nodes = np.linspace(r_in, r_out, n_nodes)


    element_nodes = []
    connectivity = []
    for e in range(n_elements):
        start = e * order
        conn = np.arange(start, start + order + 1)
        connectivity.append(conn)
        element_nodes.append(r_nodes[conn])


    n_global = n_nodes
    A_mat = np.zeros((n_global, n_global), dtype=np.float64)
    b_vec = np.zeros(n_global, dtype=np.float64)

    gauss_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    for e in range(n_elements):
        nodes_e = element_nodes[e]
        conn = connectivity[e]
        n_loc = len(conn)

        x1, x2 = nodes_e[0], nodes_e[-1]
        jac = 0.5 * (x2 - x1)
        x_phys = 0.5 * (x2 - x1) * gauss_pts + 0.5 * (x2 + x1)

        phi = lagrange_basis_1d(nodes_e, x_phys)
        dphi = lagrange_basis_derivative_1d(nodes_e, x_phys)

        for i_loc in range(n_loc):
            I = conn[i_loc]
            for j_loc in range(n_loc):
                J = conn[j_loc]

                stiff_val = np.sum(gauss_wts * x_phys * dphi[i_loc, :] * dphi[j_loc, :]) * jac

                mass_val = 0.01 * np.sum(gauss_wts * phi[i_loc, :] * phi[j_loc, :]) * jac
                A_mat[I, J] += stiff_val + mass_val


            source_vals = np.array([source_func(x) for x in x_phys])
            b_vec[I] += np.sum(gauss_wts * phi[i_loc, :] * source_vals) * jac


    if bc_left['type'] == 'dirichlet':
        A_mat[0, :] = 0.0
        A_mat[0, 0] = 1.0
        b_vec[0] = bc_left['value']
    elif bc_left['type'] == 'neumann':
        b_vec[0] += bc_left['value']

    if bc_right['type'] == 'dirichlet':
        A_mat[-1, :] = 0.0
        A_mat[-1, -1] = 1.0
        b_vec[-1] = bc_right['value']
    elif bc_right['type'] == 'neumann':
        b_vec[-1] += bc_right['value']


    Sigma = np.linalg.solve(A_mat, b_vec)

    return r_nodes, Sigma


def fem_interpolate_1d(r_nodes, values, r_query):
    r_nodes = np.asarray(r_nodes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    r_query = np.asarray(r_query, dtype=np.float64)

    result = np.zeros(len(r_query), dtype=np.float64)

    for i, rq in enumerate(r_query):

        if rq <= r_nodes[0]:
            result[i] = values[0]
        elif rq >= r_nodes[-1]:
            result[i] = values[-1]
        else:
            idx = np.searchsorted(r_nodes, rq) - 1
            idx = max(0, min(idx, len(r_nodes) - 2))
            r1, r2 = r_nodes[idx], r_nodes[idx + 1]
            t = (rq - r1) / (r2 - r1) if abs(r2 - r1) > 1e-15 else 0.0
            result[i] = (1.0 - t) * values[idx] + t * values[idx + 1]

    return result
