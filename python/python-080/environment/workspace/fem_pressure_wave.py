
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, gmres, splu
from scipy.spatial import Delaunay


def generate_square_mesh(a, b, h):
    x = np.arange(a, b + h, h)
    y = np.arange(a, b + h, h)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    tri = Delaunay(points)
    elements = tri.simplices


    valid = []
    for elem in elements:
        p0, p1, p2 = points[elem]
        area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        if area > 1e-15:
            valid.append(elem)
    elements = np.array(valid, dtype=int)

    return points, elements


def fem_matrices_2d(nodes, elements, c_sound, rho):
    n_nodes = len(nodes)
    M_data = []
    M_row = []
    M_col = []
    K_data = []
    K_row = []
    K_col = []

    for elem in elements:
        idx = elem
        x = nodes[idx, 0]
        y = nodes[idx, 1]


        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue


        b_coeff = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c_coeff = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])


        Me = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                       [1.0, 2.0, 1.0],
                                       [1.0, 1.0, 2.0]])

        Be = np.vstack([b_coeff, c_coeff]) / (2.0 * area)
        Ke = area * (Be.T @ Be)

        for i in range(3):
            for j in range(3):
                M_row.append(idx[i])
                M_col.append(idx[j])
                M_data.append(Me[i, j])
                K_row.append(idx[i])
                K_col.append(idx[j])
                K_data.append(Ke[i, j])

    M = csr_matrix((M_data, (M_row, M_col)), shape=(n_nodes, n_nodes))
    K = csr_matrix((K_data, (K_row, K_col)), shape=(n_nodes, n_nodes))
    return M, K


def apply_boundary_conditions(M, K, nodes, bubble_center, bubble_radius, p_wall, dt, c_sound):
    n_nodes = len(nodes)
    bc_nodes = []

    for i in range(n_nodes):
        dist = np.linalg.norm(nodes[i] - bubble_center)

        if dist <= bubble_radius * 1.1:
            bc_nodes.append(i)


    bc_nodes = list(set(bc_nodes))


    M_mod = M.copy()
    K_mod = K.copy()

    for i in bc_nodes:

        row_start = M_mod.indptr[i]
        row_end = M_mod.indptr[i + 1]
        M_mod.data[row_start:row_end] = 0.0
        M_mod.data[row_start] = 1.0

        row_start = K_mod.indptr[i]
        row_end = K_mod.indptr[i + 1]
        K_mod.data[row_start:row_end] = 0.0


    if n_nodes <= 2000:
        M_mod = M.toarray()
        K_mod = K.toarray()
        for i in bc_nodes:
            M_mod[i, :] = 0.0
            M_mod[:, i] = 0.0
            M_mod[i, i] = 1.0
            K_mod[i, :] = 0.0
            K_mod[:, i] = 0.0
            K_mod[i, i] = 1.0
        M_mod = csr_matrix(M_mod)
        K_mod = csr_matrix(K_mod)

    return M_mod, K_mod, bc_nodes


def solve_pressure_wave_fem(nodes, elements, c_sound, rho, t_span, dt,
                            bubble_center, bubble_radius_func, p_wall_func,
                            p_init=0.0):
    n_nodes = len(nodes)
    M, K = fem_matrices_2d(nodes, elements, c_sound, rho)


    M_lumped = diags(np.array(M.sum(axis=1)).ravel())

    n_steps = int((t_span[1] - t_span[0]) / dt)
    p = np.full(n_nodes, p_init)
    v = np.zeros(n_nodes)
    a = np.zeros(n_nodes)

    p_history = [p.copy()]


    A_mat = M_lumped + (dt**2 / 4.0) * c_sound**2 * K
    try:
        lu = splu(A_mat.tocsc())
    except RuntimeError:
        lu = None


    pass

    return np.array(p_history)


def acoustic_energy_fem(p, v, nodes, elements, rho, c_sound):
    n_nodes = len(nodes)
    E = 0.0
    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        p_avg = np.mean(p[elem])
        v_avg = np.mean(v[elem]) if hasattr(v, '__getitem__') else 0.0
        E += area * (0.5 * p_avg**2 / (rho * c_sound**2) + 0.5 * rho * v_avg**2)
    return E


def find_boundary_edges(nodes, elements):
    edges = {}
    for elem in elements:
        e = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for edge in e:
            e_sorted = tuple(sorted(edge))
            if e_sorted in edges:
                edges[e_sorted] += 1
            else:
                edges[e_sorted] = 1
    boundary = [e for e, count in edges.items() if count == 1]
    return np.array(boundary)


def pressure_gradient_at_nodes(p, nodes, elements):
    n_nodes = len(nodes)
    grad_p = np.zeros((n_nodes, 2))
    count = np.zeros(n_nodes)

    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        b_coeff = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * area)
        c_coeff = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * area)

        grad_elem = np.array([np.dot(b_coeff, p[elem]), np.dot(c_coeff, p[elem])])
        for i in range(3):
            grad_p[elem[i]] += grad_elem
            count[elem[i]] += 1

    for i in range(n_nodes):
        if count[i] > 0:
            grad_p[i] /= count[i]

    return grad_p
