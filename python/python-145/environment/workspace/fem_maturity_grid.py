
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def generate_rectangular_grid(nx, ny, xl=0.0, xr=1.0, yb=0.0, yt=1.0):
    if nx < 2 or ny < 2:
        raise ValueError("generate_rectangular_grid: nx, ny 必须至少为 2")

    node_num = (2 * nx - 1) * (2 * ny - 1)
    element_num = (nx - 1) * (ny - 1) * 2

    node_xy = np.zeros((node_num, 2), dtype=float)


    dx = (xr - xl) / (nx - 1)
    dy = (yt - yb) / (ny - 1)

    node = 0
    for j in range(2 * ny - 1):
        for i in range(2 * nx - 1):
            if j % 2 == 0 and i % 2 == 0:

                x = xl + (i // 2) * dx
                y = yb + (j // 2) * dy
            elif j % 2 == 0 and i % 2 == 1:

                x = xl + (i // 2) * dx + dx / 2
                y = yb + (j // 2) * dy
            elif j % 2 == 1 and i % 2 == 0:

                x = xl + (i // 2) * dx
                y = yb + (j // 2) * dy + dy / 2
            else:

                x = xl + (i // 2) * dx + dx / 2
                y = yb + (j // 2) * dy + dy / 2
            node_xy[node, 0] = x
            node_xy[node, 1] = y
            node += 1


    element_node = np.zeros((element_num, 6), dtype=int)
    element = 0
    row_nodes = 2 * nx - 1

    for j in range(ny - 1):
        for i in range(nx - 1):
            sw = j * 2 * row_nodes + 2 * i
            w = sw + 1
            nw = sw + 2
            s = sw + row_nodes
            c = s + 1
            n = s + 2
            se = s + row_nodes
            e = se + 1
            ne = se + 2


            element_node[element, :] = [sw, se, nw, s, c, w]
            element += 1

            element_node[element, :] = [ne, nw, se, n, c, e]
            element += 1

    return node_xy, element_node


def triangle_area(p1, p2, p3):
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def reference_to_physical_t3(t3, quad_xy):
    t3 = np.asarray(t3, dtype=float)
    quad_xy = np.asarray(quad_xy, dtype=float)
    nq = quad_xy.shape[0]
    xy = np.zeros((nq, 2), dtype=float)
    for q in range(nq):
        r, s = quad_xy[q, 0], quad_xy[q, 1]
        xy[q, 0] = t3[0, 0] + (t3[1, 0] - t3[0, 0]) * r + (t3[2, 0] - t3[0, 0]) * s
        xy[q, 1] = t3[0, 1] + (t3[1, 1] - t3[0, 1]) * r + (t3[2, 1] - t3[0, 1]) * s
    return xy


def basis_11_t6(t6, i, p):
    if i < 1 or i > 6:
        raise ValueError("basis_11_t6: i 必须在 1..6 之间")
    t6 = np.asarray(t6, dtype=float)
    p = np.asarray(p, dtype=float)


    x1, y1 = t6[0, 0], t6[0, 1]
    x2, y2 = t6[1, 0], t6[1, 1]
    x3, y3 = t6[2, 0], t6[2, 1]

    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(area2) < 1e-14:
        raise ValueError("basis_11_t6: 三角形退化")


    L1 = ((x2 - p[0]) * (y3 - p[1]) - (x3 - p[0]) * (y2 - p[1])) / area2
    L2 = ((x3 - p[0]) * (y1 - p[1]) - (x1 - p[0]) * (y3 - p[1])) / area2
    L3 = 1.0 - L1 - L2


    dL1dx = (y2 - y3) / area2
    dL1dy = (x3 - x2) / area2
    dL2dx = (y3 - y1) / area2
    dL2dy = (x1 - x3) / area2
    dL3dx = -dL1dx - dL2dx
    dL3dy = -dL1dy - dL2dy

    if i == 1:
        bi = L1 * (2.0 * L1 - 1.0)
        dbidx = dL1dx * (4.0 * L1 - 1.0)
        dbidy = dL1dy * (4.0 * L1 - 1.0)
    elif i == 2:
        bi = L2 * (2.0 * L2 - 1.0)
        dbidx = dL2dx * (4.0 * L2 - 1.0)
        dbidy = dL2dy * (4.0 * L2 - 1.0)
    elif i == 3:
        bi = L3 * (2.0 * L3 - 1.0)
        dbidx = dL3dx * (4.0 * L3 - 1.0)
        dbidy = dL3dy * (4.0 * L3 - 1.0)
    elif i == 4:
        bi = 4.0 * L1 * L2
        dbidx = 4.0 * (dL1dx * L2 + L1 * dL2dx)
        dbidy = 4.0 * (dL1dy * L2 + L1 * dL2dy)
    elif i == 5:
        bi = 4.0 * L2 * L3
        dbidx = 4.0 * (dL2dx * L3 + L2 * dL3dx)
        dbidy = 4.0 * (dL2dy * L3 + L2 * dL3dy)
    else:
        bi = 4.0 * L3 * L1
        dbidx = 4.0 * (dL3dx * L1 + L3 * dL1dx)
        dbidy = 4.0 * (dL3dy * L1 + L3 * dL1dy)

    return bi, dbidx, dbidy


def get_quad_rule_triangle(nq=3):
    if nq == 1:
        w = np.array([0.5])
        xy = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    elif nq == 3:
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        xy = np.array([
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5]
        ])
    else:
        raise ValueError("get_quad_rule_triangle: 仅支持 nq=1 或 3")
    return w, xy


def assemble_fem_matrices(node_xy, element_node, element_order=6,
                          k_coef=None, nq=3):
    node_num = node_xy.shape[0]
    element_num = element_node.shape[0]

    if k_coef is None:
        def k_coef_default(x, y):
            return 0.0
        k_coef = k_coef_default

    quad_w, quad_xy = get_quad_rule_triangle(nq)

    row_A, col_A, data_A = [], [], []
    row_M, col_M, data_M = [], [], []

    for e in range(element_num):
        nodes_e = element_node[e, :element_order]
        t3 = node_xy[nodes_e[:3], :]
        area = triangle_area(t3[0], t3[1], t3[2])
        if area < 1e-14:
            continue


        xy_phys = reference_to_physical_t3(t3, quad_xy)

        for q in range(nq):
            xq, yq = xy_phys[q, 0], xy_phys[q, 1]
            w = area * quad_w[q]
            kq = k_coef(xq, yq)

            for test in range(element_order):
                i = nodes_e[test]
                bi, dbidx, dbidy = basis_11_t6(node_xy[nodes_e, :], test + 1, xy_phys[q, :])

                for basis in range(element_order):
                    j = nodes_e[basis]
                    bj, dbjdx, dbjdy = basis_11_t6(node_xy[nodes_e, :], basis + 1, xy_phys[q, :])

                    aij = dbidx * dbjdx + dbidy * dbjdy + kq * bi * bj
                    mij = bi * bj

                    row_A.append(i)
                    col_A.append(j)
                    data_A.append(w * aij)

                    row_M.append(i)
                    col_M.append(j)
                    data_M.append(w * mij)

    A = sp.coo_matrix((data_A, (row_A, col_A)), shape=(node_num, node_num)).tocsr()
    M = sp.coo_matrix((data_M, (row_M, col_M)), shape=(node_num, node_num)).tocsr()
    return A, M


def apply_dirichlet_bc(A, rhs, node_xy, boundary_func):
    node_num = node_xy.shape[0]

    xl, xr = node_xy[:, 0].min(), node_xy[:, 0].max()
    yb, yt = node_xy[:, 1].min(), node_xy[:, 1].max()
    tol = 1e-10 * max(xr - xl, yt - yb)

    bc_indices = []
    bc_values = []
    for i in range(node_num):
        x, y = node_xy[i, 0], node_xy[i, 1]
        if (abs(x - xl) < tol or abs(x - xr) < tol or
                abs(y - yb) < tol or abs(y - yt) < tol):
            bc_indices.append(i)
            bc_values.append(boundary_func(x, y))

    bc_indices = np.array(bc_indices, dtype=int)
    bc_values = np.array(bc_values, dtype=float)


    A = A.tolil()
    for idx, val in zip(bc_indices, bc_values):
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        rhs[idx] = val
    A = A.tocsr()

    return A, rhs, bc_indices, bc_values


def solve_poisson_fem(node_xy, element_node, rhs_func, boundary_func,
                      element_order=6, nq=3):
    node_num = node_xy.shape[0]
    A, M = assemble_fem_matrices(node_xy, element_node, element_order, nq=nq)


    element_num = element_node.shape[0]
    quad_w, quad_xy = get_quad_rule_triangle(nq)
    rhs = np.zeros(node_num, dtype=float)

    for e in range(element_num):
        nodes_e = element_node[e, :element_order]
        t3 = node_xy[nodes_e[:3], :]
        area = triangle_area(t3[0], t3[1], t3[2])
        if area < 1e-14:
            continue
        xy_phys = reference_to_physical_t3(t3, quad_xy)
        for q in range(nq):
            w = area * quad_w[q]
            for test in range(element_order):
                i = nodes_e[test]
                bi, _, _ = basis_11_t6(node_xy[nodes_e, :], test + 1, xy_phys[q, :])
                rhs[i] += w * rhs_func(xy_phys[q, 0], xy_phys[q, 1]) * bi

    A, rhs, bc_indices, bc_values = apply_dirichlet_bc(A, rhs, node_xy, boundary_func)
    u = spsolve(A, rhs)
    return u


def solve_heat_fem(node_xy, element_node, u_init, dt, n_steps,
                   rhs_func, boundary_func, element_order=6, nq=3):
    node_num = node_xy.shape[0]
    A, M = assemble_fem_matrices(node_xy, element_node, element_order, nq=nq)

    u = np.asarray(u_init, dtype=float).copy()
    u_history = [u.copy()]

    for step in range(n_steps):
        t = (step + 1) * dt


        quad_w, quad_xy = get_quad_rule_triangle(nq)
        element_num = element_node.shape[0]
        f_rhs = np.zeros(node_num, dtype=float)
        for e in range(element_num):
            nodes_e = element_node[e, :element_order]
            t3 = node_xy[nodes_e[:3], :]
            area = triangle_area(t3[0], t3[1], t3[2])
            if area < 1e-14:
                continue
            xy_phys = reference_to_physical_t3(t3, quad_xy)
            for q in range(nq):
                w = area * quad_w[q]
                for test in range(element_order):
                    i = nodes_e[test]
                    bi, _, _ = basis_11_t6(node_xy[nodes_e, :], test + 1, xy_phys[q, :])
                    f_rhs[i] += w * rhs_func(xy_phys[q, 0], xy_phys[q, 1], t) * bi


        lhs = M + dt * A
        rhs = M @ u + dt * f_rhs


        lhs, rhs, bc_indices, bc_values = apply_dirichlet_bc(lhs, rhs, node_xy,
                                                              lambda x, y: boundary_func(x, y, t))
        u = spsolve(lhs, rhs)
        u_history.append(u.copy())

    return u, u_history
