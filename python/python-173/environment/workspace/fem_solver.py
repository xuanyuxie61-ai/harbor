
import numpy as np
from quadrature_rules import TriangleQuadrature


def shape_functions_p1(xi, eta):
    phi = np.array([1.0 - xi - eta, xi, eta])
    grad_phi_ref = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    return phi, grad_phi_ref


def compute_element_stiffness(nodes, triangle, D_func, c_func, quad_rule):
    p1 = nodes[triangle[0]]
    p2 = nodes[triangle[1]]
    p3 = nodes[triangle[2]]


    J = np.array([
        [p2[0] - p1[0], p3[0] - p1[0]],
        [p2[1] - p1[1], p3[1] - p1[1]]
    ])
    det_J = abs(np.linalg.det(J))

    if det_J < 1e-14:
        raise ValueError("compute_element_stiffness: 退化三角形")


    J_inv_T = np.linalg.inv(J).T

    A_local = np.zeros((3, 3))
    b_local = np.zeros(3)

    for q in range(quad_rule.n_points):
        xi = quad_rule.points[q, 0]
        eta = quad_rule.points[q, 1]
        w = quad_rule.weights[q]

        phi, grad_phi_ref = shape_functions_p1(xi, eta)


        x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
        y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta


        grad_phi = grad_phi_ref @ J_inv_T.T

        D_val = D_func(x, y)
        c_val = c_func(x, y)





        for i in range(3):
            for j in range(3):
                A_local[i, j] += w * (

                    0.0
                ) * det_J


            b_local[i] += 0.0

    return A_local, b_local


def assemble_fem_system(
    nodes, triangles,
    D_func, c_func, f_func,
    dirichlet_nodes=None, dirichlet_values=None,
    neumann_edges=None, neumann_func=None,
    quad_degree=3
):
    n_nodes = len(nodes)
    n_tri = len(triangles)
    quad_rule = TriangleQuadrature(quad_degree)

    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for t in range(n_tri):
        tri = triangles[t]
        A_local, _ = compute_element_stiffness(nodes, tri, D_func, c_func, quad_rule)


        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += A_local[i, j]


        p1 = nodes[tri[0]]
        p2 = nodes[tri[1]]
        p3 = nodes[tri[2]]
        det_J = abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        for q in range(quad_rule.n_points):
            xi = quad_rule.points[q, 0]
            eta = quad_rule.points[q, 1]
            w = quad_rule.weights[q]

            phi, _ = shape_functions_p1(xi, eta)
            x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
            y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

            f_val = f_func(x, y)
            for i in range(3):
                b[tri[i]] += w * f_val * phi[i] * det_J


    if neumann_edges is not None and neumann_func is not None:
        for edge in neumann_edges:
            n1, n2 = edge
            p1 = nodes[n1]
            p2 = nodes[n2]
            edge_len = np.linalg.norm(p2 - p1)

            if edge_len < 1e-14:
                continue


            mid = (p1 + p2) / 2.0
            g_val = neumann_func(mid[0], mid[1])


            b[n1] += g_val * edge_len / 2.0
            b[n2] += g_val * edge_len / 2.0


    if dirichlet_nodes is not None and dirichlet_values is not None:
        dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
        dirichlet_values = np.array(dirichlet_values)

        for idx, node in enumerate(dirichlet_nodes):
            val = dirichlet_values[idx]

            A[node, :] = 0.0
            A[node, node] = 1.0
            b[node] = val


            for i in range(n_nodes):
                if i != node and A[i, node] != 0:
                    b[i] -= A[i, node] * val
                    A[i, node] = 0.0

    return A, b


def solve_steady_fem(nodes, triangles, D_func, c_func, f_func,
                     dirichlet_nodes=None, dirichlet_values=None,
                     neumann_edges=None, neumann_func=None,
                     quad_degree=3):
    A, b = assemble_fem_system(
        nodes, triangles,
        D_func, c_func, f_func,
        dirichlet_nodes, dirichlet_values,
        neumann_edges, neumann_func,
        quad_degree
    )


    cond_est = np.linalg.cond(A)
    if cond_est > 1e14:

        reg = 1e-10 * np.mean(np.diag(A))
        A = A + reg * np.eye(len(A))

    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:

        solution = np.linalg.lstsq(A, b, rcond=None)[0]

    return solution, A, b
