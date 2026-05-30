
import numpy as np
from fem_solver import assemble_fem_system, shape_functions_p1
from quadrature_rules import TriangleQuadrature


def identify_boundary_edges(nodes, triangles, domain_bounds, tol=1e-10):
    n_nodes = len(nodes)
    edge_count = {}

    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    boundary_nodes = set()
    for edge in boundary_edges:
        boundary_nodes.add(edge[0])
        boundary_nodes.add(edge[1])
    boundary_nodes = np.array(sorted(list(boundary_nodes)), dtype=int)


    dirichlet_nodes = boundary_nodes.copy()


    neumann_edges = []

    return boundary_nodes, dirichlet_nodes, neumann_edges


def compute_advection_matrix(nodes, triangles, v_func, quad_degree=3):
    n_nodes = len(nodes)
    n_tri = len(triangles)
    quad_rule = TriangleQuadrature(quad_degree)
    C = np.zeros((n_nodes, n_nodes))

    for t in range(n_tri):
        tri = triangles[t]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        J = np.array([
            [p2[0] - p1[0], p3[0] - p1[0]],
            [p2[1] - p1[1], p3[1] - p1[1]]
        ])
        det_J = abs(np.linalg.det(J))

        if det_J < 1e-14:
            continue

        J_inv_T = np.linalg.inv(J).T

        for q in range(quad_rule.n_points):
            xi = quad_rule.points[q, 0]
            eta = quad_rule.points[q, 1]
            w = quad_rule.weights[q]

            phi, grad_phi_ref = shape_functions_p1(xi, eta)
            grad_phi = grad_phi_ref @ J_inv_T.T

            x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
            y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

            v_x, v_y = v_func(x, y)

            for i in range(3):
                for j in range(3):
                    C[tri[i], tri[j]] += w * (
                        (v_x * grad_phi[j, 0] + v_y * grad_phi[j, 1]) * phi[i]
                    ) * det_J

    return C


def compute_reaction_term(nodes, triangles, u_current, R_func, quad_degree=3):
    n_nodes = len(nodes)
    n_tri = len(triangles)
    quad_rule = TriangleQuadrature(quad_degree)
    b_R = np.zeros(n_nodes)

    for t in range(n_tri):
        tri = triangles[t]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        det_J = abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        if det_J < 1e-14:
            continue

        for q in range(quad_rule.n_points):
            xi = quad_rule.points[q, 0]
            eta = quad_rule.points[q, 1]
            w = quad_rule.weights[q]

            phi, _ = shape_functions_p1(xi, eta)

            x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
            y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta


            u_val = np.dot(phi, u_current[tri])
            R_val = R_func(u_val, x, y)

            for i in range(3):
                b_R[tri[i]] += w * R_val * phi[i] * det_J

    return b_R


def compute_mass_matrix_lumped(nodes, triangles):
    n_nodes = len(nodes)
    M_lumped = np.zeros(n_nodes)

    for tri in triangles:
        p1 = nodes[tri[0]]
        p2 = nodes[tri[1]]
        p3 = nodes[tri[2]]

        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        for i in range(3):
            M_lumped[tri[i]] += area / 3.0


    M_lumped = np.maximum(M_lumped, 1e-14)
    return M_lumped


def compute_cfl_condition(nodes, triangles, v_func, D_func):
    h_min = np.inf
    pe_max = 0.0

    for tri in triangles:
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]


        e1 = np.linalg.norm(p2 - p1)
        e2 = np.linalg.norm(p3 - p2)
        e3 = np.linalg.norm(p1 - p3)
        h_T = max(e1, e2, e3)
        h_min = min(h_min, h_T)


        centroid = (p1 + p2 + p3) / 3.0
        v_x, v_y = v_func(centroid[0], centroid[1])
        v_mag = np.sqrt(v_x ** 2 + v_y ** 2)
        D_val = D_func(centroid[0], centroid[1])

        if D_val > 1e-14:
            pe_T = v_mag * h_T / D_val
            pe_max = max(pe_max, pe_T)


    dt_conv = h_min / (max(v_mag, 1e-10))
    dt_diff = h_min ** 2 / (4.0 * max(D_val, 1e-10))
    dt_max = min(dt_conv, dt_diff)

    return dt_max, h_min, pe_max


def advection_diffusion_reaction_step(
    nodes, triangles,
    u_current, dt,
    D_func, c_func, v_func, R_func, f_func,
    dirichlet_nodes, dirichlet_values,
    M_lumped, scheme='implicit'
):
    n_nodes = len(nodes)


    A_diff, b_source = assemble_fem_system(
        nodes, triangles,
        D_func, c_func, f_func,
        dirichlet_nodes=None,
        quad_degree=3
    )


    C_adv = compute_advection_matrix(nodes, triangles, v_func, quad_degree=3)


    b_R = compute_reaction_term(nodes, triangles, u_current, R_func, quad_degree=3)


    b_total = b_source + b_R





    if scheme == 'implicit':
        lhs = np.eye(n_nodes)
        rhs = np.zeros(n_nodes)
    elif scheme == 'crank_nicolson':
        lhs = np.eye(n_nodes)
        rhs = np.zeros(n_nodes)
    else:
        raise ValueError(f"advection_diffusion_reaction_step: 未知 scheme={scheme}")


    if dirichlet_nodes is not None:
        for idx, node in enumerate(dirichlet_nodes):
            lhs[node, :] = 0.0
            lhs[node, node] = 1.0
            rhs[node] = dirichlet_values[idx]

            for i in range(n_nodes):
                if i != node and lhs[i, node] != 0:
                    rhs[i] -= lhs[i, node] * dirichlet_values[idx]
                    lhs[i, node] = 0.0

    try:
        u_new = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        u_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return u_new
