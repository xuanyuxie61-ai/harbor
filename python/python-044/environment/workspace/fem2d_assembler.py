
import numpy as np
from mesh_generator import triangle_area


def shape_linear_triangle(xi, eta):
    N = np.array([1.0 - xi - eta, xi, eta])
    dN_dxi = np.array([-1.0, 1.0, 0.0])
    dN_deta = np.array([-1.0, 0.0, 1.0])
    return N, dN_dxi, dN_deta


def shape_quadratic_triangle(xi, eta):
    N = np.zeros(6)
    N[0] = (1.0 - xi - eta) * (2.0 * (1.0 - xi - eta) - 1.0)
    N[1] = xi * (2.0 * xi - 1.0)
    N[2] = eta * (2.0 * eta - 1.0)
    N[3] = 4.0 * xi * (1.0 - xi - eta)
    N[4] = 4.0 * xi * eta
    N[5] = 4.0 * eta * (1.0 - xi - eta)

    dN_dxi = np.zeros(6)
    dN_dxi[0] = -4.0 * (1.0 - xi - eta) + 1.0
    dN_dxi[1] = 4.0 * xi - 1.0
    dN_dxi[2] = 0.0
    dN_dxi[3] = 4.0 * (1.0 - xi - eta) - 4.0 * xi
    dN_dxi[4] = 4.0 * eta
    dN_dxi[5] = -4.0 * eta

    dN_deta = np.zeros(6)
    dN_deta[0] = -4.0 * (1.0 - xi - eta) + 1.0
    dN_deta[1] = 0.0
    dN_deta[2] = 4.0 * eta - 1.0
    dN_deta[3] = -4.0 * xi
    dN_deta[4] = 4.0 * xi
    dN_deta[5] = 4.0 * (1.0 - xi - eta) - 4.0 * eta

    return N, dN_dxi, dN_deta


def compute_B_matrix(dN_dx, dN_dy):
    n = len(dN_dx)
    B = np.zeros((3, 2 * n))
    for i in range(n):
        B[0, 2 * i] = dN_dx[i]
        B[1, 2 * i + 1] = dN_dy[i]
        B[2, 2 * i] = dN_dy[i]
        B[2, 2 * i + 1] = dN_dx[i]
    return B


def fem2d_biot_assemble(nodes, elements_u, elements_p, mat, quad_order=3):
    from quadrature_rules import triangle_rule, map_triangle_quad

    n_nodes = nodes.shape[0]
    n_nodes_p = int(elements_p.max()) + 1
    n_elements = elements_u.shape[0]

    ndof_u = 2 * n_nodes
    ndof_p = n_nodes_p

    K_uu = np.zeros((ndof_u, ndof_u))
    C = np.zeros((ndof_u, ndof_p))
    M_p = np.zeros((ndof_p, ndof_p))
    K_p = np.zeros((ndof_p, ndof_p))
    M_uu = np.zeros((ndof_u, ndof_u))

    w_ref, xi_ref, eta_ref = triangle_rule(quad_order)
    nq = len(w_ref)

    D = mat.elastic_matrix()
    alpha = mat.alpha
    M_biot = mat.M
    kappa_eta = mat.kappa / mat.eta
    rho_bulk = mat.rho_bulk


    n_u_per_elem = elements_u.shape[1]
    use_quadratic = (n_u_per_elem == 6)

    for e in range(n_elements):

        vert = nodes[elements_p[e, :3]]


        J = np.array([
            [vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
            [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]],
        ])
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-14:
            continue
        Jinv = np.linalg.inv(J)


        u_nodes = elements_u[e, :]
        p_nodes = elements_p[e, :]

        for q in range(nq):
            xi = xi_ref[q]
            eta = eta_ref[q]
            wq = w_ref[q] * abs(detJ)


            if use_quadratic:
                N_u, dN_u_xi, dN_u_eta = shape_quadratic_triangle(xi, eta)
            else:
                N_u, dN_u_xi, dN_u_eta = shape_linear_triangle(xi, eta)
            nu_loc = len(N_u)

            grad_N_u = np.vstack([dN_u_xi, dN_u_eta])
            grad_N_u_phys = Jinv @ grad_N_u
            dN_u_dx = grad_N_u_phys[0, :]
            dN_u_dy = grad_N_u_phys[1, :]


            N_p, dN_p_xi, dN_p_eta = shape_linear_triangle(xi, eta)

            grad_N_p = np.vstack([dN_p_xi, dN_p_eta])
            grad_N_p_phys = Jinv @ grad_N_p
            dN_p_dx = grad_N_p_phys[0, :]
            dN_p_dy = grad_N_p_phys[1, :]


            B_mat = compute_B_matrix(dN_u_dx, dN_u_dy)







            Ce = None
            Mpe = None
            grad_Np_mat = np.vstack([dN_p_dx, dN_p_dy])
            Kpe = None

            N_u_expanded = np.zeros(2 * nu_loc)
            for i in range(nu_loc):
                N_u_expanded[2 * i] = N_u[i]
                N_u_expanded[2 * i + 1] = N_u[i]
            Mue = rho_bulk * np.outer(N_u_expanded, N_u_expanded) * wq


            for i in range(nu_loc):
                ii_x = 2 * u_nodes[i]
                ii_y = 2 * u_nodes[i] + 1
                for j in range(nu_loc):
                    jj_x = 2 * u_nodes[j]
                    jj_y = 2 * u_nodes[j] + 1
                    K_uu[ii_x, jj_x] += Ke_uu[2 * i, 2 * j]
                    K_uu[ii_x, jj_y] += Ke_uu[2 * i, 2 * j + 1]
                    K_uu[ii_y, jj_x] += Ke_uu[2 * i + 1, 2 * j]
                    K_uu[ii_y, jj_y] += Ke_uu[2 * i + 1, 2 * j + 1]

                    M_uu[ii_x, jj_x] += Mue[2 * i, 2 * j]
                    M_uu[ii_x, jj_y] += Mue[2 * i, 2 * j + 1]
                    M_uu[ii_y, jj_x] += Mue[2 * i + 1, 2 * j]
                    M_uu[ii_y, jj_y] += Mue[2 * i + 1, 2 * j + 1]

                for j in range(3):
                    jj = p_nodes[j]
                    C[ii_x, jj] += Ce[2 * i, j]
                    C[ii_y, jj] += Ce[2 * i + 1, j]

            for i in range(3):
                ii = p_nodes[i]
                for j in range(3):
                    jj = p_nodes[j]
                    M_p[ii, jj] += Mpe[i, j]
                    K_p[ii, jj] += Kpe[i, j]

    return K_uu, C, M_p, K_p, M_uu


def apply_dirichlet_bc(K, F, bc_nodes, bc_value, ndof_per_node=1):
    K = K.copy()
    F = F.copy()
    n = K.shape[0]

    if np.isscalar(bc_value):
        bc_value = np.full(len(bc_nodes) * ndof_per_node, bc_value)
    else:
        bc_value = np.asarray(bc_value).flatten()


    dof_list = []
    for node in bc_nodes:
        for d in range(ndof_per_node):
            dof_list.append(node * ndof_per_node + d)
    dof_list = np.array(dof_list, dtype=int)

    if len(dof_list) != len(bc_value):
        raise ValueError("Length of bc_nodes and bc_value mismatch.")


    diag_max = np.max(np.diag(K))
    penalty = max(diag_max * 1e6, 1.0)

    for idx, dof in enumerate(dof_list):
        if dof < 0 or dof >= n:
            continue
        K[dof, dof] += penalty
        F[dof] += penalty * bc_value[idx]

    return K, F


def extract_displacement_pressure(sol, n_nodes):
    u = sol[:2 * n_nodes].reshape((n_nodes, 2))
    p = sol[2 * n_nodes:]
    return u, p
