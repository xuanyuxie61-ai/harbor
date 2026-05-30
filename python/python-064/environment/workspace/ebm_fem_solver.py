
import numpy as np


def r8sr_mv(n, nz, row_ptr, col_idx, diag, off, x):
    x = np.asarray(x, dtype=float).flatten()
    diag = np.asarray(diag, dtype=float).flatten()
    b = diag * x
    for i in range(n):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            b[i] += off[k] * x[j]
    return b


def r8sr_zeros(n):
    return {
        'n': n,
        'nz': 0,
        'row_ptr': np.zeros(n + 1, dtype=int),
        'col_idx': np.array([], dtype=int),
        'diag': np.zeros(n),
        'off': np.array([], dtype=float)
    }


def dense_to_r8sr(A, tol=1e-14):
    n = A.shape[0]
    diag = np.diag(A).copy()
    row_ptr = np.zeros(n + 1, dtype=int)
    col_idx_list = []
    off_list = []

    for i in range(n):
        count = 0
        for j in range(n):
            if i != j and abs(A[i, j]) > tol:
                col_idx_list.append(j)
                off_list.append(A[i, j])
                count += 1
        row_ptr[i + 1] = row_ptr[i] + count

    return {
        'n': n,
        'nz': len(off_list),
        'row_ptr': row_ptr,
        'col_idx': np.array(col_idx_list, dtype=int),
        'diag': diag,
        'off': np.array(off_list, dtype=float)
    }


def r8sr_to_dense(sr):
    n = sr['n']
    A = np.zeros((n, n))
    np.fill_diagonal(A, sr['diag'])
    for i in range(n):
        for k in range(sr['row_ptr'][i], sr['row_ptr'][i + 1]):
            j = sr['col_idx'][k]
            A[i, j] = sr['off'][k]
    return A


def assemble_t6_element(nodes_xy, node_vals=None):

    x1, y1 = nodes_xy[0]
    x2, y2 = nodes_xy[1]
    x3, y3 = nodes_xy[2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return area


def basis_t3(x, y, nodes_xy):
    x1, y1 = nodes_xy[0]
    x2, y2 = nodes_xy[1]
    x3, y3 = nodes_xy[2]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(det) < 1e-15:
        det = 1e-15

    phi1 = ((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)) / det
    phi2 = ((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y)) / det
    phi3 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / det

    dphi1_dx = (y2 - y3) / det
    dphi1_dy = (x3 - x2) / det
    dphi2_dx = (y3 - y1) / det
    dphi2_dy = (x1 - x3) / det
    dphi3_dx = (y1 - y2) / det
    dphi3_dy = (x2 - x1) / det

    return np.array([phi1, phi2, phi3]), np.array([dphi1_dx, dphi2_dx, dphi3_dx]), \
           np.array([dphi1_dy, dphi2_dy, dphi3_dy])


def gauss_triangle_points(order=3):
    if order == 1:
        return np.array([[1.0/3.0, 1.0/3.0]]), np.array([0.5])
    elif order == 3:
        pts = np.array([
            [1.0/6.0, 1.0/6.0],
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0]
        ])
        w = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        return pts, w
    elif order == 7:
        a = 0.059715871789770
        b = 0.797426985353087
        pts = np.array([
            [1.0/3.0, 1.0/3.0],
            [a, a], [1.0 - 2.0*a, a], [a, 1.0 - 2.0*a],
            [b, b], [1.0 - 2.0*b, b], [b, 1.0 - 2.0*b]
        ])
        w = np.array([
            0.1125,
            0.066197076394253, 0.066197076394253, 0.066197076394253,
            0.062969590272414, 0.062969590272414, 0.062969590272414
        ])
        return pts, w
    else:
        return gauss_triangle_points(3)


def fem_heat_assemble(nodes, elements, k_coef_func, rhs_func, quad_order=3):
    n_nodes = len(nodes)
    n_elem = len(elements)
    A = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    quad_pts, quad_w = gauss_triangle_points(quad_order)

    for e in range(n_elem):
        elem_nodes = elements[e]
        x = nodes[elem_nodes, 0]
        y = nodes[elem_nodes, 1]


        xy = np.column_stack([x, y])


        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        for qp, w in zip(quad_pts, quad_w):

            xi, eta = qp
            x_phys = x[0] + (x[1] - x[0]) * xi + (x[2] - x[0]) * eta
            y_phys = y[0] + (y[1] - y[0]) * xi + (y[2] - y[0]) * eta

            phi, dphi_dx, dphi_dy = basis_t3(x_phys, y_phys, xy)

            k_val = k_coef_func(x_phys, y_phys)
            f_val = rhs_func(x_phys, y_phys)
            weight = w * area

            for i_local in range(3):
                i_global = elem_nodes[i_local]
                F[i_global] += weight * f_val * phi[i_local]
                for j_local in range(3):
                    j_global = elem_nodes[j_local]
                    A[i_global, j_global] += weight * (
                        k_val * (dphi_dx[i_local] * dphi_dx[j_local] +
                                 dphi_dy[i_local] * dphi_dy[j_local])
                    )

    return A, F


def apply_dirichlet_bc(A, F, bc_nodes, bc_values):
    A = A.copy()
    F = F.copy()
    n = A.shape[0]

    for idx, node in enumerate(bc_nodes):
        A[node, :] = 0.0
        A[:, node] = 0.0
        A[node, node] = 1.0
        F[node] = bc_values[idx] if hasattr(bc_values, '__len__') and len(bc_values) > 1 else bc_values

    return A, F


def solve_ebm_fem(nodes, elements, insolation_field, T_initial, dt, n_steps,
                  D_diffusivity=0.3, heat_capacity=1.0e6, albedo_field=None,
                  boundary_nodes=None, boundary_temp=None):
    n_nodes = len(nodes)
    T = np.array(T_initial, dtype=float)
    T_history = np.zeros((n_steps + 1, n_nodes))
    T_history[0] = T

    if albedo_field is None:
        albedo_field = np.zeros(n_nodes)


    A_budyko = 203.3
    B_budyko = 2.09


    def k_func(x, y):
        return D_diffusivity

    K, _ = fem_heat_assemble(nodes, elements, k_func, lambda x, y: 0.0)


    M = np.zeros((n_nodes, n_nodes))
    for e in range(len(elements)):
        elem = elements[e]
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        for i in range(3):
            M[elem[i], elem[i]] += area / 3.0

    for step in range(n_steps):





        raise NotImplementedError("Hole_2: EBM albedo and forcing computation is not implemented.")


        if boundary_nodes is not None and boundary_temp is not None:
            sys_mat, rhs = apply_dirichlet_bc(sys_mat, rhs, boundary_nodes,
                                               np.full(len(boundary_nodes), boundary_temp))

        try:
            T_new = np.linalg.solve(sys_mat, rhs)
        except np.linalg.LinAlgError:

            T_new = np.linalg.lstsq(sys_mat, rhs, rcond=None)[0]

        T_new = np.clip(T_new, 200.0, 350.0)
        T = T_new
        T_history[step + 1] = T

    return T_history


def create_spherical_mesh(n_lat=18, n_lon=36):
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(0, 360, n_lon)

    nodes = []
    node_map = {}
    idx = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            x = np.sin(np.deg2rad(lat))
            y = np.deg2rad(lon) / np.pi
            nodes.append([x, y])
            node_map[(i, j)] = idx
            idx += 1

    nodes = np.array(nodes)
    elements = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            n1 = node_map[(i, j)]
            n2 = node_map[(i, j + 1)]
            n3 = node_map[(i + 1, j)]
            n4 = node_map[(i + 1, j + 1)]
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])

    return nodes, np.array(elements)
