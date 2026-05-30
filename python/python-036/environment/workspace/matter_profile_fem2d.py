
import numpy as np
from constants import EARTH_RADIUS_KM, get_prem_density


def generate_triangular_mesh_2d(radius, n_r=20, n_theta=32):
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_r < 2 or n_theta < 3:
        raise ValueError("n_r >= 2 and n_theta >= 3 required")

    nodes = []
    node_map = {}


    for i in range(n_r):
        r = radius * i / (n_r - 1)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            node_map[(i, j)] = len(nodes)
            nodes.append([x, y])




    nodes = []
    node_map = {}


    node_map[(0, 0)] = 0
    nodes.append([0.0, 0.0])

    for i in range(1, n_r):
        r = radius * i / (n_r - 1)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            node_map[(i, j)] = len(nodes)
            nodes.append([x, y])

    n_nodes = len(nodes)
    nodes = np.array(nodes, dtype=np.float64)


    elements = []
    for i in range(n_r - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta

            if i == 0:

                n0 = node_map[(0, 0)]
                n1 = node_map[(1, j)]
                n2 = node_map[(1, j_next)]
                elements.append([n0, n1, n2])
            else:

                n0 = node_map[(i, j)]
                n1 = node_map[(i, j_next)]
                n2 = node_map[(i + 1, j)]
                n3 = node_map[(i + 1, j_next)]

                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])

    elements = np.array(elements, dtype=np.int64)
    return nodes, elements


def triangle_area_2d(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def basis_p1_2d(p, p1, p2, p3):
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(area2) < 1e-14:
        raise ValueError("Degenerate triangle")

    a1 = x2 * y3 - x3 * y2
    b1 = y2 - y3
    c1 = x3 - x2

    a2 = x3 * y1 - x1 * y3
    b2 = y3 - y1
    c2 = x1 - x3

    a3 = x1 * y2 - x2 * y1
    b3 = y1 - y2
    c3 = x2 - x1

    phi = np.array([
        (a1 + b1 * x + c1 * y) / area2,
        (a2 + b2 * x + c2 * y) / area2,
        (a3 + b3 * x + c3 * y) / area2
    ], dtype=np.float64)

    dphidx = np.array([b1, b2, b3], dtype=np.float64) / area2
    dphidy = np.array([c1, c2, c3], dtype=np.float64) / area2

    return phi, dphidx, dphidy


def quadrature_triangle_3point():
    weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    local_coords = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.6 / 6.0, 1.0 / 6.0, 2.0 / 3.0]
    ])
    return weights, local_coords


def map_to_physical_triangle(p1, p2, p3, bary_coords):
    lam1, lam2, lam3 = bary_coords
    x = lam1 * p1[0] + lam2 * p2[0] + lam3 * p3[0]
    y = lam1 * p1[1] + lam2 * p2[1] + lam3 * p3[1]
    return np.array([x, y])


def assemble_fem_2d(nodes, elements, k_diffusion=1.0,
                    source_fun=None, time=0.0):
    n_nodes = len(nodes)
    n_elements = len(elements)
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    b = np.zeros(n_nodes, dtype=np.float64)

    quad_w, quad_loc = quadrature_triangle_3point()

    for e in range(n_elements):
        idx = elements[e]
        p1 = nodes[idx[0]]
        p2 = nodes[idx[1]]
        p3 = nodes[idx[2]]

        area = triangle_area_2d(p1, p2, p3)
        if area < 1e-14:
            continue

        for q in range(len(quad_w)):
            p_phys = map_to_physical_triangle(p1, p2, p3, quad_loc[q])
            w = quad_w[q] * area

            if callable(k_diffusion):
                k_val = k_diffusion(p_phys[0], p_phys[1], time)
            else:
                k_val = float(k_diffusion)

            phi, dphidx, dphidy = basis_p1_2d(p_phys, p1, p2, p3)


            for i_local in range(3):
                i_global = idx[i_local]
                for j_local in range(3):
                    j_global = idx[j_local]
                    A[i_global, j_global] += w * k_val * (
                        dphidx[i_local] * dphidx[j_local] +
                        dphidy[i_local] * dphidy[j_local]
                    )


            if source_fun is not None:
                f_val = source_fun(p_phys[0], p_phys[1], time)
                for i_local in range(3):
                    i_global = idx[i_local]
                    b[i_global] += w * f_val * phi[i_local]

    return A, b


def identify_boundary_nodes_2d(nodes, radius, tol=1.0):
    dist = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    is_boundary = np.abs(dist - radius) < tol
    return is_boundary


def apply_dirichlet_bc_2d(A, b, nodes, is_boundary, boundary_value_fun):
    A = A.copy()
    b = b.copy()
    n_nodes = len(nodes)

    for i in range(n_nodes):
        if is_boundary[i]:
            x, y = nodes[i]
            bc_val = boundary_value_fun(x, y)
            A[i, :] = 0.0
            A[i, i] = 1.0
            b[i] = bc_val

    return A, b


def solve_steady_state_density_2d(radius_km=EARTH_RADIUS_KM, n_r=15, n_theta=24):
    nodes, elements = generate_triangular_mesh_2d(radius_km, n_r, n_theta)

    def source_fun(x, y, t):
        r = np.sqrt(x ** 2 + y ** 2)
        r_ratio = r / radius_km
        r_ratio = max(0.0, min(1.0, r_ratio))
        return 0.05 * get_prem_density(r_ratio)

    A, b = assemble_fem_2d(nodes, elements, k_diffusion=1.0,
                           source_fun=source_fun)

    is_boundary = identify_boundary_nodes_2d(nodes, radius_km, tol=radius_km / n_r)

    def bc_fun(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        r_ratio = r / radius_km
        r_ratio = max(0.0, min(1.0, r_ratio))
        return get_prem_density(r_ratio)

    A, b = apply_dirichlet_bc_2d(A, b, nodes, is_boundary, bc_fun)

    try:
        rho = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        rho = np.linalg.lstsq(A, b, rcond=None)[0]

    return rho, nodes, elements


def compute_bandwidth(element_order, elements):
    nhba = 0
    n_elements = len(elements)
    for e in range(n_elements):
        for local_i in range(element_order):
            global_i = elements[e, local_i]
            for local_j in range(element_order):
                global_j = elements[e, local_j]
                nhba = max(nhba, abs(global_j - global_i))
    return nhba
