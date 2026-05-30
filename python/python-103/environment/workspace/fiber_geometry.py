
import numpy as np



_WANDZURA05_X = np.array([
    0.33333333333333,
    0.05971587178977, 0.79742698535309, 0.14296124917414,
    0.47014206410512, 0.47014206410512, 0.05971587178977
])
_WANDZURA05_Y = np.array([
    0.33333333333333,
    0.79742698535309, 0.14296124917414, 0.05971587178977,
    0.47014206410512, 0.05971587178977, 0.47014206410512
])
_WANDZURA05_W = np.array([
    0.22500000000000,
    0.13239415278851, 0.13239415278851, 0.13239415278851,
    0.12593918054483, 0.12593918054483, 0.12593918054483
])


def triangle_area(v1, v2, v3):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    return 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0]))


def triangle_integrand_gauss(f, v1, v2, v3):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    area = triangle_area(v1, v2, v3)
    result = 0.0
    for i in range(_WANDZURA05_X.size):

        xi = _WANDZURA05_X[i]
        eta = _WANDZURA05_Y[i]
        zeta = 1.0 - xi - eta
        x = xi * v1[0] + eta * v2[0] + zeta * v3[0]
        y = xi * v1[1] + eta * v2[1] + zeta * v3[1]
        result += _WANDZURA05_W[i] * f(x, y)

    return area * result


def create_fiber_triangulation(r_core, r_cladding, n_theta=16, n_radial_core=4, n_radial_clad=6):
    if r_core <= 0 or r_cladding <= r_core or n_theta < 3:
        raise ValueError("create_fiber_triangulation: invalid geometry parameters")

    nodes = []

    nodes.append([0.0, 0.0])


    for i in range(1, n_radial_core + 1):
        r = i * r_core / n_radial_core
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes.append([r * np.cos(theta), r * np.sin(theta)])


    for i in range(1, n_radial_clad + 1):
        r = r_core + i * (r_cladding - r_core) / n_radial_clad
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes.append([r * np.cos(theta), r * np.sin(theta)])

    nodes = np.array(nodes)

    triangles = []

    for j in range(n_theta):
        n0 = 0
        n1 = 1 + j
        n2 = 1 + (j + 1) % n_theta
        triangles.append([n0, n1, n2])


    for i in range(n_radial_core - 1):
        offset = 1 + i * n_theta
        offset_next = 1 + (i + 1) * n_theta
        for j in range(n_theta):
            n0 = offset + j
            n1 = offset + (j + 1) % n_theta
            n2 = offset_next + j
            n3 = offset_next + (j + 1) % n_theta
            triangles.append([n0, n1, n2])
            triangles.append([n1, n3, n2])


    core_offset = 1 + (n_radial_core - 1) * n_theta
    clad_offset = 1 + n_radial_core * n_theta
    for j in range(n_theta):
        n0 = core_offset + j
        n1 = core_offset + (j + 1) % n_theta
        n2 = clad_offset + j
        n3 = clad_offset + (j + 1) % n_theta
        triangles.append([n0, n1, n2])
        triangles.append([n1, n3, n2])


    for i in range(n_radial_clad - 1):
        offset = clad_offset + i * n_theta
        offset_next = clad_offset + (i + 1) * n_theta
        for j in range(n_theta):
            n0 = offset + j
            n1 = offset + (j + 1) % n_theta
            n2 = offset_next + j
            n3 = offset_next + (j + 1) % n_theta
            triangles.append([n0, n1, n2])
            triangles.append([n1, n3, n2])

    triangles = np.array(triangles, dtype=int)


    boundary_flags = np.zeros(nodes.shape[0], dtype=int)

    outer_offset = clad_offset + (n_radial_clad - 1) * n_theta
    for j in range(n_theta):
        boundary_flags[outer_offset + j] = 1


    for j in range(n_theta):
        boundary_flags[core_offset + j] = 1
        boundary_flags[clad_offset + j] = 1

    return nodes, triangles, boundary_flags


def identify_boundary_nodes(triangles, node_num):
    triangles = np.asarray(triangles, dtype=int)
    edge_count = {}

    for tri in triangles:

        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_nodes = np.zeros(node_num, dtype=int)
    for e, count in edge_count.items():
        if count == 1:
            boundary_nodes[e[0]] = 1
            boundary_nodes[e[1]] = 1

    return boundary_nodes


def compute_effective_area(nodes, triangles, mode_field):
    num_integrand = lambda x, y: np.abs(mode_field(x, y)) ** 2
    den_integrand = lambda x, y: np.abs(mode_field(x, y)) ** 4

    num = 0.0
    den = 0.0
    for tri in triangles:
        v1, v2, v3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        num += triangle_integrand_gauss(num_integrand, v1, v2, v3)
        den += triangle_integrand_gauss(den_integrand, v1, v2, v3)

    if den <= 0:
        return np.inf

    return (num ** 2) / den


def compute_nonlinear_coefficient(n2, omega0, A_eff):
    c = 2.99792458e8
    if A_eff <= 0:
        raise ValueError("compute_nonlinear_coefficient: A_eff must be positive")
    return n2 * omega0 / (c * A_eff)
