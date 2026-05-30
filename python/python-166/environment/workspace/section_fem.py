
import numpy as np
from typing import Tuple, List, Callable
from mesh_utils import diaphony_compute


def shape_t6(xi: float, eta: float, node_idx: int) -> float:
    if xi < -1e-12 or eta < -1e-12 or xi + eta > 1.0 + 1e-12:
        return 0.0
    if node_idx == 0:
        return (2.0 * xi + 2.0 * eta - 1.0) * (xi + eta - 1.0)
    elif node_idx == 1:
        return xi * (2.0 * xi - 1.0)
    elif node_idx == 2:
        return eta * (2.0 * eta - 1.0)
    elif node_idx == 3:
        return -4.0 * xi * (xi + eta - 1.0)
    elif node_idx == 4:
        return 4.0 * xi * eta
    elif node_idx == 5:
        return -4.0 * eta * (xi + eta - 1.0)
    else:
        raise ValueError("node_idx must be in [0,5]")


def grad_shape_t6(xi: float, eta: float, node_idx: int) -> np.ndarray:
    if node_idx == 0:
        dxi = 4.0 * xi + 4.0 * eta - 3.0
        deta = 4.0 * xi + 4.0 * eta - 3.0
    elif node_idx == 1:
        dxi = 4.0 * xi - 1.0
        deta = 0.0
    elif node_idx == 2:
        dxi = 0.0
        deta = 4.0 * eta - 1.0
    elif node_idx == 3:
        dxi = -4.0 * (2.0 * xi + eta - 1.0)
        deta = -4.0 * xi
    elif node_idx == 4:
        dxi = 4.0 * eta
        deta = 4.0 * xi
    elif node_idx == 5:
        dxi = -4.0 * eta
        deta = -4.0 * (xi + 2.0 * eta - 1.0)
    else:
        raise ValueError("node_idx must be in [0,5]")
    return np.array([dxi, deta])


def gauss_legendre_triangle(order: int) -> Tuple[np.ndarray, np.ndarray]:
    if order <= 1:

        xi = np.array([1.0 / 3.0])
        eta = np.array([1.0 / 3.0])
        w = np.array([0.5])
    elif order == 2:

        xi = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        eta = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 3:

        a = 1.0 / 5.0
        b = 3.0 / 5.0
        xi = np.array([a, a, b, 1.0 / 3.0])
        eta = np.array([a, b, a, 1.0 / 3.0])
        w = np.array([25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, -27.0 / 96.0])
    else:

        a1 = 0.445948490915965
        b1 = 0.091576213509771
        a2 = 0.108103018168070
        b2 = 0.816847572980459
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        xi = np.array([a1, 1.0 - 2.0 * a1, a1, b1, 1.0 - 2.0 * b1, b1])
        eta = np.array([a1, a1, 1.0 - 2.0 * a1, b1, b1, 1.0 - 2.0 * b1])
        w = np.array([w1, w1, w1, w2, w2, w2])
    return np.column_stack([xi, eta]), w


def integrate_triangle(f: Callable[[np.ndarray], np.ndarray],
                       nodes_phys: np.ndarray,
                       order: int = 3) -> float:
    if nodes_phys.shape != (3, 2):
        raise ValueError("nodes_phys must be (3,2)")

    qp, w = gauss_legendre_triangle(order)
    x1, y1 = nodes_phys[0]
    x2, y2 = nodes_phys[1]
    x3, y3 = nodes_phys[2]
    detJ = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    result = 0.0
    for i in range(len(w)):
        xi, eta = qp[i]
        x = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        y = y1 + (y2 - y1) * xi + (y3 - y1) * eta
        result += w[i] * f(np.array([x, y]))
    return result * detJ


def compute_section_properties(nodes: np.ndarray, triangles: np.ndarray) -> dict:
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be (N,2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be (M,3)")

    A = 0.0
    Sx = 0.0
    Sy = 0.0
    Ixx = 0.0
    Iyy = 0.0
    Ixy = 0.0

    for tri in triangles:
        tri_nodes = nodes[tri]


        A += integrate_triangle(lambda p: 1.0, tri_nodes, order=2)

        Sx += integrate_triangle(lambda p: p[1], tri_nodes, order=2)
        Sy += integrate_triangle(lambda p: p[0], tri_nodes, order=2)

        Ixx += integrate_triangle(lambda p: p[1] ** 2, tri_nodes, order=3)
        Iyy += integrate_triangle(lambda p: p[0] ** 2, tri_nodes, order=3)
        Ixy += integrate_triangle(lambda p: p[0] * p[1], tri_nodes, order=3)

    if abs(A) < 1e-14:
        A = 1e-14
    cx = Sy / A
    cy = Sx / A


    Ixx_c = Ixx - A * cy ** 2
    Iyy_c = Iyy - A * cx ** 2
    Ixy_c = Ixy - A * cx * cy


    diaphony_val = diaphony_compute(nodes)

    return {
        'A': A,
        'cx': cx,
        'cy': cy,
        'Ixx': Ixx_c,
        'Iyy': Iyy_c,
        'Ixy': Ixy_c,
        'J': Ixx_c + Iyy_c,
        'diaphony': diaphony_val
    }


def compute_shear_correction_factor(nodes: np.ndarray, triangles: np.ndarray,
                                   E: float = 1.0, nu: float = 0.35) -> float:
    props = compute_section_properties(nodes, triangles)
    A = props['A']
    if abs(A) < 1e-14:
        return 0.886



    G = E / (2.0 * (1.0 + nu))


    denom = 0.0
    for tri in triangles:
        tri_nodes = nodes[tri]

        denom += integrate_triangle(
            lambda p: (p[0] - props['cx']) ** 2 + (p[1] - props['cy']) ** 2,
            tri_nodes, order=2
        )

    if abs(denom) < 1e-14:
        return 0.886

    kappa = A ** 2 / (12.0 * denom)

    kappa = max(0.5, min(1.0, kappa))
    return kappa


def assemble_section_stiffness(nodes: np.ndarray, triangles: np.ndarray,
                               E: float, nu: float) -> np.ndarray:
    nn = nodes.shape[0]
    K = np.zeros((2 * nn, 2 * nn))


    D_mat = E / (1.0 - nu ** 2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])

    for tri in triangles:
        tri_nodes = nodes[tri]
        x = tri_nodes[:, 0]
        y = tri_nodes[:, 1]


        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-14:
            continue


        b1 = y[1] - y[2]
        b2 = y[2] - y[0]
        b3 = y[0] - y[1]
        c1 = x[2] - x[1]
        c2 = x[0] - x[2]
        c3 = x[1] - x[0]

        B = (1.0 / (2.0 * area)) * np.array([
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])

        Ke = B.T @ D_mat @ B * area


        local_dof = []
        for nid in tri:
            local_dof.extend([2 * nid, 2 * nid + 1])
        for i in range(6):
            for j in range(6):
                gi, gj = local_dof[i], local_dof[j]
                if gi < K.shape[0] and gj < K.shape[1]:
                    K[gi, gj] += Ke[i, j]

    return K
