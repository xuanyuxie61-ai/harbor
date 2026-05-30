
import numpy as np
from typing import Tuple, Optional












def t6_shape_functions(r: float, s: float) -> np.ndarray:
    t = 1.0 - r - s
    N = np.zeros(6)
    N[0] = t * (2.0 * t - 1.0)
    N[1] = r * (2.0 * r - 1.0)
    N[2] = s * (2.0 * s - 1.0)
    N[3] = 4.0 * r * t
    N[4] = 4.0 * r * s
    N[5] = 4.0 * s * t
    return N


def t6_shape_derivatives(r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
    t = 1.0 - r - s
    dN_dr = np.zeros(6)
    dN_ds = np.zeros(6)

    dN_dr[0] = -4.0 * t + 1.0
    dN_dr[1] = 4.0 * r - 1.0
    dN_dr[2] = 0.0
    dN_dr[3] = 4.0 * (t - r)
    dN_dr[4] = 4.0 * s
    dN_dr[5] = -4.0 * s

    dN_ds[0] = -4.0 * t + 1.0
    dN_ds[1] = 0.0
    dN_ds[2] = 4.0 * s - 1.0
    dN_ds[3] = -4.0 * r
    dN_ds[4] = 4.0 * r
    dN_ds[5] = 4.0 * (t - s)

    return dN_dr, dN_ds


def t6_jacobian(
    nodes: np.ndarray, r: float, s: float
) -> Tuple[np.ndarray, float]:
    dN_dr, dN_ds = t6_shape_derivatives(r, s)

    J = np.zeros((2, 2))
    J[0, 0] = np.dot(nodes[:, 0], dN_dr)
    J[0, 1] = np.dot(nodes[:, 0], dN_ds)
    J[1, 0] = np.dot(nodes[:, 1], dN_dr)
    J[1, 1] = np.dot(nodes[:, 1], dN_ds)

    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return J, det_J


def gauss_points_triangle_t6(order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if order == 1:
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])
    elif order == 3:
        points = np.array([
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 6.0, 1.0 / 6.0],
        ])
        weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 4:

        a = 1.0 / 3.0
        b = 0.6
        c = 0.2
        w1 = -27.0 / 96.0
        w2 = 25.0 / 96.0
        points = np.array([
            [a, a],
            [b, c],
            [c, b],
            [c, c],
        ])
        weights = np.array([w1, w2, w2, w2])
    else:
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])

    return points, weights


def plane_stress_constitutive_matrix(
    E: float, nu: float
) -> np.ndarray:



    pass


def plane_strain_constitutive_matrix(
    E: float, nu: float
) -> np.ndarray:
    factor = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = factor * np.array([
        [1.0, nu / (1.0 - nu), 0.0],
        [nu / (1.0 - nu), 1.0, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))],
    ])
    return D


def compute_B_matrix_t6(
    nodes: np.ndarray, r: float, s: float
) -> np.ndarray:
    dN_dr, dN_ds = t6_shape_derivatives(r, s)
    J, det_J = t6_jacobian(nodes, r, s)

    if abs(det_J) < 1e-14:
        det_J = 1e-14


    J_inv = np.array([
        [J[1, 1], -J[0, 1]],
        [-J[1, 0], J[0, 0]],
    ]) / det_J


    dN_dx = J_inv[0, 0] * dN_dr + J_inv[0, 1] * dN_ds
    dN_dy = J_inv[1, 0] * dN_dr + J_inv[1, 1] * dN_ds

    B = np.zeros((3, 12))
    for i in range(6):
        B[0, 2 * i] = dN_dx[i]
        B[1, 2 * i + 1] = dN_dy[i]
        B[2, 2 * i] = dN_dy[i]
        B[2, 2 * i + 1] = dN_dx[i]

    return B


def assemble_stiffness_matrix_t6(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
    thickness: float = 1.0,
    plane_stress: bool = True,
) -> np.ndarray:
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))

    D = plane_stress_constitutive_matrix(E, nu) if plane_stress else plane_strain_constitutive_matrix(E, nu)
    gp_points, gp_weights = gauss_points_triangle_t6(order=3)

    for elem in elements:
        elem_nodes = nodes[elem]
        ke = np.zeros((12, 12))

        for gp, w in zip(gp_points, gp_weights):
            r, s = gp
            B = compute_B_matrix_t6(elem_nodes, r, s)
            _, det_J = t6_jacobian(elem_nodes, r, s)

            if det_J <= 0:
                det_J = abs(det_J) + 1e-14

            ke += w * det_J * thickness * (B.T @ D @ B)


        dof_map = []
        for node_idx in elem:
            dof_map.extend([2 * node_idx, 2 * node_idx + 1])

        for i in range(12):
            for j in range(12):
                K[dof_map[i], dof_map[j]] += ke[i, j]

    return K


def apply_dirichlet_boundary(
    K: np.ndarray, F: np.ndarray, bc_nodes: np.ndarray,
    bc_values: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    K_mod = K.copy()
    F_mod = F.copy()
    big_number = 1e20

    if bc_values is None:
        bc_values = np.zeros(len(bc_nodes))

    for idx, node_dof in enumerate(bc_nodes):
        val = bc_values[idx]
        K_mod[node_dof, node_dof] += big_number
        F_mod[node_dof] = (K_mod[node_dof, node_dof]) * val

    return K_mod, F_mod


def compute_nodal_forces_uniform(
    nodes: np.ndarray, elements: np.ndarray,
    qx: float, qy: float, thickness: float = 1.0
) -> np.ndarray:
    n_nodes = len(nodes)
    F = np.zeros(2 * n_nodes)

    gp_points, gp_weights = gauss_points_triangle_t6(order=3)

    for elem in elements:
        elem_nodes = nodes[elem]
        fe = np.zeros(12)

        for gp, w in zip(gp_points, gp_weights):
            r, s = gp
            N = t6_shape_functions(r, s)
            _, det_J = t6_jacobian(elem_nodes, r, s)
            if det_J <= 0:
                det_J = abs(det_J) + 1e-14

            for i in range(6):
                fe[2 * i] += w * det_J * thickness * qx * N[i]
                fe[2 * i + 1] += w * det_J * thickness * qy * N[i]

        dof_map = []
        for node_idx in elem:
            dof_map.extend([2 * node_idx, 2 * node_idx + 1])

        for i in range(12):
            F[dof_map[i]] += fe[i]

    return F


def compute_equivalent_creep_load(
    nodes: np.ndarray,
    elements: np.ndarray,
    epsilon_creep: np.ndarray,
    E_eff: float,
    nu: float,
    thickness: float = 1.0,
    plane_stress: bool = True,
) -> np.ndarray:
    n_nodes = len(nodes)
    F_cr = np.zeros(2 * n_nodes)

    D = plane_stress_constitutive_matrix(E_eff, nu) if plane_stress else plane_strain_constitutive_matrix(E_eff, nu)
    gp_points, gp_weights = gauss_points_triangle_t6(order=3)

    for e, elem in enumerate(elements):
        elem_nodes = nodes[elem]
        fecr = np.zeros(12)

        eps_cr = epsilon_creep[e] if len(epsilon_creep.shape) > 1 else epsilon_creep

        for gp, w in zip(gp_points, gp_weights):
            r, s = gp
            B = compute_B_matrix_t6(elem_nodes, r, s)
            _, det_J = t6_jacobian(elem_nodes, r, s)
            if det_J <= 0:
                det_J = abs(det_J) + 1e-14

            sigma_cr = D @ eps_cr
            fecr += w * det_J * thickness * (B.T @ sigma_cr)

        dof_map = []
        for node_idx in elem:
            dof_map.extend([2 * node_idx, 2 * node_idx + 1])

        for i in range(12):
            F_cr[dof_map[i]] += fecr[i]

    return F_cr


def compute_strain_stress_at_nodes(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    E: float,
    nu: float,
    plane_stress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    n_nodes = len(nodes)
    strains = np.zeros((n_nodes, 3))
    stresses = np.zeros((n_nodes, 3))
    count = np.zeros(n_nodes)

    D = plane_stress_constitutive_matrix(E, nu) if plane_stress else plane_strain_constitutive_matrix(E, nu)

    for elem in elements:
        elem_nodes = nodes[elem]

        r, s = 1.0 / 3.0, 1.0 / 3.0
        B = compute_B_matrix_t6(elem_nodes, r, s)

        d_elem = np.zeros(12)
        for i, node_idx in enumerate(elem):
            d_elem[2 * i] = displacements[2 * node_idx]
            d_elem[2 * i + 1] = displacements[2 * node_idx + 1]

        eps = B @ d_elem
        sig = D @ eps

        for node_idx in elem:
            strains[node_idx] += eps
            stresses[node_idx] += sig
            count[node_idx] += 1


    for i in range(n_nodes):
        if count[i] > 0:
            strains[i] /= count[i]
            stresses[i] /= count[i]

    return strains, stresses
