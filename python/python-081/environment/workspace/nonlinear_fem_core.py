
import numpy as np
from typing import Tuple, Optional, List
from hyperelastic_constitutive import (
    deformation_gradient, right_cauchy_green,
    neo_hookean_pk2_stress, neo_hookean_material_tangent,
    green_lagrange_strain, voigt_strain, voigt_stress,
    cauchy_stress_from_pk2, von_mises_cauchy,
    solve_effective_shear_modulus
)
from stiffness_solver import apply_dirichlet_to_system


def tet_p1_shape_derivatives() -> Tuple[np.ndarray, np.ndarray]:

    xi = 1.0 / 4.0
    eta = 1.0 / 4.0
    zeta = 1.0 / 4.0
    N = np.array([1.0 - xi - eta - zeta, xi, eta, zeta], dtype=np.float64)
    dN_dxi = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return N, dN_dxi


def compute_element_jacobian_and_dNdX(nodes_e: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    _, dN_dxi = tet_p1_shape_derivatives()
    J0 = dN_dxi.T @ nodes_e
    detJ0 = float(np.linalg.det(J0))
    if abs(detJ0) < 1e-14:
        raise ValueError(f"单元Jacobian行列式接近零: {detJ0}")
    J0_inv = np.linalg.inv(J0)
    dN_dX = dN_dxi @ J0_inv.T
    return detJ0, J0, dN_dX


def compute_B_matrix(F: np.ndarray, dN_dX: np.ndarray) -> np.ndarray:

    raise NotImplementedError("Hole 2: 请实现compute_B_matrix")


def compute_G_matrix(dN_dX: np.ndarray) -> np.ndarray:

    raise NotImplementedError("Hole 2: 请实现compute_G_matrix")


def assemble_element_stiffness_force(nodes_e: np.ndarray, u_e: np.ndarray,
                                      mu: float, lam: float,
                                      use_damage: bool = False,
                                      gamma_e: float = 0.0,
                                      alpha_damage: float = 0.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    detJ0, _, dN_dX = compute_element_jacobian_and_dNdX(nodes_e)
    F = deformation_gradient(dN_dX, u_e)
    C = right_cauchy_green(F)


    if use_damage and gamma_e > 1e-12 and alpha_damage > 0:
        mu_eff = solve_effective_shear_modulus(mu, gamma_e, alpha_damage)
    else:
        mu_eff = mu

    try:
        S = neo_hookean_pk2_stress(C, mu_eff, lam)
        C_mat = neo_hookean_material_tangent(C, mu_eff, lam)
    except ValueError:

        k_e = np.eye(12, dtype=np.float64) * 1e-6
        f_int = np.zeros(12, dtype=np.float64)
        stress_data = {
            "F": F, "C": C, "S": np.zeros((3,3)),
            "sigma": np.zeros((3,3)), "sigma_vm": 0.0,
            "gamma": 0.0, "mu_eff": mu_eff,
        }
        return k_e, f_int, stress_data

    B = compute_B_matrix(F, dN_dX)
    G = compute_G_matrix(dN_dX)


    S_mtx = np.zeros((3, 3), dtype=np.float64)
    S_mtx[0, 0] = S[0, 0]
    S_mtx[1, 1] = S[1, 1]
    S_mtx[2, 2] = S[2, 2]
    S_mtx[0, 1] = S_mtx[1, 0] = S[0, 1]
    S_mtx[0, 2] = S_mtx[2, 0] = S[0, 2]
    S_mtx[1, 2] = S_mtx[2, 1] = S[1, 2]

    S9 = np.zeros((9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            S9[i * 3 + j, i * 3 + j] = S_mtx[i, j]
            if j != i:
                S9[i * 3 + j, j * 3 + i] = S_mtx[i, j]


    w = abs(detJ0) / 6.0


    k_mat = B.T @ C_mat @ B * w

    k_geo = G.T @ S9 @ G * w
    k_e = k_mat + k_geo


    f_int = B.T @ voigt_stress(S) * w


    sigma = cauchy_stress_from_pk2(F, S)
    sigma_vm = von_mises_cauchy(sigma)


    E = green_lagrange_strain(C)
    gamma_calc = float(np.sqrt(2.0 * np.sum(E * E)))

    stress_data = {
        "F": F,
        "C": C,
        "S": S,
        "sigma": sigma,
        "sigma_vm": sigma_vm,
        "gamma": gamma_calc,
        "mu_eff": mu_eff,
    }

    return k_e, f_int, stress_data


def assemble_global_system(nodes: np.ndarray, elements: np.ndarray,
                            u: np.ndarray, mu: float, lam: float,
                            use_damage: bool = False,
                            gamma_elements: Optional[np.ndarray] = None,
                            alpha_damage: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    n_nodes = nodes.shape[0]
    n_dof = 3 * n_nodes
    K_global = np.zeros((n_dof, n_dof), dtype=np.float64)
    R = np.zeros(n_dof, dtype=np.float64)
    stress_list = []

    if gamma_elements is None:
        gamma_elements = np.zeros(elements.shape[0], dtype=np.float64)

    for e_idx, e in enumerate(elements):
        nodes_e = nodes[e]
        u_e = u[3 * e[:, None] + np.arange(3)].reshape(4, 3)
        gamma_e = gamma_elements[e_idx]

        k_e, f_int, sdata = assemble_element_stiffness_force(
            nodes_e, u_e, mu, lam, use_damage, gamma_e, alpha_damage
        )
        stress_list.append(sdata)


        dof_map = []
        for n in e:
            dof_map.extend([3 * n, 3 * n + 1, 3 * n + 2])
        dof_map = np.array(dof_map, dtype=np.int32)

        for i_local in range(12):
            i_global = dof_map[i_local]
            R[i_global] += f_int[i_local]
            for j_local in range(12):
                j_global = dof_map[j_local]
                K_global[i_global, j_global] += k_e[i_local, j_local]

    return K_global, R, stress_list


def compute_external_force(nodes: np.ndarray, elements: np.ndarray,
                            surface_tris: np.ndarray,
                            traction: np.ndarray) -> np.ndarray:
    n_nodes = nodes.shape[0]
    F_ext = np.zeros(3 * n_nodes, dtype=np.float64)

    for tri in surface_tris:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))

        for n in tri:
            F_ext[3 * n:3 * n + 3] += traction * (area / 3.0)

    return F_ext
