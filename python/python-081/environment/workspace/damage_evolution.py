
import numpy as np
from typing import Tuple, Optional


def heaviside(x: float) -> float:
    if x > 1e-6:
        return 1.0
    elif x < -1e-6:
        return 0.0
    else:
        return 0.5 + x / (2e-6)


def damage_evolution_rate(D: float, eps_p: float,
                           A: float = 2.0, eps_f: float = 0.5) -> float:
    if D < 0:
        D = 0.0
    if D > 1:
        D = 1.0
    if eps_f < 1e-12:
        eps_f = 1e-12
    ratio = eps_p / eps_f

    sat = min(ratio, 1.0)
    rate = A * D * (sat - D)

    if D > 0.99:
        rate = min(rate, 0.0)
    return rate


def plastic_strain_rate(eps_dot_eq: float, D: float,
                         B: float = 1.0, sigma_vm: float = 0.0,
                         sigma_y: float = 1e6) -> float:
    H = heaviside(sigma_vm - sigma_y)
    rate = B * eps_dot_eq * (1.0 - D) * H
    return rate


def forward_euler_damage_step(D_n: float, eps_p_n: float,
                               dt: float, eps_dot_eq: float,
                               sigma_vm: float, sigma_y: float,
                               A: float = 2.0, B: float = 1.0,
                               eps_f: float = 0.5) -> Tuple[float, float]:
    dDdt = damage_evolution_rate(D_n, eps_p_n, A, eps_f)
    depdt = plastic_strain_rate(eps_dot_eq, D_n, B, sigma_vm, sigma_y)

    D_next = D_n + dt * dDdt
    eps_p_next = eps_p_n + dt * depdt


    D_next = np.clip(D_next, 0.0, 1.0)
    eps_p_next = max(eps_p_next, 0.0)

    return D_next, eps_p_next


def update_element_damage(n_elements: int,
                          D_elements: np.ndarray,
                          eps_p_elements: np.ndarray,
                          dt: float,
                          eps_dot_elements: np.ndarray,
                          sigma_vm_elements: np.ndarray,
                          sigma_y: float = 1e6,
                          A: float = 2.0, B: float = 1.0,
                          eps_f: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    D_new = np.zeros(n_elements, dtype=np.float64)
    eps_p_new = np.zeros(n_elements, dtype=np.float64)

    for e in range(n_elements):
        D_new[e], eps_p_new[e] = forward_euler_damage_step(
            float(D_elements[e]), float(eps_p_elements[e]), dt,
            float(eps_dot_elements[e]), float(sigma_vm_elements[e]),
            sigma_y, A, B, eps_f
        )

    return D_new, eps_p_new


def compute_equivalent_strain_rate(u_current: np.ndarray,
                                    u_prev: np.ndarray,
                                    dt: float,
                                    elements: np.ndarray,
                                    nodes: np.ndarray) -> np.ndarray:
    from hyperelastic_constitutive import deformation_gradient, right_cauchy_green, green_lagrange_strain
    from tetrahedral_mesh import tetrahedron_volume

    n_elements = elements.shape[0]
    eps_dot = np.zeros(n_elements, dtype=np.float64)


    _, dN_dxi = tetrahedron_volume, None




    for e_idx, e in enumerate(elements):
        nodes_e = nodes[e]

        x0, x1, x2, x3 = nodes_e[0], nodes_e[1], nodes_e[2], nodes_e[3]
        mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
        detJ0 = np.linalg.det(mat)
        if abs(detJ0) < 1e-14:
            eps_dot[e_idx] = 0.0
            continue


        dN_dxi = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        J0 = dN_dxi.T @ nodes_e
        J0_inv = np.linalg.inv(J0)
        dN_dX = dN_dxi @ J0_inv.T

        u_e_cur = u_current[3 * e[:, None] + np.arange(3)].reshape(4, 3)
        u_e_prev = u_prev[3 * e[:, None] + np.arange(3)].reshape(4, 3)

        F_cur = deformation_gradient(dN_dX, u_e_cur)
        F_prev = deformation_gradient(dN_dX, u_e_prev)

        E_cur = green_lagrange_strain(right_cauchy_green(F_cur))
        E_prev = green_lagrange_strain(right_cauchy_green(F_prev))
        dE = E_cur - E_prev
        eps_dot[e_idx] = np.sqrt(2.0 * np.sum(dE * dE)) / max(dt, 1e-12)

    return eps_dot
