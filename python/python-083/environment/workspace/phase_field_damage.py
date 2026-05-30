
import numpy as np
from typing import Tuple, Optional






def degradation_function(phi: np.ndarray, k_res: float = 1e-6) -> np.ndarray:
    return (1.0 - phi)**2 + k_res


def degradation_derivative(phi: np.ndarray) -> np.ndarray:
    return -2.0 * (1.0 - phi)


def double_well_potential(phi: np.ndarray) -> np.ndarray:
    return phi**2 * (1.0 - phi)**2


def double_well_derivative(phi: np.ndarray) -> np.ndarray:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)






def compute_elastic_strain_energy(U: np.ndarray, K_dense: np.ndarray) -> float:
    return 0.5 * np.dot(U, K_dense @ U)


def split_strain_energy(U: np.ndarray, node_xy: np.ndarray,
                        element_node: np.ndarray, E: float, nu: float,
                        plane_stress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    n_elements = element_node.shape[0]
    psi_pos = np.zeros(n_elements, dtype=np.float64)
    psi_neg = np.zeros(n_elements, dtype=np.float64)










    raise NotImplementedError("Hole 3: split_strain_energy 未实现")
    return psi_pos, psi_neg






def solve_phase_field_evolution(element_node: np.ndarray, node_xy: np.ndarray,
                                 psi_e_pos: np.ndarray, G_c: float,
                                 l_0: float, n_iter: int = 50,
                                 mobility: float = 1.0) -> np.ndarray:
    n_elements = element_node.shape[0]
    phi = np.zeros(n_elements, dtype=np.float64)


    psi_hist = psi_e_pos.copy()


    from fem_core import build_vtoe
    vtoe_ptr, vtoe = build_vtoe(element_node, node_xy.shape[0])


    adjacency = [set() for _ in range(n_elements)]
    for v in range(node_xy.shape[0]):
        cells = vtoe[vtoe_ptr[v]:vtoe_ptr[v+1]]
        for ci in cells:
            for cj in cells:
                if ci != cj:
                    adjacency[ci].add(cj)


    centers = np.zeros((n_elements, 2), dtype=np.float64)
    for e in range(n_elements):
        centers[e] = np.mean(node_xy[element_node[e, :], :], axis=0)


    areas = np.zeros(n_elements, dtype=np.float64)
    for e in range(n_elements):
        x = node_xy[element_node[e, :], 0]
        y = node_xy[element_node[e, :], 1]
        areas[e] = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))

    for _ in range(n_iter):
        phi_new = np.zeros_like(phi)
        for e in range(n_elements):

            lap = 0.0
            total_weight = 0.0
            for j in adjacency[e]:
                dist2 = np.sum((centers[j] - centers[e])**2)
                if dist2 > 1e-14:
                    weight = 1.0 / dist2
                    lap += weight * (phi[j] - phi[e])
                    total_weight += weight
            if total_weight > 1e-14:
                lap /= total_weight
            else:
                lap = 0.0


            coeff = 2.0 * psi_hist[e] + G_c / l_0 + mobility * G_c * l_0 * total_weight
            rhs = 2.0 * psi_hist[e] + mobility * G_c * l_0 * lap
            if coeff > 1e-14:
                phi_new[e] = rhs / coeff
            else:
                phi_new[e] = 0.0


            phi_new[e] = max(0.0, min(1.0, phi_new[e]))


        if np.max(np.abs(phi_new - phi)) < 1e-6:
            break
        phi = phi_new

    return phi


def compute_crack_driving_force(node_xy: np.ndarray, element_node: np.ndarray,
                                 U: np.ndarray, E: float, nu: float,
                                 G_c: float, l_0: float,
                                 plane_stress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi_pos, psi_neg = split_strain_energy(
        U, node_xy, element_node, E, nu, plane_stress)
    phi = solve_phase_field_evolution(element_node, node_xy, psi_pos, G_c, l_0)



    areas = np.zeros(len(psi_pos), dtype=np.float64)
    for e in range(len(psi_pos)):
        x = node_xy[element_node[e, :], 0]
        y = node_xy[element_node[e, :], 1]
        areas[e] = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))

    J_integral = np.sum(G_c * (phi / l_0) * areas)
    return psi_pos, phi, J_integral
