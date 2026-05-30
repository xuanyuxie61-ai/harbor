
import numpy as np
from typing import Tuple, List
from berry_curvature import berry_curvature_numeric, chern_number_2d_slice, weyl_charge_surface_integral


def compute_chern_numbers_vs_kz(ham, kz_values: np.ndarray,
                                 kx_range: Tuple[float, float],
                                 ky_range: Tuple[float, float],
                                 grid_size: int = 30,
                                 band_index: int = 0) -> np.ndarray:
    chern_numbers = np.zeros(len(kz_values))
    
    for i, kz in enumerate(kz_values):
        c = chern_number_2d_slice(ham, kx_range, ky_range, kz, grid_size, band_index)

        chern_numbers[i] = round(c)
    
    return chern_numbers


def locate_weyl_nodes_from_chern_jump(kz_values: np.ndarray,
                                       chern_numbers: np.ndarray) -> np.ndarray:
    jumps = np.diff(chern_numbers)
    
    node_positions = []
    node_charges = []
    
    for i in range(len(jumps)):
        if abs(jumps[i]) > 0.5:

            pos = 0.5 * (kz_values[i] + kz_values[i + 1])
            charge = int(round(jumps[i]))
            node_positions.append(pos)
            node_charges.append(charge)
    
    return np.array(node_positions), np.array(node_charges)


def compute_weyl_charges_spherical(ham, weyl_nodes: np.ndarray,
                                    radius: float = 0.3,
                                    n_theta: int = 16,
                                    n_phi: int = 16,
                                    band_index: int = 0) -> np.ndarray:
    n_nodes = weyl_nodes.shape[0] if weyl_nodes.ndim > 1 else 1
    charges = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        node = weyl_nodes[i] if weyl_nodes.ndim > 1 else weyl_nodes
        q = weyl_charge_surface_integral(ham, node, radius, n_theta, n_phi, band_index)

        charges[i] = round(q)
    
    return charges


def nielsen_ninomiya_theorem_check(charges: np.ndarray) -> bool:
    total = np.sum(charges)
    return abs(total) < 0.5


def berry_phase_wilson_loop(ham, kx_line: np.ndarray, ky_fixed: float,
                            kz_fixed: float, band_index: int = 0) -> float:
    n_points = len(kx_line)
    if n_points < 2:
        return 0.0
    

    vectors = []
    for kx in kx_line:
        k = np.array([kx, ky_fixed, kz_fixed])
        _, eigvecs = ham.eigenproblem(k)
        vec = eigvecs[:, band_index].copy()

        if abs(vec[0]) > 1e-14:
            vec *= np.exp(-1.0j * np.angle(vec[0]))
        vectors.append(vec)
    

    prod = 1.0 + 0.0j
    for i in range(n_points - 1):
        overlap = np.vdot(vectors[i], vectors[i + 1])
        prod *= overlap
    

    overlap = np.vdot(vectors[-1], vectors[0])
    prod *= overlap
    
    phase = -np.angle(prod)
    return phase


def compute_z2_index(ham, kx_values: np.ndarray, ky_values: np.ndarray,
                     kz_fixed: float, band_index: int = 0) -> int:

    phases = []
    for ky in ky_values:
        phase = berry_phase_wilson_loop(ham, kx_values, ky, kz_fixed, band_index)
        phases.append(phase)
    
    phases = np.array(phases)
    


    z2 = 0
    for p in phases:
        if abs(abs(p) - np.pi) < 0.3 * np.pi:
            z2 = 1
            break
    
    return z2
