
import numpy as np
from typing import Tuple, Optional


def compute_rmsd(coords: np.ndarray, native_coords: np.ndarray) -> float:
    if coords.shape != native_coords.shape:
        raise ValueError("coords and native_coords must have the same shape")
    diff = coords - native_coords
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_radius_of_gyration(coords: np.ndarray, masses: Optional[np.ndarray] = None) -> float:
    if masses is None:
        masses = np.ones(coords.shape[0])
    total_mass = np.sum(masses)
    if total_mass <= 0:
        raise ValueError("Total mass must be positive")
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    diff = coords - center_of_mass
    rg_sq = np.sum(masses * np.sum(diff ** 2, axis=1)) / total_mass
    return float(np.sqrt(max(rg_sq, 0.0)))


def compute_native_contact_fraction(coords: np.ndarray, native_coords: np.ndarray,
                                    contact_cutoff: float = 1.2,
                                    native_cutoff: float = 1.5) -> float:
    N = coords.shape[0]
    native_dists = np.linalg.norm(native_coords[:, np.newaxis, :] - native_coords[np.newaxis, :, :], axis=2)
    current_dists = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    
    native_contacts = (native_dists < native_cutoff) & (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > 2)

    np.fill_diagonal(native_contacts, False)
    
    total_native = np.count_nonzero(native_contacts) // 2
    if total_native == 0:
        return 0.0
    
    formed = current_dists < contact_cutoff * native_dists
    formed_contacts = formed & native_contacts
    count_formed = np.count_nonzero(formed_contacts) // 2
    return float(count_formed / total_native)


def generate_reaction_coordinate_grid(q_min: float, q_max: float, nq: int,
                                      rmsd_min: float, rmsd_max: float, nrmsd: int) -> Tuple[np.ndarray, np.ndarray]:
    if nq < 2 or nrmsd < 2:
        raise ValueError("Grid dimensions must be at least 2 in each direction")
    q_vals = np.linspace(q_min, q_max, nq)
    rmsd_vals = np.linspace(rmsd_min, rmsd_max, nrmsd)
    Q_grid, R_grid = np.meshgrid(q_vals, rmsd_vals, indexing='ij')
    nodes = np.column_stack((Q_grid.ravel(), R_grid.ravel()))
    
    elements = []
    for i in range(nq - 1):
        for j in range(nrmsd - 1):
            n0 = i * nrmsd + j
            n1 = (i + 1) * nrmsd + j
            n2 = (i + 1) * nrmsd + (j + 1)
            n3 = i * nrmsd + (j + 1)
            elements.append([n0, n1, n2, n3])
    elements = np.array(elements, dtype=int)
    return nodes, elements


def write_grid_to_file(nodes: np.ndarray, elements: np.ndarray, label: str,
                       output_dir: str = ".") -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    node_file = os.path.join(output_dir, f"{label}_nodes.txt")
    elem_file = os.path.join(output_dir, f"{label}_elements.txt")
    
    np.savetxt(node_file, nodes, fmt="%.8e", header=f"{nodes.shape[0]} {nodes.shape[1]}", comments='')
    np.savetxt(elem_file, elements, fmt="%d", header=f"{elements.shape[0]} {elements.shape[1]}", comments='')


def compute_end_to_end_distance(coords: np.ndarray) -> float:
    return float(np.linalg.norm(coords[-1] - coords[0]))


def dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-12)
    
    n1 = np.cross(b1, b2)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    
    n2 = np.cross(b2, b3)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    
    m1 = np.cross(n1, b2_norm)
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))
