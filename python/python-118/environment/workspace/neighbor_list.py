
import numpy as np
from config import R_CUTOFF, R_CUTOFF_SQ, MIN_DISTANCE


def build_neighbor_list(positions, box, rcut, skin=0.0):
    n_atoms = positions.shape[0]
    rcut_eff = rcut + skin
    rcut_eff_sq = rcut_eff ** 2
    
    neighbors = [[] for _ in range(n_atoms)]
    distances_sq = [[] for _ in range(n_atoms)]
    neighbor_counts = np.zeros(n_atoms, dtype=int)
    

    for i in range(n_atoms):
        dr = positions - positions[i]
        dr -= box * np.round(dr / box)
        r_sq = np.sum(dr ** 2, axis=1)
        
        mask = (r_sq < rcut_eff_sq) & (r_sq > MIN_DISTANCE ** 2) & (np.arange(n_atoms) > i)
        j_indices = np.where(mask)[0]
        
        neighbors[i] = j_indices.tolist()
        distances_sq[i] = r_sq[mask].tolist()
        neighbor_counts[i] = len(j_indices)
    
    return neighbors, neighbor_counts, distances_sq


def compute_graph_degree(neighbors, n_atoms):
    degrees = np.zeros(n_atoms, dtype=int)
    
    for i in range(n_atoms):
        degrees[i] = len(neighbors[i])
    

    for i in range(n_atoms):
        for j in neighbors[i]:
            degrees[j] += 1
    
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    
    return degrees, avg_degree, max_degree


def compute_coordination_shells(positions, box, rcut, n_shells=3):
    n_atoms = positions.shape[0]
    

    neighbors, counts, dists_sq = build_neighbor_list(positions, box, rcut, skin=0.0)
    

    all_dists = []
    for i in range(n_atoms):
        all_dists.extend(np.sqrt(np.array(dists_sq[i])))
    
    all_dists = np.array(all_dists)
    
    if len(all_dists) == 0:
        return np.zeros((n_atoms, n_shells)), np.zeros(n_shells)
    

    shell_boundaries = np.linspace(0, rcut, n_shells + 1)
    
    shell_radii = np.zeros(n_shells)
    for s in range(n_shells):
        r_min = shell_boundaries[s]
        r_max = shell_boundaries[s + 1]
        mask = (all_dists >= r_min) & (all_dists < r_max)
        if np.any(mask):
            shell_radii[s] = np.mean(all_dists[mask])
        else:
            shell_radii[s] = (r_min + r_max) / 2.0
    

    shell_counts = np.zeros((n_atoms, n_shells), dtype=int)
    
    for i in range(n_atoms):
        dists = np.sqrt(np.array(dists_sq[i]))
        for s in range(n_shells):
            r_min = shell_boundaries[s]
            r_max = shell_boundaries[s + 1]
            shell_counts[i, s] = np.sum((dists >= r_min) & (dists < r_max))
    
    return shell_counts, shell_radii
