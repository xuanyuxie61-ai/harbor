"""
neighbor_list.py

Neighbor list construction for molecular dynamics.

Synthesizes concepts from:
    - 489_grf_display: Graph adjacency structure for node connectivity
    - 1233_tet_mesh_l2q: Mesh topology and edge detection
"""

import numpy as np
from config import R_CUTOFF, R_CUTOFF_SQ, MIN_DISTANCE


def build_neighbor_list(positions, box, rcut, skin=0.0):
    """
    Build a Verlet neighbor list using vectorized distance computation.
    
    For small to moderate system sizes, computing the full distance matrix
    is efficient and straightforward.
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        rcut: cutoff distance
        skin: skin distance for Verlet list
        
    Returns:
        neighbors: list of lists, neighbors[i] contains indices j > i
        neighbor_counts: (N,) array with number of neighbors
        distances_sq: list of arrays with squared distances
    """
    n_atoms = positions.shape[0]
    rcut_eff = rcut + skin
    rcut_eff_sq = rcut_eff ** 2
    
    neighbors = [[] for _ in range(n_atoms)]
    distances_sq = [[] for _ in range(n_atoms)]
    neighbor_counts = np.zeros(n_atoms, dtype=int)
    
    # Vectorized distance computation
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
    """
    Compute the degree distribution of the neighbor graph.
    
    Args:
        neighbors: list of neighbor lists
        n_atoms: total number of atoms
        
    Returns:
        degrees: (N,) array with degree of each node
        avg_degree: average degree
        max_degree: maximum degree
    """
    degrees = np.zeros(n_atoms, dtype=int)
    
    for i in range(n_atoms):
        degrees[i] = len(neighbors[i])
    
    # Count reverse connections
    for i in range(n_atoms):
        for j in neighbors[i]:
            degrees[j] += 1
    
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    
    return degrees, avg_degree, max_degree


def compute_coordination_shells(positions, box, rcut, n_shells=3):
    """
    Compute coordination shells using the neighbor graph.
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        rcut: cutoff for neighbor search
        n_shells: number of shells to compute
        
    Returns:
        shell_counts: (N, n_shells) array with neighbor counts per shell
        shell_radii: (n_shells,) array with average shell radii
    """
    n_atoms = positions.shape[0]
    
    # Build full neighbor list with distances
    neighbors, counts, dists_sq = build_neighbor_list(positions, box, rcut, skin=0.0)
    
    # Collect all distances
    all_dists = []
    for i in range(n_atoms):
        all_dists.extend(np.sqrt(np.array(dists_sq[i])))
    
    all_dists = np.array(all_dists)
    
    if len(all_dists) == 0:
        return np.zeros((n_atoms, n_shells)), np.zeros(n_shells)
    
    # Determine shell boundaries using histogram
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
    
    # Count neighbors in each shell for each atom
    shell_counts = np.zeros((n_atoms, n_shells), dtype=int)
    
    for i in range(n_atoms):
        dists = np.sqrt(np.array(dists_sq[i]))
        for s in range(n_shells):
            r_min = shell_boundaries[s]
            r_max = shell_boundaries[s + 1]
            shell_counts[i, s] = np.sum((dists >= r_min) & (dists < r_max))
    
    return shell_counts, shell_radii
