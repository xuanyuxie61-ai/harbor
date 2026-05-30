
import numpy as np
from typing import Tuple, List


def partition_greedy(w: np.ndarray) -> np.ndarray:
    w = np.array(w, dtype=float)
    n = len(w)
    if n == 0:
        return np.array([])
    

    idx_sorted = np.argsort(-w)
    x = np.zeros(n, dtype=int)
    
    s0 = 0.0
    s1 = 0.0
    for idx in idx_sorted:
        if s0 < s1:
            x[idx] = 0
            s0 += w[idx]
        else:
            x[idx] = 1
            s1 += w[idx]
    
    return x


def partition_residues_by_contact(contacts: np.ndarray, n_partitions: int = 2) -> List[np.ndarray]:
    N = contacts.shape[0]
    total_weights = np.sum(contacts, axis=1)
    
    groups = [np.arange(N)]
    while len(groups) < n_partitions:
        new_groups = []
        for g in groups:
            if len(g) <= 1:
                new_groups.append(g)
                continue
            sub_weights = total_weights[g]
            partition = partition_greedy(sub_weights)
            g0 = g[partition == 0]
            g1 = g[partition == 1]
            new_groups.append(g0)
            new_groups.append(g1)
        groups = new_groups
    
    return groups


def partition_free_energy_landscape(energies: np.ndarray, n_bins: int = 4) -> List[Tuple[float, float]]:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    

    w = np.exp(-energies)
    sorted_idx = np.argsort(energies)
    sorted_w = w[sorted_idx]
    
    total_weight = np.sum(sorted_w)
    target = total_weight / n_bins
    
    ranges = []
    start_idx = 0
    current_weight = 0.0
    
    for i in range(len(sorted_w)):
        current_weight += sorted_w[i]
        if current_weight >= target or i == len(sorted_w) - 1:
            e_min = energies[sorted_idx[start_idx]]
            e_max = energies[sorted_idx[i]]
            ranges.append((float(e_min), float(e_max)))
            start_idx = i + 1
            current_weight = 0.0
            if len(ranges) >= n_bins:
                break
    

    if start_idx < len(sorted_w) and len(ranges) < n_bins:
        e_min = energies[sorted_idx[start_idx]]
        e_max = energies[sorted_idx[-1]]
        ranges.append((float(e_min), float(e_max)))
    
    return ranges
