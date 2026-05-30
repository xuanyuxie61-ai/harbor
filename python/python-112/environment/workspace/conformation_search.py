
import numpy as np
from typing import List, Tuple, Callable, Optional





def greedy_conformation_search(
    n_rotatable: int,
    n_bins_per_torsion: int,
    energy_grid: np.ndarray,
    start_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, float]:
    if n_rotatable < 1:
        raise ValueError("greedy_conformation_search: n_rotatable must be >= 1.")
    if n_bins_per_torsion < 2:
        raise ValueError("greedy_conformation_search: n_bins_per_torsion must be >= 2.")
    if energy_grid.ndim != 3:
        raise ValueError("greedy_conformation_search: energy_grid must be 3D.")
    if energy_grid.shape != (n_rotatable, n_bins_per_torsion, n_bins_per_torsion):
        raise ValueError("greedy_conformation_search: energy_grid shape mismatch.")

    if start_indices is None:
        start_indices = list(range(n_bins_per_torsion))

    best_sequence = None
    best_energy = float('inf')

    for start in start_indices:
        seq = np.zeros(n_rotatable, dtype=int)
        seq[0] = start
        total_energy = 0.0

        for i in range(n_rotatable - 1):
            current_state = seq[i]

            next_energies = energy_grid[i, current_state, :]
            next_state = int(np.argmin(next_energies))
            seq[i + 1] = next_state
            total_energy += next_energies[next_state]

        if total_energy < best_energy:
            best_energy = total_energy
            best_sequence = seq.copy()

    return best_sequence, best_energy


def path_cost(n: int, distance: np.ndarray, p: np.ndarray) -> float:
    if distance.shape != (n, n):
        raise ValueError("path_cost: distance matrix must be square.")
    if p.shape[0] != n:
        raise ValueError("path_cost: p length must equal n.")

    cost = 0.0
    for i2 in range(n):
        i1 = (i2 - 1) % n
        cost += distance[p[i1], p[i2]]
    return cost





def backtrack_search(
    n_vars: int,
    domain_size: int,
    constraint_checker: Callable[[List[int]], bool],
    max_solutions: int = 1000,
) -> List[List[int]]:
    if n_vars < 1:
        raise ValueError("backtrack_search: n_vars must be >= 1.")
    if domain_size < 1:
        raise ValueError("backtrack_search: domain_size must be >= 1.")
    if max_solutions < 1:
        raise ValueError("backtrack_search: max_solutions must be >= 1.")

    solutions: List[List[int]] = []
    current = [0] * n_vars

    def _bt(pos: int):
        if len(solutions) >= max_solutions:
            return
        if pos == n_vars:
            if constraint_checker(current):
                solutions.append(current.copy())
            return

        for val in range(domain_size):
            current[pos] = val

            if constraint_checker(current[:pos + 1]):
                _bt(pos + 1)

    _bt(0)
    return solutions





def dock_drug_greedy_rotamer(
    n_torsions: int = 5,
    n_bins: int = 12,
    vdw_radius_drug: float = 3.5,
    vdw_radius_pocket: np.ndarray = None,
    pocket_coords: np.ndarray = None,
    base_energy: float = -5.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    if not (1 <= n_torsions <= 10):
        raise ValueError("dock_drug_greedy_rotamer: n_torsions must be in [1, 10].")
    if n_bins < 2:
        raise ValueError("dock_drug_greedy_rotamer: n_bins must be >= 2.")
    if vdw_radius_drug <= 0:
        raise ValueError("dock_drug_greedy_rotamer: vdw_radius_drug must be > 0.")


    np.random.seed(42)
    energy_grid = np.zeros((n_torsions, n_bins, n_bins), dtype=float)

    for i in range(n_torsions):
        for a in range(n_bins):
            for b in range(n_bins):

                phi_a = 2.0 * np.pi * a / n_bins
                phi_b = 2.0 * np.pi * b / n_bins
                e_torsion = 0.5 * (1.0 + np.cos(3.0 * phi_a - np.pi / 4.0))
                e_torsion += 0.3 * (1.0 + np.cos(2.0 * phi_b))

                e_pocket = -2.0 * np.exp(-((a - b) ** 2) / 8.0)
                energy_grid[i, a, b] = e_torsion + e_pocket + base_energy

    best_seq, best_energy = greedy_conformation_search(n_torsions, n_bins, energy_grid)


    best_dihedrals = 2.0 * np.pi * best_seq / n_bins

    return best_seq, best_energy, best_dihedrals
