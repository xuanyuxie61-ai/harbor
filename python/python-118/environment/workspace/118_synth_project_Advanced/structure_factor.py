
import numpy as np
from scipy.special import sph_harm
from utils_numeric import check_bounds, safe_sqrt


class LocalStructureAnalyzer:

    def __init__(self, l=6, q_threshold=0.40, r_bond_factor=1.35):
        self.l = int(l)
        self.q_threshold = float(q_threshold)
        self.r_bond_factor = float(r_bond_factor)

    def compute_qlm(self, positions, box, i_atom, neighbors_rij):
        n_bonds = len(neighbors_rij)
        if n_bonds == 0:
            return np.zeros(2 * self.l + 1, dtype=np.complex128)

        qlm = np.zeros(2 * self.l + 1, dtype=np.complex128)
        for _, r, rij in neighbors_rij:

            x, y, z = rij
            theta = np.arccos(check_bounds(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
            for m in range(-self.l, self.l + 1):
                idx = m + self.l
                qlm[idx] += sph_harm(m, self.l, phi, theta)
        qlm /= n_bonds
        return qlm

    def compute_ql(self, qlm):
        norm = np.sum(np.abs(qlm) ** 2)
        return safe_sqrt(4.0 * np.pi / (2.0 * self.l + 1.0) * norm)

    def analyze_system(self, positions, box, neighbor_list):
        n_atoms = positions.shape[0]
        q_values = np.zeros(n_atoms, dtype=np.float64)
        qlm_array = np.zeros((n_atoms, 2 * self.l + 1), dtype=np.complex128)

        for i in range(n_atoms):
            qlm = self.compute_qlm(positions, box, i, neighbor_list[i])
            qlm_array[i] = qlm
            q_values[i] = self.compute_ql(qlm)

        is_solid = q_values > self.q_threshold
        return q_values, is_solid, qlm_array

    def compute_coherent_ql(self, qlm_array):
        avg_qlm = np.mean(qlm_array, axis=0)
        return safe_sqrt(4.0 * np.pi / (2.0 * self.l + 1.0) * np.sum(np.abs(avg_qlm) ** 2))

    def compute_wl_invariant(self, qlm_array):

        n_atoms = qlm_array.shape[0]
        wl_sum = 0.0



        for i in range(n_atoms):
            q = qlm_array[i]
            norm = np.sum(np.abs(q) ** 2)
            if norm < 1e-12:
                continue

            cumulant = 0.0
            for m1 in range(-self.l, self.l + 1):
                for m2 in range(-self.l, self.l + 1):
                    m3 = -(m1 + m2)
                    if abs(m3) > self.l:
                        continue


                    coeff = 1.0 if (m1 + m2 + m3 == 0) else 0.0
                    cumulant += coeff * q[m1 + self.l] * q[m2 + self.l] * q[m3 + self.l]
            wl_sum += cumulant.real / (norm ** 1.5 + 1e-12)
        return wl_sum / n_atoms


def build_neighbor_list_with_cutoff(positions, box, r_cut):
    n_atoms = positions.shape[0]
    r_cut2 = r_cut ** 2
    neighbors = []
    for i in range(n_atoms):
        neigh = []
        for j in range(n_atoms):
            if i == j:
                continue
            rij = positions[j] - positions[i]
            rij -= box * np.round(rij / box)
            r2 = np.dot(rij, rij)
            if r2 < r_cut2 and r2 > 1e-12:
                r = safe_sqrt(r2)
                neigh.append((j, r, rij))
        neighbors.append(neigh)
    return neighbors


def radial_distribution_function(positions, box, species_idx, dr=0.05, r_max=None):
    n_atoms = positions.shape[0]
    if r_max is None:
        r_max = 0.5 * np.min(box)
    n_bins = int(r_max / dr)
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])


    hist = np.zeros(n_bins, dtype=np.float64)
    volume = np.prod(box)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[j] - positions[i]
            rij -= box * np.round(rij / box)
            r = safe_sqrt(np.dot(rij, rij))
            if r < r_max:
                idx = int(r / dr)
                if idx < n_bins:
                    hist[idx] += 2.0


    rho = n_atoms / volume
    for idx in range(n_bins):
        r_inner = r_bins[idx]
        r_outer = r_bins[idx + 1]
        shell_vol = 4.0 / 3.0 * np.pi * (r_outer ** 3 - r_inner ** 3)
        ideal = rho * shell_vol * n_atoms
        if ideal > 0:
            hist[idx] /= ideal

    return r_centers, hist
