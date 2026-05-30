
import numpy as np
from typing import Tuple, List
from icf_parameters import PC, FP, NP, TP
from utils import clamp_array






class NiederreiterSequence:

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0

        self._init_direction_numbers()

    def _init_direction_numbers(self):
        max_bits = 31
        self.directions = np.zeros((self.dim, max_bits), dtype=np.uint32)

        for d in range(self.dim):

            for j in range(max_bits):
                self.directions[d, j] = np.uint32(1) << (max_bits - j - 1)

    def next(self) -> np.ndarray:
        self.count += 1
        max_bits = 31
        recip = 2.0**(-max_bits)


        gray = self.count ^ (self.count >> 1)
        quasi = np.zeros(self.dim)

        for d in range(self.dim):
            x = np.uint32(0)
            g = gray
            bit = 0
            while g > 0 and bit < max_bits:
                if g & 1:
                    x ^= self.directions[d, bit]
                g >>= 1
                bit += 1
            quasi[d] = float(x) * recip

        return np.clip(quasi, 0.0, 1.0)






def dt_reactivity(T_i_kev: float) -> float:
    return FP.reactivity_dt(T_i_kev)


def compute_fusion_rate_density(n_d: np.ndarray, n_t: np.ndarray,
                                T_i: np.ndarray) -> np.ndarray:
    n_cells = len(n_d)
    rate = np.zeros(n_cells)
    for i in range(n_cells):
        T_kev = T_i[i] * PC.BOLTZMANN / (1.0e3 * PC.ELEMENTARY_CHARGE)
        T_kev = max(T_kev, 0.1)
        sv = dt_reactivity(T_kev)
        rate[i] = n_d[i] * n_t[i] * sv
    return rate


def alpha_deposition_local(rate_density: np.ndarray, cell_volume: np.ndarray) -> np.ndarray:
    return rate_density * FP.Q_ALPHA






def build_energy_relaxation_matrix(dt: float, tau_eq: float) -> np.ndarray:
    p = min(dt / max(tau_eq, 1.0e-30), 0.5)
    A = np.array([
        [1.0 - p, p],
        [p, 1.0 - p]
    ])
    return A


def spitzer_equilibration_time(n_e: float, T_e: float, Z_eff: float,
                               A_ion: float) -> float:
    if n_e <= 0.0 or T_e <= 0.0 or Z_eff <= 0.0:
        return 1.0e30
    ln_lambda = max(23.5 - np.log(np.sqrt(n_e) / max(T_e, 1.0)), 2.0)
    tau = 3.0e-18 * A_ion * T_e**1.5 / (Z_eff**2 * n_e * ln_lambda)
    return max(tau, 1.0e-30)


def apply_energy_relaxation(E_ion: float, E_e: float, dt: float,
                            n_e: float, T_e: float, Z_eff: float,
                            A_ion: float) -> Tuple[float, float]:
    tau_eq = spitzer_equilibration_time(n_e, T_e, Z_eff, A_ion)
    A = build_energy_relaxation_matrix(dt, tau_eq)
    pop = np.array([E_ion, E_e])
    pop_new = A @ pop
    return float(pop_new[0]), float(pop_new[1])






class NeutronMC:

    def __init__(self, n_samples: int = NP.MC_NEUTRON_SAMPLES):
        self.n_samples = n_samples
        self.sequence = NiederreiterSequence(dim=6)
        self.energies = []
        self.escaped = []

    def sample_isotropic_direction(self) -> np.ndarray:
        q = self.sequence.next()
        mu = 2.0 * q[0] - 1.0
        phi = 2.0 * np.pi * q[1]
        sin_theta = np.sqrt(max(1.0 - mu**2, 0.0))
        return np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            mu
        ])

    def neutron_mean_free_path(self, rho: float, A_avg: float = 2.5) -> float:
        n = rho * PC.AVOGADRO / (A_avg * 1.0e-3)
        sigma = 3.5e-28
        return 1.0 / max(n * sigma, 1.0e-30)

    def transport_batch(self, r_cells: np.ndarray, rho_cells: np.ndarray,
                        source_positions: np.ndarray, source_weights: np.ndarray) -> dict:
        n_cells = len(r_cells)
        cell_flux = np.zeros(n_cells)
        escaped_energy = 0.0
        deposited_energy = 0.0

        n_sources = len(source_positions)
        samples_per_source = self.n_samples // max(n_sources, 1)

        for src_idx in range(n_sources):
            r_src = source_positions[src_idx]
            weight = source_weights[src_idx]

            for _ in range(samples_per_source):

                pos = np.array([r_src, 0.0, 0.0])
                direction = self.sample_isotropic_direction()
                energy = FP.Q_NEUTRON


                q = self.sequence.next()
                mfp = self.neutron_mean_free_path(
                    np.interp(r_src, r_cells, rho_cells))
                step = -mfp * np.log(max(q[2], 1.0e-30))

                new_pos = pos + step * direction
                r_new = np.sqrt(np.sum(new_pos**2))


                if r_new > r_cells[-1] + 1.0e-6:
                    escaped_energy += energy * weight / samples_per_source
                    self.escaped.append(energy)
                else:

                    cell_idx = np.searchsorted(r_cells, r_new)
                    cell_idx = min(max(cell_idx, 0), n_cells - 1)
                    cell_flux[cell_idx] += weight / samples_per_source
                    deposited_energy += energy * weight / samples_per_source
                    self.energies.append(energy)

        return {
            "cell_flux": cell_flux,
            "escaped_energy": escaped_energy,
            "deposited_energy": deposited_energy,
            "total_samples": samples_per_source * n_sources,
        }






def histogramize_spectrum(energies: List[float], n_bins: int = 20,
                          e_min: float = 13.0e6 * PC.ELEMENTARY_CHARGE,
                          e_max: float = 15.0e6 * PC.ELEMENTARY_CHARGE) -> Tuple[np.ndarray, np.ndarray]:
    if not energies:
        return np.zeros(n_bins), np.zeros(n_bins)

    bin_edges = np.linspace(e_min, e_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(n_bins)

    for e in energies:
        idx = int(n_bins * (e - e_min) / (e_max - e_min))
        if 0 <= idx < n_bins:
            counts[idx] += 1
        elif idx == n_bins and e < bin_edges[-1] + 1.0e-15 * (e_max - e_min):
            counts[-1] += 1

    return bin_centers, counts
