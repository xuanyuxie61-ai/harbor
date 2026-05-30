
import numpy as np
from typing import Tuple, Optional, Callable


def effective_transverse_coupling(dtau: float, gamma: float) -> float:



    raise NotImplementedError("Hole 1: 请补全 Trotter-Suzuki 有效横向耦合公式")


class PathIntegralMonteCarlo:

    def __init__(self, n_spins: int, beta: float, n_slices: int,
                 energy_func: Callable, gamma_schedule: np.ndarray,
                 seed: int = 154):
        if n_spins <= 0 or beta <= 0 or n_slices <= 0:
            raise ValueError("Physical parameters must be positive")
        if gamma_schedule.size != n_slices:
            raise ValueError("gamma_schedule length must equal n_slices")
        self.n_spins = n_spins
        self.beta = float(beta)
        self.n_slices = n_slices
        self.energy_func = energy_func
        self.gamma_schedule = np.array(gamma_schedule, dtype=float)
        self.dtau = beta / n_slices
        self.rng = np.random.default_rng(seed)

        base = 2 * self.rng.integers(0, 2, size=n_spins) - 1
        self.worldlines = np.tile(base, (n_slices, 1)).astype(int)

        self.energies = np.array([energy_func(self.worldlines[m, :])
                                   for m in range(n_slices)])

    def _slice_energy_change(self, m: int, i: int) -> float:
        s_old = self.worldlines[m, :].copy()
        s_new = s_old.copy()
        s_new[i] *= -1
        e_old = self.energy_func(s_old)
        e_new = self.energy_func(s_new)
        return float(e_new - e_old)

    def _worldline_coupling_change(self, m: int, i: int) -> float:
        s_m = int(self.worldlines[m, i])
        s_prev = int(self.worldlines[(m - 1) % self.n_slices, i])
        s_next = int(self.worldlines[(m + 1) % self.n_slices, i])
        gamma_m = self.gamma_schedule[m]
        J_perp = effective_transverse_coupling(self.dtau, gamma_m)

        delta_coupling = -J_perp * ((-s_m) - s_m) * (s_prev + s_next)
        return float(delta_coupling)

    def _metropolis_step_single(self) -> int:
        accepted = 0
        for m in range(self.n_slices):
            for i in range(self.n_spins):
                delta_e = self._slice_energy_change(m, i)
                delta_w = self._worldline_coupling_change(m, i)
                delta_total = self.dtau * delta_e + delta_w
                if delta_total <= 0:
                    accept = True
                else:
                    prob = np.exp(-delta_total)
                    prob = min(prob, 1.0)
                    accept = self.rng.random() < prob
                if accept:
                    self.worldlines[m, i] *= -1
                    self.energies[m] += delta_e
                    accepted += 1
        return accepted

    def _cluster_update(self) -> int:
        accepted = 0
        for i in range(self.n_spins):

            bonds = np.zeros(self.n_slices, dtype=int)
            for m in range(self.n_slices):
                gamma_m = self.gamma_schedule[m]
                J_perp = effective_transverse_coupling(self.dtau, gamma_m)
                p_bond = 1.0 - np.exp(-2.0 * J_perp)
                if self.rng.random() < p_bond:
                    bonds[m] = 1

            visited = np.zeros(self.n_slices, dtype=int)
            for m0 in range(self.n_slices):
                if visited[m0]:
                    continue
                cluster = []
                m = m0
                while True:
                    cluster.append(m)
                    visited[m] = 1
                    next_m = (m + 1) % self.n_slices
                    if bonds[m] and not visited[next_m]:
                        m = next_m
                    else:
                        break

                e_flip = 0.0
                for m in cluster:
                    s_old = self.worldlines[m, :].copy()
                    s_new = s_old.copy()
                    s_new[i] *= -1
                    e_flip += self.energy_func(s_new) - self.energy_func(s_old)


                if e_flip <= 0 or self.rng.random() < np.exp(-self.dtau * e_flip):
                    for m in cluster:
                        self.worldlines[m, i] *= -1

                    for m in cluster:
                        self.energies[m] = self.energy_func(self.worldlines[m, :])
                    accepted += len(cluster)
        return accepted

    def thermalize(self, n_sweeps: int = 500) -> None:
        for _ in range(n_sweeps):
            self._metropolis_step_single()
            if _ % 10 == 0:
                self._cluster_update()

    def measure_observables(self, n_measurements: int = 100,
                            sampling_interval: int = 5) -> dict:
        e_vals = []
        m_vals = []
        m2_vals = []
        winding = []
        for k in range(n_measurements):
            for _ in range(sampling_interval):
                self._metropolis_step_single()
                if _ % 5 == 0:
                    self._cluster_update()
            e_avg = self.energies.mean()
            mag = self.worldlines.mean(axis=0).mean()
            m2 = (self.worldlines.mean(axis=0) ** 2).mean()

            wind = 0.0
            for i in range(self.n_spins):
                flips = np.sum(np.abs(np.diff(self.worldlines[:, i]))) // 2
                wind += flips
            e_vals.append(e_avg)
            m_vals.append(mag)
            m2_vals.append(m2)
            winding.append(wind / self.n_spins)
        e_vals = np.array(e_vals)
        m_vals = np.array(m_vals)
        m2_vals = np.array(m2_vals)
        winding = np.array(winding)
        chi = self.beta * (m2_vals.mean() - m_vals.mean() ** 2)
        return {
            "energy_mean": float(e_vals.mean()),
            "energy_std": float(e_vals.std(ddof=1)),
            "magnetization": float(m_vals.mean()),
            "magnetization_std": float(m_vals.std(ddof=1)),
            "susceptibility": float(chi),
            "winding_number": float(winding.mean()),
        }

    def estimate_ground_state_energy(self, n_replicas: int = 3,
                                      n_sweeps_each: int = 300) -> float:
        best_e = float('inf')
        for rep in range(n_replicas):

            base = 2 * self.rng.integers(0, 2, size=self.n_spins) - 1
            self.worldlines = np.tile(base, (self.n_slices, 1))
            self.thermalize(n_sweeps=n_sweeps_each)
            e_mean = self.energies.mean()
            if e_mean < best_e:
                best_e = e_mean
        return float(best_e)
