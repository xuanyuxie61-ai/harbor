
import numpy as np
from combinatorial_stats import binomial_coefficient


class LatticeChannel:
    def __init__(self, shape, binding_energies=None):
        self.shape = shape
        self.n_sites = np.prod(shape)
        if binding_energies is None:

            self.energies = np.zeros(shape)
            if len(shape) == 1:
                mid = shape[0] // 2
                for i in range(shape[0]):
                    self.energies[i] = -1.0e-20 * np.exp(-0.5 * ((i - mid) / 1.5) ** 2)
            else:
                mid_z = shape[1] // 2
                for j in range(shape[1]):
                    self.energies[:, j] = -1.0e-20 * np.exp(-0.5 * ((j - mid_z) / 1.5) ** 2)
        else:
            self.energies = binding_energies

    def valid_configurations(self, n_ions, min_distance=1):
        if len(self.shape) == 1:
            return self._valid_1d(self.n_sites, n_ions, min_distance)
        else:
            return self._valid_2d(self.shape, n_ions, min_distance)

    def _valid_1d(self, n, k, d):
        configs = []

        def backtrack(start, chosen):
            if len(chosen) == k:
                configs.append(np.array(chosen))
                return
            for i in range(start, n):
                if len(chosen) == 0 or i - chosen[-1] >= d:
                    backtrack(i + 1, chosen + [i])

        backtrack(0, [])
        return configs

    def _valid_2d(self, shape, k, d):
        n = np.prod(shape)
        return self._valid_1d(n, k, d)

    def configuration_energy(self, config):
        e_charge = 1.602176634e-19
        eps0 = 8.854187817e-12
        eps_r = 40.0
        coeff = e_charge ** 2 / (4.0 * np.pi * eps0 * eps_r)

        E = 0.0
        for idx in config:
            E += self.energies.flat[idx]

        for i in range(len(config)):
            for j in range(i + 1, len(config)):

                r_ij = abs(config[i] - config[j]) * 0.3e-9
                if r_ij > 0:
                    E += coeff / r_ij
        return E

    def partition_function(self, n_ions, T=300.0, min_distance=1):
        kB = 1.380649e-23
        configs = self.valid_configurations(n_ions, min_distance)
        Z = 0.0
        for conf in configs:
            E = self.configuration_energy(conf)
            Z += np.exp(-E / (kB * T))
        return Z, configs

    def most_probable_configuration(self, n_ions, T=300.0, min_distance=1):
        Z, configs = self.partition_function(n_ions, T, min_distance)
        kB = 1.380649e-23
        probs = []
        for conf in configs:
            E = self.configuration_energy(conf)
            probs.append(np.exp(-E / (kB * T)) / Z)
        idx = int(np.argmax(probs))
        return configs[idx], probs[idx]


def knock_on_energy_barrier(dK_K=0.3e-9, dK_Na=0.25e-9):

    epsilon = 1.0e-21
    sigma_K = 0.28e-9
    sigma_Na = 0.24e-9

    def lj(r, sigma):
        x = sigma / r
        return 4.0 * epsilon * (x ** 12 - x ** 6)

    V_K = lj(dK_K, sigma_K)
    V_Na = lj(dK_Na, sigma_Na)
    return V_K, V_Na
