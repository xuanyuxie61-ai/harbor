
import numpy as np
from monte_carlo_integrator import integrate_sparse_grid, fibonacci_lattice_2d
from special_functions import log_gamma_lanczos


class FreeEnergySurface:
    def __init__(self, temperature=300.0):
        self.kB = 1.380649e-23
        self.T = temperature
        self.beta = 1.0 / (self.kB * temperature)

    def pmf_1d_from_histogram(self, positions, bins=50, range_z=None):
        hist, bin_edges = np.histogram(positions, bins=bins, range=range_z, density=True)
        z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


        hist = np.maximum(hist, 1e-30)
        pmf = -self.kB * self.T * np.log(hist / np.max(hist))
        return z_centers, pmf

    def pmf_2d(self, pos1, pos2, bins=30):
        hist, xedges, yedges = np.histogram2d(pos1, pos2, bins=bins, density=True)
        hist = np.maximum(hist, 1e-30)
        pmf = -self.kB * self.T * np.log(hist / np.max(hist))
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        return xcenters, ycenters, pmf

    def barrier_height(self, z, pmf):
        g_min = np.min(pmf)
        g_max = np.max(pmf)
        return g_max - g_min

    def integration_factor(self, z, pmf):
        dz = z[1] - z[0]
        integral = np.sum(np.exp(-self.beta * pmf)) * dz
        return integral


def partition_function_integral(potential_func, dim, level_max=3):
    kB = 1.380649e-23
    T = 300.0
    beta = 1.0 / (kB * T)

    def integrand(x):
        return np.exp(-beta * potential_func(x))

    result = integrate_sparse_grid(integrand, dim, level_max)
    return result


def sphere_solvation_free_energy(charge, radius, epsilon=78.5, T=300.0):
    e_charge = 1.602176634e-19
    eps0 = 8.854187817e-12
    kB = 1.380649e-23

    delta_G = -(1.0 - 1.0 / epsilon) * (charge ** 2 * e_charge ** 2) / (8.0 * np.pi * eps0 * radius)
    return delta_G


def debye_huckel_excess_energy(ionic_strength, charge, radius, T=300.0):
    NA = 6.02214076e23
    e_charge = 1.602176634e-19
    eps0 = 8.854187817e-12
    eps_r = 78.5
    kB = 1.380649e-23

    kappa = np.sqrt(2000.0 * NA * e_charge ** 2 * ionic_strength / (eps0 * eps_r * kB * T))
    numerator = NA * e_charge ** 2 * kappa * charge ** 2
    denominator = 8.0 * np.pi * eps0 * eps_r * (1.0 + kappa * radius)
    return -numerator / denominator


def selective_permeability_ratio(dG_k, dG_na, D_k=1.96e-9, D_na=1.33e-9, T=300.0):
    kB = 1.380649e-23
    ratio = (D_k / D_na) * np.exp(-(dG_k - dG_na) / (kB * T))
    return ratio
