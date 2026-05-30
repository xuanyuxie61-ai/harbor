#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from special_functions import alnorm






class VolumeScatteringModel:

    def __init__(self, sigma0=1e-6, z0=50.0, alpha=0.3, Lambda=100.0,
                 particle_radius=0.01, gamma_kappa=0.1, gamma_rho=0.05):
        self.sigma0 = sigma0
        self.z0 = z0
        self.alpha = alpha
        self.Lambda = Lambda
        self.particle_radius = particle_radius
        self.gamma_kappa = gamma_kappa
        self.gamma_rho = gamma_rho

    def scattering_strength_linear(self, z):
        z = np.asarray(z, dtype=np.float64)
        base = self.sigma0 * np.exp(-z / self.z0)
        modulation = 1.0 + self.alpha * np.sin(2.0 * np.pi * z / self.Lambda)
        return base * modulation

    def scattering_strength_db(self, z):
        sigma = self.scattering_strength_linear(z)
        sigma = np.maximum(sigma, 1e-20)
        return 10.0 * np.log10(sigma)

    def rayleigh_scattering_amplitude(self, k, theta):
        theta = np.asarray(theta, dtype=np.float64)
        return (k ** 2) * (self.particle_radius ** 3) * \
               (self.gamma_kappa - self.gamma_rho * np.cos(theta))

    def bistatic_cross_section(self, k, theta_i, theta_s):
        f = self.rayleigh_scattering_amplitude(k, theta_i)

        return np.abs(f) ** 2






class ReverberationModel:

    def __init__(self, scattering_model, c_water=1500.0):
        self.scat = scattering_model
        self.c = c_water

    def scattering_volume(self, R, tau_pulse, beamwidth_az=10.0, beamwidth_el=10.0):
        domega = np.radians(beamwidth_az) * np.radians(beamwidth_el)
        return (self.c * tau_pulse / 2.0) * (R ** 2) * domega

    def reverberation_level(self, R, tau_pulse, SL_db, TL_db,
                            beamwidth_az=10.0, beamwidth_el=10.0,
                            z_scatter=100.0):
        V = self.scattering_volume(R, tau_pulse, beamwidth_az, beamwidth_el)
        Sv_db = self.scat.scattering_strength_db(z_scatter)
        RL = SL_db - 2.0 * TL_db + Sv_db + 10.0 * np.log10(max(V, 1e-20))
        return RL






class BoxDistanceStatistics:

    def __init__(self, a, b, c, seed=None):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.rng = np.random.default_rng(seed)

    def sample_points(self, n):
        x = self.rng.random((n, 3)) * [self.a, self.b, self.c]
        return x

    def mean_distance_monte_carlo(self, n_samples=50000):
        X = self.sample_points(n_samples)
        Y = self.sample_points(n_samples)
        D = np.linalg.norm(X - Y, axis=1)
        return float(np.mean(D)), float(np.std(D))

    def mean_distance_exact(self):
        a, b, c = self.a, self.b, self.c


        if abs(a - b) < 1e-9 and abs(b - c) < 1e-9:
            return 0.661707182 * a

        return self.mean_distance_monte_carlo()[0]

    def distance_pdf_histogram(self, n_samples=50000, bins=100):
        X = self.sample_points(n_samples)
        Y = self.sample_points(n_samples)
        D = np.linalg.norm(X - Y, axis=1)
        hist, edges = np.histogram(D, bins=bins, density=True)
        return hist, edges






from source_field import hammersley_sequence


def qmc_scattering_integral(integrand_func, bounds, n_samples=4096):
    dim = len(bounds)
    seq = hammersley_sequence(0, n_samples, dim, n=n_samples)

    for j, (low, high) in enumerate(bounds):
        seq[:, j] = low + seq[:, j] * (high - low)
    values = np.array([integrand_func(seq[i, :]) for i in range(n_samples)])
    volume = 1.0
    for low, high in bounds:
        volume *= (high - low)
    return volume * np.mean(values)






class SpatialCorrelation:

    def __init__(self, U, r_grid, z_grid):
        self.U = np.asarray(U, dtype=np.complex128)
        self.r_grid = np.asarray(r_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)

    def correlation_1d(self, axis='r', lag_index=1):
        if axis == 'r':

            nr, nz = self.U.shape
            C = np.zeros(nz, dtype=np.float64)
            for j in range(nz):
                u1 = self.U[:-lag_index, j]
                u2 = self.U[lag_index:, j]
                num = np.mean(u1 * np.conj(u2))
                den = np.sqrt(np.mean(np.abs(u1) ** 2) * np.mean(np.abs(u2) ** 2))
                if abs(den) > 1e-20:
                    C[j] = np.abs(num) / den
            return C
        else:
            nr, nz = self.U.shape
            C = np.zeros(nr, dtype=np.float64)
            for i in range(nr):
                if nz <= lag_index:
                    continue
                u1 = self.U[i, :-lag_index]
                u2 = self.U[i, lag_index:]
                num = np.mean(u1 * np.conj(u2))
                den = np.sqrt(np.mean(np.abs(u1) ** 2) * np.mean(np.abs(u2) ** 2))
                if abs(den) > 1e-20:
                    C[i] = np.abs(num) / den
            return C

    def mean_correlation_length(self, axis='r'):

        if axis == 'r':
            C = self.correlation_1d('r', lag_index=1)

            if len(C) > 0 and np.mean(C) > 0:
                return -1.0 / np.log(max(np.mean(C), 1e-6))
        return 0.0
