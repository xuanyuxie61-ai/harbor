# -*- coding: utf-8 -*-

import numpy as np
from constants import (
    LDA_VOLUME, LDA_SURFACE, LDA_COULOMB, LDA_ASYMMETRY, LDA_PAIRING,
    MASS_PROTON, MASS_NEUTRON
)


def liquid_drop_binding_energy(Z, N):
    A = Z + N
    if np.any(A == 0):
        return 0.0
    delta_pair = 0.0

    if np.isscalar(Z):
        if Z % 2 == 0 and N % 2 == 0:
            delta_pair = LDA_PAIRING / np.sqrt(A)
        elif Z % 2 == 1 and N % 2 == 1:
            delta_pair = -LDA_PAIRING / np.sqrt(A)
    else:
        Z = np.asarray(Z)
        N = np.asarray(N)
        A_arr = np.asarray(A)
        delta_pair = np.zeros_like(A_arr, dtype=float)
        ee = (Z % 2 == 0) & (N % 2 == 0)
        oo = (Z % 2 == 1) & (N % 2 == 1)
        delta_pair[ee] = LDA_PAIRING / np.sqrt(A_arr[ee])
        delta_pair[oo] = -LDA_PAIRING / np.sqrt(A_arr[oo])

    B = (LDA_VOLUME * A
         - LDA_SURFACE * (A ** (2.0 / 3.0))
         - LDA_COULOMB * (Z ** 2) / (A ** (1.0 / 3.0))
         - LDA_ASYMMETRY * ((N - Z) ** 2) / A
         + delta_pair)
    return B


def atomic_mass_ldm(Z, N):
    A = Z + N
    return Z * MASS_PROTON + N * MASS_NEUTRON - liquid_drop_binding_energy(Z, N)


def shell_correction_from_spectrum(energies, occupancy, lambda_):
    energies = np.asarray(energies)
    occupancy = np.asarray(occupancy)
    E_sp = np.sum(energies * occupancy)

    sigma = 7.0
    weights = np.exp(-0.5 * ((energies - lambda_) / sigma) ** 2)
    weights /= np.sum(weights)
    E_smooth = np.sum(energies * weights) * np.sum(occupancy)
    return E_sp - E_smooth


class NuclearMassSurface:

    def __init__(self, data_N, data_Z, data_mass):
        self.data_N = np.asarray(data_N, dtype=float)
        self.data_Z = np.asarray(data_Z, dtype=float)
        self.data_mass = np.asarray(data_mass, dtype=float)
        self.n_data = self.data_N.size

        ldm_masses = np.array([atomic_mass_ldm(int(z), int(n))
                               for z, n in zip(self.data_Z, self.data_N)])
        self.residuals = self.data_mass - ldm_masses

        self._build_rbf()

    def _rbf(self, r):
        r = np.where(r < 1e-10, 1e-10, r)
        return (r ** 2) * np.log(r)

    def _build_rbf(self):
        Phi = np.zeros((self.n_data, self.n_data))
        for i in range(self.n_data):
            for j in range(self.n_data):
                dx = self.data_N[i] - self.data_N[j]
                dy = self.data_Z[i] - self.data_Z[j]
                Phi[i, j] = self._rbf(np.sqrt(dx * dx + dy * dy))

        P = np.vstack([np.ones(self.n_data), self.data_N, self.data_Z]).T

        A = np.zeros((self.n_data + 3, self.n_data + 3))
        A[:self.n_data, :self.n_data] = Phi
        A[:self.n_data, self.n_data:] = P
        A[self.n_data:, :self.n_data] = P.T
        rhs = np.zeros(self.n_data + 3)
        rhs[:self.n_data] = self.residuals
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.c = sol[:self.n_data]
        self.p = sol[self.n_data:]

    def evaluate(self, N, Z):
        scalar = np.isscalar(N)
        N = np.atleast_1d(N)
        Z = np.atleast_1d(Z)
        ldm = np.array([atomic_mass_ldm(int(zz), int(nn))
                        for zz, nn in zip(Z.ravel(), N.ravel())])
        ldm = ldm.reshape(N.shape)

        corr = np.zeros_like(N, dtype=float)
        for i in range(self.n_data):
            dx = N - self.data_N[i]
            dy = Z - self.data_Z[i]
            r = np.sqrt(dx * dx + dy * dy)
            corr += self.c[i] * self._rbf(r)

        corr += self.p[0] + self.p[1] * N + self.p[2] * Z
        mass = ldm + corr
        return float(mass) if scalar else mass

    def separation_energy(self, N, Z, nucleon='neutron'):
        if nucleon == 'neutron':
            if N <= 0:
                return np.inf
            return self.evaluate(N - 1, Z) - self.evaluate(N, Z) + MASS_NEUTRON
        elif nucleon == 'proton':
            if Z <= 0:
                return np.inf
            return self.evaluate(N, Z - 1) - self.evaluate(N, Z) + MASS_PROTON
        else:
            raise ValueError("nucleon must be 'neutron' or 'proton'.")

    def dripline_location(self, Z, direction='neutron'):
        search_range = range(1, 3 * Z + 20)
        for N in search_range:
            S = self.separation_energy(N, Z, direction)
            if S < 0:
                return max(N - 1, 0)
        return search_range[-1]


def genz_test_function_oscillatory(m, c, w, x):
    x = np.asarray(x)
    c = np.asarray(c)
    return np.cos(2.0 * np.pi * w + np.dot(x, c))


def mass_surface_curvature(mass_surface, N0, Z0, h=1.0):
    M00 = mass_surface.evaluate(N0, Z0)
    M_p0 = mass_surface.evaluate(N0 + h, Z0)
    M_m0 = mass_surface.evaluate(N0 - h, Z0)
    M_0p = mass_surface.evaluate(N0, Z0 + h)
    M_0m = mass_surface.evaluate(N0, Z0 - h)
    M_pp = mass_surface.evaluate(N0 + h, Z0 + h)
    M_mm = mass_surface.evaluate(N0 - h, Z0 - h)
    M_pm = mass_surface.evaluate(N0 + h, Z0 - h)
    M_mp = mass_surface.evaluate(N0 - h, Z0 + h)

    d2N = (M_p0 - 2 * M00 + M_m0) / (h * h)
    d2Z = (M_0p - 2 * M00 + M_0m) / (h * h)
    dNdZ = (M_pp - M_pm - M_mp + M_mm) / (4 * h * h)
    return d2N + d2Z - 2 * dNdZ
