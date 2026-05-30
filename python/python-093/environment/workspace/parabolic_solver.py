#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg as la
from scipy.fft import fft, ifft, fftfreq
from boundary_conditions import solve_tridiag, BoundaryConditionHandler
from utils import chebyshev_to_monomial_matrix, legendre_to_monomial_matrix


class ParabolicSolver:

    def __init__(self, env, mesh, bc_handler, method='cn_fd'):
        self.env = env
        self.mesh = mesh
        self.bc = bc_handler
        self.method = method
        self.k0 = env.k0
        self.energy_history = []
        self.range_history = []

    def _build_laplacian_fd(self, z, nz):
        dz = np.diff(z)
        dz = np.concatenate([dz, [dz[-1]]])
        a = np.zeros(nz, dtype=np.complex128)
        b = np.zeros(nz, dtype=np.complex128)
        c = np.zeros(nz, dtype=np.complex128)
        for j in range(1, nz - 1):
            dz_j = dz[j]
            dz_jm1 = dz[j - 1]
            denom = dz_j + dz_jm1
            if denom < 1e-12:
                denom = 1e-12
            a[j] = 2.0 / (denom * dz_jm1)
            c[j] = 2.0 / (denom * dz_j)
            b[j] = -a[j] - c[j]

        b[0] = 1.0
        c[0] = 0.0
        a[-1] = 0.0
        b[-1] = 1.0
        return a, b, c

    def _crank_nicolson_step(self, u, z, dr, m):
        nz = len(z)
        a, b, c = self._build_laplacian_fd(z, nz)
        n2_dev = self.env.refractive_index_squared_deviation(z)

        k_pml = self.bc.pml_modified_wavenumber(z)

        n2_eff = (k_pml / self.k0) ** 2 - 1.0










        raise NotImplementedError("HOLE 2: CN discretization missing")


        aL, bL, cL = self.bc.apply_seabed_bc_tridiagonal(aL, bL, cL, u, m)
        aR, bR, cR = self.bc.apply_seabed_bc_tridiagonal(aR, bR, cR, u, m)


        bL[0] = 1.0
        cL[0] = 0.0
        bR[0] = 1.0
        cR[0] = 0.0
        u[0] = 0.0


        d = np.zeros(nz, dtype=np.complex128)
        d[0] = 0.0
        for j in range(1, nz - 1):
            d[j] = aR[j] * u[j - 1] + bR[j] * u[j] + cR[j] * u[j + 1]
        d[-1] = bR[-1] * u[-1]
        if nz > 1:
            d[-1] += aR[-1] * u[-2]

        u_new = solve_tridiag(aL, bL, cL, d)

        u_new[0] = 0.0
        return u_new

    def _ssf_step(self, u, z, dr, m):
        nz = len(z)
        dz = z[1] - z[0]

        n2_dev = self.env.refractive_index_squared_deviation(z)
        u_half = u * np.exp(1j * dr * self.k0 * n2_dev / 2.0)

        kz = 2.0 * np.pi * fftfreq(nz, dz)
        u_fft = fft(u_half)
        u_fft *= np.exp(1j * dr * kz ** 2 / (2.0 * self.k0))
        u_new = ifft(u_fft)

        u_new *= np.exp(1j * dr * self.k0 * n2_dev / 2.0)

        u_new[0] = 0.0

        u_new = self.bc.mask_field_by_bathymetry(u_new, m)
        return u_new

    def compute_energy_flux(self, u, z, m):
        mask = self.mesh.node_mask[m, :]
        z_valid = z[mask]
        u_valid = u[mask]
        if len(z_valid) < 2:
            return 0.0
        intensity = np.abs(u_valid) ** 2
        return np.trapezoid(intensity, z_valid)

    def solve(self, u0, z_s, progress_interval=None):
        nr = self.mesh.nr
        nz = self.mesh.nz
        U = np.zeros((nr, nz), dtype=np.complex128)
        U[0, :] = u0.copy()
        u = u0.copy()
        z = self.mesh.z_grid


        e0 = self.compute_energy_flux(u, z, 0)
        self.energy_history.append(e0)
        self.range_history.append(0.0)

        for m in range(1, nr):
            dr = self.mesh.dr
            if self.method == 'cn_fd':
                u = self._crank_nicolson_step(u, z, dr, m)
            elif self.method == 'ssf':

                dzs = np.diff(z)
                if np.max(np.abs(dzs - dzs[0])) < 1e-6:
                    u = self._ssf_step(u, z, dr, m)
                else:
                    u = self._crank_nicolson_step(u, z, dr, m)
            else:
                raise ValueError(f"Unknown method: {self.method}")


            u = self.bc.mask_field_by_bathymetry(u, m)
            U[m, :] = u.copy()


            e = self.compute_energy_flux(u, z, m)
            self.energy_history.append(e)
            self.range_history.append(self.mesh.r_grid[m])

            if progress_interval and m % progress_interval == 0:
                rel_err = abs(e - e0) / max(abs(e0), 1e-15)
                print(f"  Range {self.mesh.r_grid[m]/1000:.1f} km: "
                      f"energy flux = {e:.6e}, rel. change = {rel_err:.6e}")

        return U

    def energy_conservation_error(self):
        if len(self.energy_history) < 2:
            return 0.0
        e0 = self.energy_history[0]
        if abs(e0) < 1e-20:
            return 0.0
        return max(abs(e - e0) for e in self.energy_history) / abs(e0)


class SpectralElementSolver:

    def __init__(self, env, n_cheb=32):
        self.env = env
        self.n = n_cheb

        self.xi = np.cos(np.pi * (np.arange(n_cheb + 1) + 0.5) / (n_cheb + 1))

        self.T = np.zeros((n_cheb + 1, n_cheb + 1))
        for k in range(n_cheb + 1):
            self.T[:, k] = np.cos(k * np.arccos(self.xi))

        self.M_cheb_mono = chebyshev_to_monomial_matrix(n_cheb)

        self.D = self._chebyshev_differentiation_matrix()

    def _chebyshev_differentiation_matrix(self):
        n = self.n
        x = self.xi
        c = np.ones(n + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c *= ((-1.0) ** np.arange(n + 1))
        X = np.tile(x, (n + 1, 1))
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
        D -= np.diag(np.sum(D, axis=1))
        return D

    def map_to_physical(self, z_min, z_max):
        return 0.5 * (z_max - z_min) * self.xi + 0.5 * (z_max + z_min)

    def solve_eigenproblem(self, z_min, z_max):
        z = self.map_to_physical(z_min, z_max)
        J = 2.0 / (z_max - z_min)

        D2 = J ** 2 * self.D @ self.D

        n2 = self.env.refractive_index(z) ** 2
        A = D2 + np.diag(self.env.k0 ** 2 * n2)

        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0

        eigvals, eigvecs = la.eig(A)

        idx = np.argsort(np.real(eigvals))
        return eigvals[idx], eigvecs[:, idx]
