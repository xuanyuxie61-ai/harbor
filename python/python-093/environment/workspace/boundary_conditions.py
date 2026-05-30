#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mesh_builder import point_in_polygon


class BoundaryConditionHandler:

    def __init__(self, env, mesh):
        self.env = env
        self.mesh = mesh

        self.pml_fraction = 0.15
        self.pml_power = 2.0
        self.pml_sigma_max = 0.5

    def apply_surface_bc(self, u):
        u[0] = 0.0
        return u

    def compute_seabed_admittance(self, theta_grazing, max_iter=20, tol=1e-10):
        k0 = self.env.k0
        n_b = self.env.c0 / self.env.seabed_cp
        cos_theta = np.cos(theta_grazing)

        def g(gamma):
            val = n_b ** 2 - cos_theta ** 2 + (gamma / k0) ** 2
            val = np.maximum(val, 0.0)
            return 1j * k0 * np.sqrt(val)

        gamma = 1j * k0 * np.sqrt(max(n_b ** 2 - cos_theta ** 2, 0.0))
        for it in range(max_iter):
            gamma_new = g(gamma)
            err = abs(gamma_new - gamma)
            if err < tol:
                return gamma_new

            dg = (g(gamma + 1e-8) - g(gamma - 1e-8)) / (2e-8)
            if abs(1 - dg) > 1e-12:
                gamma = gamma - (gamma - gamma_new) / (1.0 - dg)
            else:
                gamma = 0.5 * (gamma + gamma_new)
        return gamma

    def apply_seabed_bc_tridiagonal(self, a, b, c, u, m):
        z_grid = self.mesh.z_grid
        h_b = self.mesh.seafloor_depth[m]

        idx = np.searchsorted(z_grid, h_b, side='right') - 1
        if idx < 1:
            idx = len(z_grid) - 1
        dz = z_grid[idx] - z_grid[idx - 1]
        if dz < 1e-9:
            dz = 1.0

        theta = 0.1
        gamma_b = self.compute_seabed_admittance(theta)






        if idx < len(b):
            a[idx] = -1.0 / dz
            b[idx] = 1.0 / dz + gamma_b
            if idx + 1 < len(c):
                c[idx] = 0.0
        return a, b, c

    def pml_profile(self, z):
        z = np.asarray(z, dtype=np.float64)
        z_pml = self.mesh.z_grid[-1] * (1.0 - self.pml_fraction)
        L_pml = self.mesh.z_grid[-1] - z_pml
        sigma = np.zeros_like(z, dtype=np.float64)
        mask = z > z_pml
        if np.any(mask):
            ratio = (z[mask] - z_pml) / max(L_pml, 1e-9)
            sigma[mask] = self.pml_sigma_max * (ratio ** self.pml_power)
        return sigma

    def pml_modified_wavenumber(self, z):
        z = np.asarray(z, dtype=np.float64)
        sigma = self.pml_profile(z)
        k = self.env.wavenumber(z)
        return k / (1.0 + 1j * sigma)

    def bathymetry_polygon(self, r_extra=0.0):
        r = self.mesh.r_grid
        h = self.mesh.seafloor_depth

        x_poly = list(r) + [r[-1] + r_extra, r[0] - r_extra]
        y_poly = list(h) + [0.0, 0.0]
        return np.asarray(x_poly, dtype=np.float64), np.asarray(y_poly, dtype=np.float64)

    def mask_field_by_bathymetry(self, u, m):
        u = np.asarray(u, dtype=np.complex128)
        mask = self.mesh.node_mask[m, :]
        u_out = u.copy()
        u_out[~mask] = 0.0
        return u_out


def solve_tridiag(a, b, c, d):
    n = len(b)
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)

    cp = np.zeros(n - 1, dtype=np.complex128)
    dp = np.zeros(n, dtype=np.complex128)
    x = np.zeros(n, dtype=np.complex128)


    dp[0] = d[0] / b[0]
    if n > 1:
        cp[0] = c[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-20:
            denom = 1e-20
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom


    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
