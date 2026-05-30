# -*- coding: utf-8 -*-

import numpy as np
from linear_solvers import conjugate_gradient, gmres_restart
from utils import safe_divide, clip_positive

R_EARTH = 6371.0e3
G_GRAV = 9.80665


def temperature_profile_bvp(z_nodes, T_tropopause=220.0, T_stratopause=270.0,
                            z_trop_km=15.0, z_strat_km=50.0):
    z = np.asarray(z_nodes, dtype=float)
    n = z.size
    if n < 3:
        return np.full(n, T_tropopause)

    z_trop = z_trop_km * 1000.0
    z_strat = z_strat_km * 1000.0
    dz_total = z_strat - z_trop


    frac = np.clip((z - z_trop) / dz_total, 0.0, 1.0)
    T_eq = T_tropopause + (T_stratopause - T_tropopause) * (np.sin(0.5 * np.pi * frac) ** 2)


    K_t = 5.0
    alpha_ir = 1.0e-5


    A = np.zeros((n, n))
    rhs = np.zeros(n)

    for i in range(n):
        if i == 0:
            A[i, i] = 1.0
            rhs[i] = T_tropopause
        elif i == n - 1:
            A[i, i] = 1.0
            rhs[i] = T_stratopause
        else:
            dz_p = z[i + 1] - z[i]
            dz_m = z[i] - z[i - 1]
            denom = dz_m * dz_p * (dz_m + dz_p)
            A[i, i - 1] = 2.0 * dz_p / denom
            A[i, i] = -2.0 * (dz_p + dz_m) / denom - alpha_ir / K_t
            A[i, i + 1] = 2.0 * dz_m / denom
            rhs[i] = -alpha_ir * T_eq[i] / K_t


    try:
        x = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    T = np.clip(x, 180.0, 300.0)
    return T


def eddy_diffusivity(z_m, lat_rad):
    z_km = z_m / 1000.0
    K0 = 0.1
    K1 = 1.0
    z_max = 30.0
    sigma_z = 8.0
    Kzz = K0 + K1 * np.exp(-0.5 * ((z_km - z_max) / sigma_z) ** 2)
    Kyy0 = 1.0e5
    Kyy = Kyy0 * (np.cos(lat_rad) ** 2 + 0.1)
    return Kyy, Kzz


def background_wind_field(lon, lat, z_km):
    V0 = 0.5
    W0 = 3.0e-4
    H = 7.0
    v = V0 * np.sin(2.0 * lat) * np.exp(-(z_km - 30.0) / H)
    if 15.0 <= z_km <= 50.0:
        w = W0 * np.sin(np.pi * (z_km - 15.0) / 35.0) * np.cos(2.0 * lat)
    else:
        w = 0.0
    u = 0.0
    return u, v, w


class TransportOperator:

    def __init__(self, mesh):
        from stratospheric_mesh import StratosphericMesh
        if not isinstance(mesh, StratosphericMesh):
            raise TypeError("mesh 必须是 StratosphericMesh 实例")
        self.mesh = mesh
        self._compute_cell_properties()

    def _compute_cell_properties(self):
        self.cell_u = np.zeros(self.mesh.n_cells)
        self.cell_v = np.zeros(self.mesh.n_cells)
        self.cell_w = np.zeros(self.mesh.n_cells)
        self.cell_Kyy = np.zeros(self.mesh.n_cells)
        self.cell_Kzz = np.zeros(self.mesh.n_cells)

        for i in range(self.mesh.n_cells):
            cent = self.mesh.cell_centroids[i]
            lon, lat, z_km = cent[0], cent[1], cent[2]
            u, v, w = background_wind_field(lon, lat, z_km)
            self.cell_u[i] = u
            self.cell_v[i] = v
            self.cell_w[i] = w
            Kyy, Kzz = eddy_diffusivity(z_km * 1000.0, lat)
            self.cell_Kyy[i] = Kyy
            self.cell_Kzz[i] = Kzz

    def transport_source(self, c_cell):


        raise NotImplementedError("Hole 3: 请实现输运算子的源汇项离散计算")

    def apply_matrix(self, x_flat):
        n_cells = self.mesh.n_cells
        n_spec = len(x_flat) // n_cells
        x = x_flat.reshape((n_cells, n_spec))
        S = self.transport_source(x)
        y = x + S
        return y.ravel()
