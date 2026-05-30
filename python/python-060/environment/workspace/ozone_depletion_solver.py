# -*- coding: utf-8 -*-

import numpy as np
from chemistry_kinetics import StratosphericChemistry
from transport_operator import TransportOperator, temperature_profile_bvp
from photolysis_rates import PhotolysisRateCalculator
from linear_solvers import gmres_restart, conjugate_gradient
from utils import clip_positive, safe_divide


NA = 6.02214076e23


class OzoneDepletionSolver:

    def __init__(self, mesh, solar_zenith_deg=60.0):
        from stratospheric_mesh import StratosphericMesh
        self.mesh = mesh
        self.solar_zenith = solar_zenith_deg
        self.n_cells = mesh.n_cells
        self.photo = PhotolysisRateCalculator(degree=7)
        self.transport = TransportOperator(mesh)
        self._initialize_temperature()
        self._initialize_concentrations()
        self._compute_photolysis()

    def _initialize_temperature(self):
        z_km_unique = np.linspace(self.mesh.alt_range[0], self.mesh.alt_range[1],
                                   self.mesh.n_alt)
        z_m = z_km_unique * 1000.0
        T_prof = temperature_profile_bvp(z_m)
        self.cell_temperature = np.zeros(self.n_cells)
        self.cell_altitude = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            z_km = self.mesh.cell_centroids[i, 2]
            self.cell_altitude[i] = z_km

            idx = np.searchsorted(z_km_unique, z_km)
            idx = np.clip(idx, 1, len(z_km_unique) - 1)
            z0, z1 = z_km_unique[idx - 1], z_km_unique[idx]
            T0, T1 = T_prof[idx - 1], T_prof[idx]
            if abs(z1 - z0) > 1e-10:
                frac = (z_km - z0) / (z1 - z0)
            else:
                frac = 0.0
            self.cell_temperature[i] = T0 + frac * (T1 - T0)

    def _initialize_concentrations(self):
        nsp = StratosphericChemistry.N_SPECIES
        self.c_cell = np.zeros((self.n_cells, nsp))
        for i in range(self.n_cells):
            z_km = self.cell_altitude[i]
            T = self.cell_temperature[i]


            n_dens = self._number_density(z_km, T)
            self.c_cell[i, StratosphericChemistry.IDX_O2] = 0.21 * n_dens
            self.c_cell[i, StratosphericChemistry.IDX_N2] = 0.78 * n_dens
            self.c_cell[i, StratosphericChemistry.IDX_M] = n_dens

            n_o3 = 5.0e12 * np.exp(-0.5 * ((z_km - 25.0) / 5.0) ** 2)
            self.c_cell[i, StratosphericChemistry.IDX_O3] = n_o3

            ppbv = 1e-9 * n_dens
            self.c_cell[i, StratosphericChemistry.IDX_NO] = 1.0 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_NO2] = 0.5 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_Cl] = 0.05 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_ClO] = 0.03 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_OH] = 0.01 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_HO2] = 0.02 * ppbv

            self.c_cell[i, StratosphericChemistry.IDX_O] = 1e6
            self.c_cell[i, StratosphericChemistry.IDX_O1D] = 1e2

    def _number_density(self, z_km, T):
        n0 = 2.55e19
        H = 7.0
        return n0 * np.exp(-z_km / H) * (288.0 / T)

    def _compute_photolysis(self):
        self.J_o2_cell = np.zeros(self.n_cells)
        self.J_o3_cell = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            z_km = self.cell_altitude[i]
            T = self.cell_temperature[i]

            col_o3 = self._column_density(i, StratosphericChemistry.IDX_O3)
            col_o2 = self._column_density(i, StratosphericChemistry.IDX_O2)
            self.J_o2_cell[i] = self.photo.photolysis_rate_o2(
                z_km, self.solar_zenith, col_o3, col_o2
            )
            self.J_o3_cell[i] = self.photo.photolysis_rate_o3(
                z_km, self.solar_zenith, col_o3, col_o2, T
            )

    def _column_density(self, cell_idx, species_idx):
        z_km = self.cell_altitude[cell_idx]
        mask = self.cell_altitude >= z_km
        if not np.any(mask):
            return 0.0

        conc = self.c_cell[mask, species_idx]
        dz = (self.mesh.alt_range[1] - self.mesh.alt_range[0]) / (self.mesh.n_alt - 1)
        return float(np.mean(conc) * dz * 1e5)

    def _chemistry_step(self, c_in, dt, cell_idx):

        raise NotImplementedError("Hole 2: 请实现单个单元的化学步调用逻辑")

    def _transport_step_implicit(self, c_in, dt):
        nsp = StratosphericChemistry.N_SPECIES
        x0 = c_in.ravel().copy()

        def Ax(x):
            x_reshaped = x.reshape((self.n_cells, nsp))
            S = self.transport.transport_source(x_reshaped)
            return (x_reshaped - dt * S).ravel()

        b = c_in.ravel()
        x, res, it = gmres_restart(Ax, b, x0=x0, max_iter=20, restart=15,
                                    tol_abs=1e-8, tol_rel=1e-6)
        c_out = x.reshape((self.n_cells, nsp))
        c_out = np.maximum(c_out, 1e-30)
        return c_out

    def step(self, dt):

        c_half = np.zeros_like(self.c_cell)
        for i in range(self.n_cells):
            c_half[i] = self._chemistry_step(self.c_cell[i], dt * 0.5, i)


        c_star = self._transport_step_implicit(c_half, dt)


        c_new = np.zeros_like(self.c_cell)
        for i in range(self.n_cells):
            c_new[i] = self._chemistry_step(c_star[i], dt * 0.5, i)

        self.c_cell = c_new
        return c_new

    def integrate(self, t_total_hours=24.0, dt_max_minutes=10.0):
        t_total = t_total_hours * 3600.0
        dt_max = dt_max_minutes * 60.0
        t = 0.0
        history = {
            'time_hours': [0.0],
            'total_ozone_du': [self._total_ozone_dobson()],
            'o3_min': [np.min(self.c_cell[:, StratosphericChemistry.IDX_O3])],
            'o3_max': [np.max(self.c_cell[:, StratosphericChemistry.IDX_O3])],
        }

        while t < t_total:

            L_max = 0.0
            for i in range(self.n_cells):
                chem = StratosphericChemistry(T_k=self.cell_temperature[i],
                                               M_cm3=self.c_cell[i, StratosphericChemistry.IDX_M])
                chem.set_photolysis_rates(self.J_o2_cell[i], self.J_o3_cell[i])
                _, L = chem.production_loss(self.c_cell[i])
                L_max = max(L_max, np.max(L))
            dt = min(dt_max, safe_divide(0.05, L_max, 60.0))
            if t + dt > t_total:
                dt = t_total - t

            self.step(dt)
            t += dt

            if len(history['time_hours']) == 0 or t / 3600.0 - history['time_hours'][-1] >= 0.5:
                history['time_hours'].append(t / 3600.0)
                history['total_ozone_du'].append(self._total_ozone_dobson())
                history['o3_min'].append(np.min(self.c_cell[:, StratosphericChemistry.IDX_O3]))
                history['o3_max'].append(np.max(self.c_cell[:, StratosphericChemistry.IDX_O3]))


        if abs(t / 3600.0 - history['time_hours'][-1]) > 1e-6:
            history['time_hours'].append(t / 3600.0)
            history['total_ozone_du'].append(self._total_ozone_dobson())
            history['o3_min'].append(np.min(self.c_cell[:, StratosphericChemistry.IDX_O3]))
            history['o3_max'].append(np.max(self.c_cell[:, StratosphericChemistry.IDX_O3]))

        return history

    def _total_ozone_dobson(self):

        o3 = self.c_cell[:, StratosphericChemistry.IDX_O3]
        vol = self.mesh.cell_volumes
        total_o3 = np.sum(o3 * vol) / np.sum(vol)

        thickness_cm = (self.mesh.alt_range[1] - self.mesh.alt_range[0]) * 1e5
        col = total_o3 * thickness_cm
        du = col / 2.687e16
        return float(du)

    def get_ozone_distribution(self):
        return self.c_cell[:, StratosphericChemistry.IDX_O3].copy()

    def perturb_catalytic_species(self, factor_no=1.0, factor_cl=1.0):
        self.c_cell[:, StratosphericChemistry.IDX_NO] *= factor_no
        self.c_cell[:, StratosphericChemistry.IDX_NO2] *= factor_no
        self.c_cell[:, StratosphericChemistry.IDX_Cl] *= factor_cl
        self.c_cell[:, StratosphericChemistry.IDX_ClO] *= factor_cl
