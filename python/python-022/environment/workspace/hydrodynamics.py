
import numpy as np
from typing import Tuple
from utils import safe_divide, clamp_array
from icf_parameters import NP, PC, TP
from mesh_generator import RadialMesh
from state_equation import total_pressure, sound_speed, electron_number_density


class LagrangeHydro:

    def __init__(self, mesh: RadialMesh):
        self.mesh = mesh
        n = mesh.n_cells


        self.rho = np.zeros(n)
        self.u = np.zeros(n + 1)
        self.P = np.zeros(n)
        self.e = np.zeros(n)
        self.T_e = np.zeros(n)
        self.T_i = np.zeros(n)


        self.mass = np.zeros(n)
        self.vol = np.zeros(n)
        self.q_art = np.zeros(n)
        self.c_s = np.zeros(n)

        self._initialize()

    def _initialize(self):
        n = self.mesh.n_cells
        self.vol = self.mesh.cell_volumes()

        for i in range(n):
            self.rho[i] = self.mesh.get_density_by_zone(i)
            self.mass[i] = self.rho[i] * self.vol[i]

            self.T_e[i] = 20.0
            self.T_i[i] = 20.0


            zone = self.mesh.get_material_zone(i)
            if zone == "ablator":
                mass_per_atom = TP.ablator_average_atomic_mass * 1.0e-3 / PC.AVOGADRO
            else:
                mass_per_atom = 2.5 * 1.0e-3 / PC.AVOGADRO
            self.e[i] = 1.5 * PC.BOLTZMANN * self.T_e[i] / mass_per_atom


        self._update_pressure()

    def _update_pressure(self):
        from state_equation import ionization_state_Saha
        n = self.mesh.n_cells
        for i in range(n):
            zone = self.mesh.get_material_zone(i)





            if zone == "ablator":
                Z_nuc = 6.0
                A_avg = TP.ablator_average_atomic_mass
                ion_E = 11.3 * PC.ELEMENTARY_CHARGE
            elif zone == "dt_ice":
                Z_nuc = 1.0
                A_avg = 2.5
                ion_E = 13.6 * PC.ELEMENTARY_CHARGE
            else:
                Z_nuc = 1.0
                A_avg = 2.5
                ion_E = 13.6 * PC.ELEMENTARY_CHARGE




            Z_eff = ionization_state_Saha(self.rho[i], self.T_e[i], Z_nuc, ion_E)
            Z_eff = max(Z_eff, 1.0e-6)

            self.P[i] = total_pressure(self.rho[i], self.T_e[i], self.T_i[i], Z_eff, A_avg)
            self.c_s[i] = sound_speed(self.rho[i], self.T_e[i], self.T_i[i], Z_eff, A_avg)

    def _compute_artificial_viscosity(self) -> np.ndarray:
        n = self.mesh.n_cells
        q = np.zeros(n)
        C0 = 2.0
        C1 = 0.5

        for i in range(n):
            du = self.u[i + 1] - self.u[i]
            if du < 0.0:
                dr = self.mesh.r[i + 1] - self.mesh.r[i]
                strain_rate = du / max(dr, 1.0e-15)
                q_quad = C0**2 * self.rho[i] * dr**2 * strain_rate**2
                q_lin = C1 * self.rho[i] * self.c_s[i] * dr * abs(strain_rate)
                q[i] = q_quad + q_lin
            else:
                q[i] = 0.0

        return q

    def compute_time_step(self) -> float:
        n = self.mesh.n_cells
        dt_min = NP.MAX_DT

        for i in range(n):
            dr = self.mesh.r[i + 1] - self.mesh.r[i]
            if dr <= 1.0e-15:
                continue
            speed = self.c_s[i] + abs(self.u[i]) + abs(self.u[i + 1])
            if speed < 1.0e-10:
                dt_local = NP.MAX_DT
            else:
                dt_local = NP.CFL * dr / speed
            dt_min = min(dt_min, dt_local)

        return clamp_array(np.array([dt_min]), NP.MIN_DT, NP.MAX_DT)[0]

    def momentum_equation_rhs(self) -> np.ndarray:
        n_nodes = self.mesh.n_nodes
        rhs = np.zeros(n_nodes)
        q = self.q_art

        for i in range(1, n_nodes - 1):
            i_left = i - 1
            r_i = self.mesh.r[i]
            A_i = 4.0 * np.pi * r_i**2
            m_left = self.mass[i_left]
            m_right = self.mass[i]
            P_left = self.P[i_left] + q[i_left]
            P_right = self.P[i] + q[i]

            m_node = 0.5 * (m_left + m_right)
            if m_node < 1.0e-30:
                rhs[i] = 0.0
                continue


            grad_term = A_i * (P_left - P_right) / m_node


            rho_avg = 0.5 * (self.rho[i_left] + self.rho[i])
            P_avg = 0.5 * (P_left + P_right)
            if r_i < 1.0e-15 or rho_avg < 1.0e-30:
                geom_term = 0.0
            else:
                geom_term = -2.0 * P_avg / (rho_avg * r_i)

            rhs[i] = grad_term + geom_term


        rhs[0] = 0.0

        if n_nodes > 1:
            A_outer = 4.0 * np.pi * self.mesh.r[-1]**2
            m_last = 0.5 * (self.mass[-1] + self.mass[-1])
            if m_last > 1.0e-30:
                rhs[-1] = A_outer * (self.P[-1] + q[-1]) / m_last
            else:
                rhs[-1] = 0.0

        return rhs

    def energy_equation_rhs(self, laser_heating: np.ndarray,
                            fusion_heating: np.ndarray,
                            conduction_work: np.ndarray) -> np.ndarray:
        n = self.mesh.n_cells
        rhs = np.zeros(n)

        for i in range(n):
            r1, r2 = self.mesh.r[i], self.mesh.r[i + 1]
            rc = 0.5 * (r1 + r2)
            dr = r2 - r1


            if rc < 1.0e-12:
                div_u = 0.0
            else:
                div_u = (r2**2 * self.u[i + 1] - r1**2 * self.u[i]) / (rc**2 * max(dr, 1.0e-15))


            div_u = np.clip(div_u, -1.0e15, 1.0e15)


            pdV = -(self.P[i] + self.q_art[i]) * div_u / max(self.rho[i], 1.0e-30)

            rhs[i] = pdV + laser_heating[i] + fusion_heating[i] + conduction_work[i]

        return rhs

    def advance(self, dt: float,
                laser_heating: np.ndarray,
                fusion_heating: np.ndarray,
                conduction_work: np.ndarray):
        n_cells = self.mesh.n_cells
        n_nodes = self.mesh.n_nodes


        self.q_art = self._compute_artificial_viscosity()


        u_old = self.u.copy()
        e_old = self.e.copy()
        r_old = self.mesh.r.copy()


        rhs_u = self.momentum_equation_rhs()
        rhs_e = self.energy_equation_rhs(laser_heating, fusion_heating, conduction_work)


        u_pred = u_old + dt * rhs_u
        e_pred = e_old + dt * rhs_e


        r_pred = r_old.copy()
        for i in range(n_nodes):
            r_pred[i] = r_old[i] + dt * u_pred[i]


        r_pred[0] = 0.0
        r_pred = np.maximum(r_pred, 0.0)

        for i in range(1, n_nodes):
            if r_pred[i] < r_pred[i - 1] + 1.0e-15:
                r_pred[i] = r_pred[i - 1] + 1.0e-15


        self.mesh.r = r_pred
        self.u = u_pred
        self.e = e_pred


        for i in range(n_cells):
            new_vol = 4.0 * np.pi / 3.0 * (r_pred[i + 1]**3 - r_pred[i]**3)
            self.vol[i] = max(new_vol, 1.0e-30)
            self.rho[i] = self.mass[i] / self.vol[i]


        for i in range(n_cells):
            zone = self.mesh.get_material_zone(i)
            if zone == "ablator":
                cv = 1.5 * PC.BOLTZMANN / (TP.ablator_average_atomic_mass * 1.0e-3 / PC.AVOGADRO)
            else:
                cv = 1.5 * PC.BOLTZMANN / (2.5 * 1.0e-3 / PC.AVOGADRO)
            dT = self.e[i] - e_old[i]

            dT_clamped = np.clip(dT, -0.5 * self.T_e[i] * cv, 0.5 * self.T_e[i] * cv)
            self.T_e[i] = max(self.T_e[i] + dT_clamped / max(cv, 1.0e-30), 1.0)
            self.T_i[i] = self.T_e[i]

        self._update_pressure()


        self.q_art = self._compute_artificial_viscosity()
        rhs_u2 = self.momentum_equation_rhs()
        rhs_e2 = self.energy_equation_rhs(laser_heating, fusion_heating, conduction_work)


        du = 0.5 * dt * (rhs_u + rhs_u2)
        du_clamped = np.clip(du, -1.0e6, 1.0e6)
        self.u = u_old + du_clamped
        self.e = e_old + 0.5 * dt * (rhs_e + rhs_e2)


        for i in range(n_nodes):
            self.mesh.r[i] = r_old[i] + dt * self.u[i]
        self.mesh.r[0] = 0.0
        self.mesh.r = np.maximum(self.mesh.r, 0.0)
        for i in range(1, n_nodes):
            if self.mesh.r[i] < self.mesh.r[i - 1] + 1.0e-15:
                self.mesh.r[i] = self.mesh.r[i - 1] + 1.0e-15


        for i in range(n_cells):
            new_vol = 4.0 * np.pi / 3.0 * (self.mesh.r[i + 1]**3 - self.mesh.r[i]**3)
            self.vol[i] = max(new_vol, 1.0e-30)
            self.rho[i] = self.mass[i] / self.vol[i]

            zone = self.mesh.get_material_zone(i)
            if zone == "ablator":
                cv = 1.5 * PC.BOLTZMANN / (TP.ablator_average_atomic_mass * 1.0e-3 / PC.AVOGADRO)
            else:
                cv = 1.5 * PC.BOLTZMANN / (2.5 * 1.0e-3 / PC.AVOGADRO)
            dT = self.e[i] - e_old[i]
            dT_clamped = np.clip(dT, -0.5 * self.T_e[i] * cv, 0.5 * self.T_e[i] * cv)
            self.T_e[i] = max(self.T_e[i] + dT_clamped / max(cv, 1.0e-30), 1.0)
            self.T_i[i] = self.T_e[i]

        self._update_pressure()

    def get_kinetic_energy(self) -> float:
        n_nodes = self.mesh.n_nodes
        ke = 0.0
        for i in range(n_nodes - 1):

            m_node = 0.5 * (self.mass[max(i - 1, 0)] + self.mass[min(i, self.mesh.n_cells - 1)])
            m_node = max(m_node, 0.0)
            ke += 0.5 * m_node * self.u[i]**2
        return ke

    def get_internal_energy(self) -> float:
        return float(np.sum(self.mass * self.e))
