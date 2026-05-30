# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Optional, Tuple
from vandermonde_basis import (
    jacobi_gauss_lobatto_points,
    jacobi_gauss_lobatto_weights,
    vandermonde_matrix_1d,
    differentiation_matrix_1d,
)


class DGSolver1D:

    def __init__(self, num_elements: int, poly_order: int,
                 x_min: float, x_max: float,
                 rho_func: Callable, E_func: Callable,
                 refine_strength: float = 0.0):
        if num_elements < 1:
            raise ValueError("num_elements must be >= 1.")
        if poly_order < 1:
            raise ValueError("poly_order must be >= 1.")

        self.num_elements = num_elements
        self.poly_order = poly_order
        self.x_min = x_min
        self.x_max = x_max
        self.L = x_max - x_min
        self.rho_func = rho_func
        self.E_func = E_func


        self.ref_nodes = jacobi_gauss_lobatto_points(poly_order)
        self.ref_weights = jacobi_gauss_lobatto_weights(self.ref_nodes)


        self.D_ref = differentiation_matrix_1d(poly_order, self.ref_nodes)


        self._build_mesh(refine_strength)


        self._build_element_data()


        self.lserk_a = np.array([0.0, -567301805773.0 / 1357537059087.0,
                                 -2404267990393.0 / 2016746695238.0,
                                 -3550918686646.0 / 2091501179385.0,
                                 -1275806237668.0 / 842570457699.0])
        self.lserk_b = np.array([1432997174477.0 / 9575080441755.0,
                                 5161836677717.0 / 13612068292357.0,
                                 1720146321549.0 / 2090206949498.0,
                                 3134564353537.0 / 4481467310338.0,
                                 2277821191437.0 / 14882151754819.0])
        self.lserk_c = np.array([0.0, 1432997174477.0 / 9575080441755.0,
                                 2526269341429.0 / 6820363962896.0,
                                 2006345519317.0 / 3224310063776.0,
                                 2802321613138.0 / 2924317926251.0])


        self.num_dof = num_elements * (poly_order + 1)
        self.strain = np.zeros(self.num_dof)
        self.velocity = np.zeros(self.num_dof)
        self.stress = np.zeros(self.num_dof)
        self.time = 0.0

    def _build_mesh(self, refine_strength: float):
        xi_uniform = np.linspace(0.0, 1.0, self.num_elements + 1)
        a = refine_strength
        f_xi = xi_uniform + a * 0.5 * (1.0 - np.cos(np.pi * xi_uniform))
        f_1 = 1.0 + a * 0.5 * (1.0 - np.cos(np.pi))
        x_nodes = self.x_min + self.L * f_xi / f_1
        x_nodes[0] = self.x_min
        x_nodes[-1] = self.x_max
        self.elem_vertices = np.stack([x_nodes[:-1], x_nodes[1:]], axis=1)

    def _build_element_data(self):
        Np = self.poly_order + 1
        self.elem_nodes = np.zeros((self.num_elements, Np))
        self.elem_jac = np.zeros(self.num_elements)
        self.elem_invjac = np.zeros(self.num_elements)
        self.elem_rho = np.zeros((self.num_elements, Np))
        self.elem_E = np.zeros((self.num_elements, Np))
        self.elem_Z = np.zeros((self.num_elements, Np))
        self.elem_mass_inv = np.zeros((self.num_elements, Np))

        for e in range(self.num_elements):
            xL, xR = self.elem_vertices[e]
            h_e = xR - xL
            J = h_e / 2.0
            self.elem_jac[e] = J
            self.elem_invjac[e] = 1.0 / J


            x_phys = 0.5 * (xL + xR) + J * self.ref_nodes
            self.elem_nodes[e, :] = x_phys


            rho_e = self.rho_func(x_phys)
            E_e = self.E_func(x_phys)
            self.elem_rho[e, :] = rho_e
            self.elem_E[e, :] = E_e
            self.elem_Z[e, :] = np.sqrt(rho_e * E_e)


            self.elem_mass_inv[e, :] = 1.0 / (J * self.ref_weights * rho_e)

    def get_dof_index(self, elem: int, node: int) -> int:
        return elem * (self.poly_order + 1) + node

    def get_element_solution(self, elem: int):
        idx = slice(elem * (self.poly_order + 1), (elem + 1) * (self.poly_order + 1))
        return self.strain[idx], self.velocity[idx]

    def set_element_solution(self, elem: int, strain_loc: np.ndarray, velocity_loc: np.ndarray):
        idx = slice(elem * (self.poly_order + 1), (elem + 1) * (self.poly_order + 1))
        self.strain[idx] = strain_loc
        self.velocity[idx] = velocity_loc

    def compute_stress(self):
        self.stress = self.elem_E.flatten() * self.strain

    def _compute_interface_fluxes(self) -> Tuple[np.ndarray, np.ndarray]:
        Np = self.poly_order + 1

        num_interfaces = self.num_elements + 1
        flux_v = np.zeros(num_interfaces)
        flux_sigma = np.zeros(num_interfaces)

        for iface in range(num_interfaces):
            if iface == 0:



                e = 0
                idx = e * Np
                sigma_in = self.stress[idx]
                v_in = self.velocity[idx]
                Z_in = self.elem_Z[e, 0]



                w_out = sigma_in - Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = -0.5 * w_out / Z_in
            elif iface == num_interfaces - 1:

                e = self.num_elements - 1
                idx = (e + 1) * Np - 1
                sigma_in = self.stress[idx]
                v_in = self.velocity[idx]
                Z_in = self.elem_Z[e, -1]

                w_out = sigma_in + Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = 0.5 * w_out / Z_in
            else:

                eL = iface - 1
                eR = iface
                idxL = (eL + 1) * Np - 1
                idxR = eR * Np
                sigmaL = self.stress[idxL]
                sigmaR = self.stress[idxR]
                vL = self.velocity[idxL]
                vR = self.velocity[idxR]
                ZL = self.elem_Z[eL, -1]
                ZR = self.elem_Z[eR, 0]
                Z_sum = ZL + ZR
                if Z_sum < 1e-30:
                    Z_sum = 1e-30

                flux_sigma[iface] = (ZR * sigmaL + ZL * sigmaR) / Z_sum + (ZL * ZR) * (vL - vR) / Z_sum
                flux_v[iface] = (ZR * vR + ZL * vL) / Z_sum + (sigmaL - sigmaR) / Z_sum
        return flux_v, flux_sigma

    def _rhs_strain_velocity(self, strain: np.ndarray, velocity: np.ndarray,
                             f_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        Np = self.poly_order + 1

        stress = self.elem_E.flatten() * strain

        dstrain_dt = np.zeros_like(strain)
        dvelocity_dt = np.zeros_like(velocity)

        flux_v, flux_sigma = self._compute_interface_fluxes_from_state(strain, velocity, stress)

        for e in range(self.num_elements):
            idx = slice(e * Np, (e + 1) * Np)
            strain_e = strain[idx]
            velocity_e = velocity[idx]
            stress_e = stress[idx]
            invJ = self.elem_invjac[e]
            mass_inv = self.elem_mass_inv[e, :]



            surface_term_v = np.zeros(Np)
            surface_term_v[0] = -flux_v[e]
            surface_term_v[-1] = flux_v[e + 1]
            volume_term_v = self.D_ref @ velocity_e * invJ
            dstrain_dt[idx] = mass_inv * (surface_term_v - self.ref_weights * volume_term_v)


            surface_term_s = np.zeros(Np)
            surface_term_s[0] = -flux_sigma[e]
            surface_term_s[-1] = flux_sigma[e + 1]
            volume_term_s = self.D_ref @ stress_e * invJ
            rhs_v = surface_term_s - self.ref_weights * volume_term_s

            if f_func is not None:
                x_e = self.elem_nodes[e, :]
                f_e = f_func(x_e, self.time)
                rhs_v += self.ref_weights * f_e * self.elem_jac[e]


            dvelocity_dt[idx] = mass_inv * rhs_v

        return dstrain_dt, dvelocity_dt

    def _compute_interface_fluxes_from_state(self, strain: np.ndarray, velocity: np.ndarray,
                                              stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Np = self.poly_order + 1
        num_interfaces = self.num_elements + 1
        flux_v = np.zeros(num_interfaces)
        flux_sigma = np.zeros(num_interfaces)

        for iface in range(num_interfaces):
            if iface == 0:
                e = 0
                idx = e * Np
                sigma_in = stress[idx]
                v_in = velocity[idx]
                Z_in = self.elem_Z[e, 0]
                w_out = sigma_in - Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = -0.5 * w_out / (Z_in + 1e-30)
            elif iface == num_interfaces - 1:
                e = self.num_elements - 1
                idx = (e + 1) * Np - 1
                sigma_in = stress[idx]
                v_in = velocity[idx]
                Z_in = self.elem_Z[e, -1]
                w_out = sigma_in + Z_in * v_in
                flux_sigma[iface] = 0.5 * w_out
                flux_v[iface] = 0.5 * w_out / (Z_in + 1e-30)
            else:
                eL = iface - 1
                eR = iface
                idxL = (eL + 1) * Np - 1
                idxR = eR * Np
                sigmaL = stress[idxL]
                sigmaR = stress[idxR]
                vL = velocity[idxL]
                vR = velocity[idxR]
                ZL = self.elem_Z[eL, -1]
                ZR = self.elem_Z[eR, 0]
                Z_sum = ZL + ZR
                if Z_sum < 1e-30:
                    Z_sum = 1e-30
                flux_sigma[iface] = (ZR * sigmaL + ZL * sigmaR) / Z_sum + (ZL * ZR) * (vL - vR) / Z_sum
                flux_v[iface] = (ZR * vR + ZL * vL) / Z_sum + (sigmaL - sigmaR) / Z_sum
        return flux_v, flux_sigma

    def step(self, dt: float, f_func: Optional[Callable] = None):
        Np = self.poly_order + 1
        strain0 = self.strain.copy()
        velocity0 = self.velocity.copy()

        strain_curr = strain0.copy()
        velocity_curr = velocity0.copy()

        for stage in range(5):
            self.strain = strain_curr
            self.velocity = velocity_curr
            self.compute_stress()
            rhs_strain, rhs_vel = self._rhs_strain_velocity(strain_curr, velocity_curr, f_func)

            if stage == 0:
                strain0 = strain_curr.copy()
                velocity0 = velocity_curr.copy()


            strain_curr = strain0 + self.lserk_b[stage] * dt * rhs_strain
            velocity_curr = velocity0 + self.lserk_b[stage] * dt * rhs_vel

        self.strain = strain_curr
        self.velocity = velocity_curr
        self.time += dt
        self.compute_stress()

    def run(self, t_final: float, dt: Optional[float] = None,
            f_func: Optional[Callable] = None,
            callback: Optional[Callable] = None) -> dict:
        if dt is None:

            min_h = np.min(self.elem_vertices[:, 1] - self.elem_vertices[:, 0])
            max_c = np.max(self.elem_Z / self.elem_rho)
            cfl = 0.5
            dt = cfl * min_h / (self.poly_order ** 2 * max_c + 1e-30)

        num_steps = int(np.ceil(t_final / dt))
        dt = t_final / num_steps


        hist = {
            "time": [],
            "strain_max": [],
            "velocity_max": [],
            "stress_max": [],
            "energy_kinetic": [],
            "energy_strain": [],
        }

        for step in range(num_steps):
            self.step(dt, f_func)
            if callback is not None:
                callback(self.time, self)


            if step % max(1, num_steps // 100) == 0:
                hist["time"].append(self.time)
                hist["strain_max"].append(np.max(np.abs(self.strain)))
                hist["velocity_max"].append(np.max(np.abs(self.velocity)))
                hist["stress_max"].append(np.max(np.abs(self.stress)))


                ke = 0.0
                se = 0.0
                for e in range(self.num_elements):
                    idx = slice(e * (self.poly_order + 1), (e + 1) * (self.poly_order + 1))
                    v_e = self.velocity[idx]
                    eps_e = self.strain[idx]
                    rho_e = self.elem_rho[e, :]
                    E_e = self.elem_E[e, :]
                    J = self.elem_jac[e]
                    ke += np.sum(J * self.ref_weights * rho_e * v_e ** 2)
                    se += np.sum(J * self.ref_weights * E_e * eps_e ** 2)
                hist["energy_kinetic"].append(0.5 * ke)
                hist["energy_strain"].append(0.5 * se)


        for key in hist:
            hist[key] = np.array(hist[key])
        return hist


if __name__ == "__main__":

    L = 1.0
    rho0 = 1600.0
    E0 = 100e9
    solver = DGSolver1D(
        num_elements=20, poly_order=3,
        x_min=0.0, x_max=L,
        rho_func=lambda x: np.full_like(np.asarray(x), rho0),
        E_func=lambda x: np.full_like(np.asarray(x), E0),
        refine_strength=0.5
    )


    x_all = solver.elem_nodes.flatten()
    sigma0 = 1e6 * np.exp(-((x_all - 0.3 * L) ** 2) / (2 * (0.05 * L) ** 2))
    solver.stress = sigma0
    solver.strain = sigma0 / E0
    solver.velocity = np.zeros_like(solver.velocity)

    c = np.sqrt(E0 / rho0)
    t_final = 2 * L / c
    dt = 0.2 * (L / 20) / (3 ** 2 * c)
    hist = solver.run(t_final=t_final, dt=dt)
    print("DG solver self-test completed.")
    print("Final time:", solver.time)
    print("Max strain history:", hist["strain_max"][:5])
