# -*- coding: utf-8 -*-

import numpy as np
from parameters import get_parameters


class SheathODE:

    def __init__(self, params=None):
        if params is None:
            params = get_parameters()
        self.params = params
        self._setup_coefficients()

    def _setup_coefficients(self):
        p = self.params
        self.lambda_D = p.debye_length()
        self.c_s = p.ion_sound_speed()
        self.n0 = p.get('n_0')
        self.Te = p.get('T_e')
        self.Ti = p.get('T_i')
        self.mi_amu = p.get('m_i')
        self.Zi = p.get('Z_i')

        self.alpha_rec = 1.0e-20 * (self.Te / 10.0)**(-0.5)
        if self.alpha_rec < 1.0e-25:
            self.alpha_rec = 1.0e-25

        self.k_ion = 1.0e-14 * np.exp(-13.6 / self.Te) if self.Te > 0 else 0.0

    def density_derivative(self, x, n_i, v_i):

        if n_i <= 0:
            n_i = 1.0e10
        if v_i <= 0:
            v_i = self.c_s


        logistic_term = (n_i / self.lambda_D) * (1.0 - n_i / self.n0)


        if v_i > 1.0e-10:
            rec_term = self.alpha_rec * n_i**2 / v_i
        else:
            rec_term = 0.0

        dnidx = logistic_term - rec_term


        max_rate = self.n0 / self.lambda_D
        if abs(dnidx) > max_rate:
            dnidx = np.sign(dnidx) * max_rate

        return dnidx

    def velocity_derivative(self, x, n_i, v_i, e_field):


        raise NotImplementedError("Hole_1: 请实现 velocity_derivative 方法")


    def exact_density_solution(self, x_arr):
        x_arr = np.asarray(x_arr, dtype=float)
        n_s = self.n0 * 0.5
        lam = self.lambda_D
        if lam <= 0:
            return np.full_like(x_arr, self.n0)

        exp_term = np.exp(x_arr / lam)
        numerator = self.n0 * n_s * exp_term
        denominator = self.n0 + n_s * (exp_term - 1.0)


        denominator = np.where(denominator < 1.0, 1.0, denominator)

        n_i = numerator / denominator
        return n_i

    def solve_sheath_profile(self, nx=None, x_max=None):
        if nx is None:
            nx = self.params.get('nx')
        if x_max is None:
            x_max = self.params.get('x_max')

        x = np.linspace(0.0, x_max, nx)
        dx = x[1] - x[0]

        n_i = np.zeros(nx)
        v_i = np.zeros(nx)
        phi = np.zeros(nx)
        e_field = np.zeros(nx)


        n_i[0] = self.n0 * 0.5
        v_i[0] = self.c_s
        phi[0] = 0.0


        for idx in range(nx):
            denom = self.lambda_D + x[idx]
            if denom > 0:
                e_field[idx] = self.Te / denom
            else:
                e_field[idx] = 0.0


        for idx in range(nx - 1):
            xi = x[idx]
            ni = n_i[idx]
            vi = v_i[idx]
            Ei = e_field[idx]


            k1_n = self.density_derivative(xi, ni, vi)
            k1_v = self.velocity_derivative(xi, ni, vi, Ei)


            k2_n = self.density_derivative(xi + 0.5*dx, ni + 0.5*dx*k1_n, vi + 0.5*dx*k1_v)
            k2_v = self.velocity_derivative(xi + 0.5*dx, ni + 0.5*dx*k1_n, vi + 0.5*dx*k1_v, Ei)


            k3_n = self.density_derivative(xi + 0.5*dx, ni + 0.5*dx*k2_n, vi + 0.5*dx*k2_v)
            k3_v = self.velocity_derivative(xi + 0.5*dx, ni + 0.5*dx*k2_n, vi + 0.5*dx*k2_v, Ei)


            k4_n = self.density_derivative(xi + dx, ni + dx*k3_n, vi + dx*k3_v)
            k4_v = self.velocity_derivative(xi + dx, ni + dx*k3_n, vi + dx*k3_v, Ei)

            n_i[idx+1] = ni + (dx/6.0)*(k1_n + 2*k2_n + 2*k3_n + k4_n)
            v_i[idx+1] = vi + (dx/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)


            if n_i[idx+1] < 1.0e10:
                n_i[idx+1] = 1.0e10
            if v_i[idx+1] < self.c_s:
                v_i[idx+1] = self.c_s


            phi[idx+1] = phi[idx] - e_field[idx] * dx

        return x, n_i, v_i, phi, e_field

    def compute_ion_flux(self, n_i, v_i):
        return n_i * v_i

    def compute_ion_energy_at_wall(self, v_i_wall):
        m_p = 1.67262192369e-27
        e_charge = 1.602176634e-19
        mi_kg = self.mi_amu * m_p

        kinetic_ev = 0.5 * mi_kg * v_i_wall**2 / e_charge
        sheath_potential = abs(self.params.sheath_potential())

        E_total = kinetic_ev + self.Zi * sheath_potential + self.Zi * self.Te
        return E_total

    def compute_sheath_edge_mach(self, v_i):
        if self.c_s <= 0:
            return np.zeros_like(v_i)
        M = v_i / self.c_s
        return M


def demo_sheath_ode():
    sheath = SheathODE()
    x, n_i, v_i, phi, e_field = sheath.solve_sheath_profile(nx=128, x_max=0.005)
    gamma = sheath.compute_ion_flux(n_i, v_i)
    M = sheath.compute_sheath_edge_mach(v_i)
    E_wall = sheath.compute_ion_energy_at_wall(v_i[-1])

    print("鞘层ODE求解结果:")
    print(f"  壁面离子密度     = {n_i[-1]:.3e} m^-3")
    print(f"  壁面离子速度     = {v_i[-1]:.3e} m/s")
    print(f"  壁面Mach数       = {M[-1]:.3f}")
    print(f"  壁面离子通量     = {gamma[-1]:.3e} m^-2 s^-1")
    print(f"  壁面离子能量     = {E_wall:.2f} eV")
    print(f"  Bohm判据满足     = {M[0] >= 1.0}")
    return x, n_i, v_i, phi, e_field


if __name__ == "__main__":
    demo_sheath_ode()
