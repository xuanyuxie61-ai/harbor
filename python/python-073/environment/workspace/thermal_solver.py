# -*- coding: utf-8 -*-

import numpy as np
from utils import blasius_function, sutherland_viscosity, safe_divide, compressible_blasius_velocity


class HypersonicThermalSolver:

    def __init__(self, Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4,
                 Tw_over_Te=1.0, L=1.0, N_eta=200, eta_max=12.0):
        self.Ma = Ma
        self.Re = Re
        self.Pr = Pr
        self.gamma = gamma
        self.Tw_over_Te = Tw_over_Te
        self.L = L
        self.N_eta = N_eta
        self.eta_max = eta_max


        self.eta = np.linspace(0.0, eta_max, N_eta)
        self.deta = self.eta[1] - self.eta[0]

    def solve_self_similar_energy(self, epsilon=1e-10, max_iter=50000):
        n = self.N_eta
        deta = self.deta
        deta2 = deta ** 2


        Te = 1.0
        Tw = self.Tw_over_Te
        T_old = np.full(n, Tw + (Te - Tw) * self.eta / self.eta_max)
        T_new = T_old.copy()


        f, fp, fpp = blasius_function(self.eta)
        u = np.clip(fp, 0.0, 1.0)


        dup = np.zeros(n)
        dup[1:-1] = (u[2:] - u[:-2]) / (2.0 * deta)
        dup[0] = (u[1] - u[0]) / deta
        dup[-1] = (u[-1] - u[-2]) / deta


        diss_coeff = (self.gamma - 1.0) / 2.0 * self.Ma**2

        iterations = 0
        diff = epsilon + 1.0

        while diff >= epsilon and iterations < max_iter:
            T_old[:] = T_new[:]


            mu = sutherland_viscosity(T_old)

            rho = safe_divide(1.0, T_old, fill_value=1.0)




            for j in range(1, n - 1):

                mu_m = 0.5 * (mu[j] + mu[j - 1])
                mu_p = 0.5 * (mu[j] + mu[j + 1])


                D_m = mu_m / self.Pr
                D_p = mu_p / self.Pr









                a_j = D_m / deta2 + 0.5 * f[j] / deta
                b_j = -(D_m + D_p) / deta2 - 0.5 * f[j] / deta
                c_j = D_p / deta2


                source = diss_coeff * 2.0 * mu[j] * dup[j]**2


                denom = b_j
                if abs(denom) < 1e-15:
                    denom = -1e-15
                T_new[j] = (-a_j * T_old[j - 1] - c_j * T_old[j + 1] - source) / denom


            T_new[0] = Tw
            T_new[-1] = Te


            diff = np.max(np.abs(T_new - T_old))
            iterations += 1


        mu_final = sutherland_viscosity(T_new)
        rho_final = safe_divide(1.0, T_new, fill_value=1.0)

        return {
            'eta': self.eta,
            'T': T_new,
            'u': u,
            'mu': mu_final,
            'rho': rho_final,
            'iterations': iterations,
            'diff': diff
        }

    def compute_wall_heat_flux(self, solution):
        T = solution['T']
        deta = self.deta
        dTdeta_w = (T[1] - T[0]) / deta




        St_approx = -dTdeta_w / (self.Pr * np.sqrt(self.Re))
        return St_approx

    def compute_skin_friction(self, solution):
        u = solution['u']
        mu = solution['mu']
        deta = self.deta

        dudeta_w = (u[1] - u[0]) / deta
        cf = 2.0 * mu[0] * dudeta_w / np.sqrt(self.Re)
        return cf
