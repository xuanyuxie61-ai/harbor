# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple
from utils import robust_divide, ensure_positive


class CosmologyParams:
    def __init__(self):
        self.Omega_b = 0.022383
        self.Omega_c = 0.12011
        self.Omega_L = 0.684
        self.h = 0.6732
        self.Tcmb = 2.7255
        self.YHe = 0.2454
        self.Neff = 3.046


class BoltzmannSolver:

    def __init__(self, params: CosmologyParams, k_mode: float,
                 n_eta: int = 2000, eta_max: float = 14000.0):
        self.params = params
        self.k = ensure_positive(k_mode, "k_mode")
        self.n_eta = n_eta
        self.eta_max = eta_max

        self.eta = np.linspace(0.0, eta_max, n_eta)
        self.deta = self.eta[1] - self.eta[0]

        self._init_background()




    def _scale_factor(self, eta: float) -> float:
        Omega_r = 2.47e-5 / (self.params.h ** 2)
        Omega_m = (self.params.Omega_b + self.params.Omega_c) / (self.params.h ** 2)
        a_eq = Omega_r / Omega_m



        H0 = 100.0 * self.params.h

        eta_eq = 14.0 / np.sqrt(Omega_m * (self.params.h ** 2))
        if eta <= 0.0:
            return 1e-10
        a = a_eq * ((eta / eta_eq) ** 2 + 2.0 * (eta / eta_eq))
        return max(a, 1e-10)

    def _hub_conformal(self, eta: float) -> float:
        a = self._scale_factor(eta)

        dh = 1e-4
        a_p = self._scale_factor(eta + dh)
        a_m = self._scale_factor(max(eta - dh, 1e-6))
        dadeta = (a_p - a_m) / (2.0 * dh)
        return dadeta / a

    def _thomson_opacity(self, eta: float) -> float:
        a = self._scale_factor(eta)
        z = 1.0 / a - 1.0
        if z < 50.0:
            return 0.0

        tau_dot = 7.0e-4 * ((1.0 + z) / 1100.0) ** 2.5
        return tau_dot

    def _init_background(self):
        self.a_arr = np.array([self._scale_factor(e) for e in self.eta])
        self.H_arr = np.array([self._hub_conformal(e) for e in self.eta])
        self.tau_dot_arr = np.array([self._thomson_opacity(e) for e in self.eta])




    def _rhs(self, y: np.ndarray, ieta: int) -> np.ndarray:
        eta = self.eta[ieta]
        k = self.k
        a = self.a_arr[ieta]
        H = self.H_arr[ieta]
        tau_dot = self.tau_dot_arr[ieta]

        Delta0, Delta1, Delta2, vb, Phi = y


        rho_b_over_rho = self.params.Omega_b / (self.params.Omega_b + self.params.Omega_c)
        R = 3.0 * rho_b_over_rho / (4.0 * (1.0 - rho_b_over_rho))


        slip = Delta1 - vb
        if tau_dot > 1e-3:

            slip = 0.0


        Psi = -Phi


        dDelta0 = - (k / 3.0) * Delta1 - self._dPhi_deta(ieta)


        dDelta1 = k * (Delta0 + Psi)
        if tau_dot > 1e-3:
            dDelta1 -= tau_dot * slip


        dDelta2 = (2.0 * k / 3.0) * Delta1
        if tau_dot > 1e-3:
            dDelta2 -= (9.0 / 10.0) * tau_dot * Delta2


        dvb = -H * vb + k * Psi
        if tau_dot > 1e-3:
            dvb += (tau_dot / R) * slip


        dPhi = self._dPhi_deta(ieta)

        return np.array([dDelta0, dDelta1, dDelta2, dvb, dPhi])

    def _dPhi_deta(self, ieta: int) -> float:
        if ieta <= 0 or ieta >= self.n_eta - 1:
            return 0.0

        H = self.H_arr[ieta]
        return -H * self._Phi_from_constraint(ieta)

    def _Phi_from_constraint(self, ieta: int) -> float:
        k = self.k
        if k < 1e-12:
            return 1.0
        H = self.H_arr[ieta]
        rho_b_over_rho = self.params.Omega_b / (self.params.Omega_b + self.params.Omega_c)

        Delta0_init = 1.0
        Phi = -1.5 * (H ** 2) / (k ** 2) * Delta0_init
        return Phi




    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_eta
        Delta0 = np.zeros(n)
        Delta1 = np.zeros(n)
        Delta2 = np.zeros(n)
        vb = np.zeros(n)
        Phi = np.zeros(n)



        Phi0 = 1.0
        Delta0[0] = -2.0 / 3.0 * Phi0
        Delta1[0] = 0.0
        Delta2[0] = 0.0
        vb[0] = 0.0
        Phi[0] = Phi0


        for i in range(n - 1):
            y = np.array([Delta0[i], Delta1[i], Delta2[i], vb[i], Phi[i]])
            k1 = self._rhs(y, i)
            y_pred = y + self.deta * k1

            y_pred = np.clip(y_pred, -1e6, 1e6)
            k2 = self._rhs(y_pred, i + 1)
            y_new = y + 0.5 * self.deta * (k1 + k2)
            Delta0[i + 1], Delta1[i + 1], Delta2[i + 1], vb[i + 1], Phi[i + 1] = y_new

        return self.eta, Delta0, Delta1, Delta2, vb

    def transfer_function_today(self) -> float:



        raise NotImplementedError("Hole_1: 请补全 transfer_function_today 的实现")
