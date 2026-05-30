# -*- coding: utf-8 -*-

import numpy as np
from utils import clip_positive, safe_divide



ARR2 = {
    'O_O3':     {'A': 8.0e-12, 'Ea': 2060.0},
    'NO_O3':    {'A': 1.8e-12, 'Ea': 1370.0},
    'NO2_O':    {'A': 9.3e-12, 'Ea': -120.0},
    'Cl_O3':    {'A': 2.3e-11, 'Ea': 200.0},
    'ClO_O':    {'A': 2.8e-11, 'Ea': -260.0},
    'OH_O3':    {'A': 1.7e-12, 'Ea': 940.0},
    'HO2_O':    {'A': 2.9e-11, 'Ea': -200.0},
}


ARR3 = {'O_O2_M': {'A': 6.0e-34, 'Ea': -1450.0, 'n': 2.4}}

R_GAS = 1.987


def arrhenius_rate(A, Ea, T):
    T = clip_positive(T, 100.0)
    return A * np.exp(-Ea / (R_GAS * T))


def three_body_rate(A, Ea, n, T, M):
    T = clip_positive(T, 100.0)
    k0 = A * (T / 300.0) ** (-n) * np.exp(-Ea / (R_GAS * T))
    return k0 * M


class StratosphericChemistry:


    IDX_O = 0
    IDX_O1D = 1
    IDX_O3 = 2
    IDX_NO = 3
    IDX_NO2 = 4
    IDX_Cl = 5
    IDX_ClO = 6
    IDX_OH = 7
    IDX_HO2 = 8
    IDX_O2 = 9
    IDX_N2 = 10
    IDX_M = 11
    N_SPECIES = 12

    def __init__(self, T_k=220.0, M_cm3=2.5e19):
        self.T = T_k
        self.M = M_cm3
        self._compute_rate_constants()

    def _compute_rate_constants(self):
        self.k = {}
        for key, param in ARR2.items():
            self.k[key] = arrhenius_rate(param['A'], param['Ea'], self.T)
        for key, param in ARR3.items():
            self.k[key] = three_body_rate(param['A'], param['Ea'], param['n'], self.T, self.M)

    def set_photolysis_rates(self, J_o2, J_o3):
        self.J_o2 = float(clip_positive(J_o2))
        self.J_o3 = float(clip_positive(J_o3))

    def set_temperature(self, T_k):
        self.T = T_k
        self._compute_rate_constants()

    def production_loss(self, c, J_o2=None, J_o3=None):
        c = np.asarray(c, dtype=float)
        if c.size < self.N_SPECIES:
            c = np.pad(c, (0, self.N_SPECIES - c.size), constant_values=0.0)
        c = np.maximum(c, 1e-30)

        if J_o2 is None:
            J_o2 = getattr(self, 'J_o2', 1e-11)
        if J_o3 is None:
            J_o3 = getattr(self, 'J_o3', 1e-3)

        k = self.k
        O, O1D, O3 = c[self.IDX_O], c[self.IDX_O1D], c[self.IDX_O3]
        NO, NO2 = c[self.IDX_NO], c[self.IDX_NO2]
        Cl, ClO = c[self.IDX_Cl], c[self.IDX_ClO]
        OH, HO2 = c[self.IDX_OH], c[self.IDX_HO2]
        O2 = c[self.IDX_O2]
        M = c[self.IDX_M]

        P = np.zeros(self.N_SPECIES)
        L = np.zeros(self.N_SPECIES)


        P[self.IDX_O] += 2.0 * J_o2 * O2
        L[self.IDX_O2] += J_o2


        P[self.IDX_O3] += k['O_O2_M'] * O * O2
        L[self.IDX_O] += k['O_O2_M'] * O2
        L[self.IDX_O2] += k['O_O2_M'] * O


        P[self.IDX_O1D] += J_o3 * O3
        P[self.IDX_O2] += J_o3 * O3
        L[self.IDX_O3] += J_o3


        rx = k['O_O3'] * O * O3
        P[self.IDX_O2] += 2.0 * rx
        L[self.IDX_O] += k['O_O3'] * O3
        L[self.IDX_O3] += k['O_O3'] * O


        rx = k['NO_O3'] * NO * O3
        P[self.IDX_NO2] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_NO] += k['NO_O3'] * O3
        L[self.IDX_O3] += k['NO_O3'] * NO


        rx = k['NO2_O'] * NO2 * O
        P[self.IDX_NO] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_NO2] += k['NO2_O'] * O
        L[self.IDX_O] += k['NO2_O'] * NO2


        rx = k['Cl_O3'] * Cl * O3
        P[self.IDX_ClO] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_Cl] += k['Cl_O3'] * O3
        L[self.IDX_O3] += k['Cl_O3'] * Cl


        rx = k['ClO_O'] * ClO * O
        P[self.IDX_Cl] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_ClO] += k['ClO_O'] * O
        L[self.IDX_O] += k['ClO_O'] * ClO


        rx = k['OH_O3'] * OH * O3
        P[self.IDX_HO2] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_OH] += k['OH_O3'] * O3
        L[self.IDX_O3] += k['OH_O3'] * OH


        rx = k['HO2_O'] * HO2 * O
        P[self.IDX_OH] += rx
        P[self.IDX_O2] += rx
        L[self.IDX_HO2] += k['HO2_O'] * O
        L[self.IDX_O] += k['HO2_O'] * HO2


        k_quench = 2.9e-11 * np.exp(-67.0 / self.T) if self.T > 0 else 2.9e-11
        P[self.IDX_O] += k_quench * O1D * M
        L[self.IDX_O1D] += k_quench * M

        return P, L

    def rhs(self, c, t=0.0, transport_source=None):
        P, L = self.production_loss(c)
        f = P - L * c[:self.N_SPECIES]
        if transport_source is not None:
            s = np.asarray(transport_source, dtype=float)
            if s.size < self.N_SPECIES:
                s = np.pad(s, (0, self.N_SPECIES - s.size))
            f += s[:self.N_SPECIES]
        return f

    def jacobian_approx(self, c, J_o2=None, J_o3=None):
        _, L = self.production_loss(c, J_o2, J_o3)
        return -L

    def step_rosenbrock(self, c, dt, transport_source=None):


        raise NotImplementedError("Hole 1: 请实现对角隐式 Rosenbrock 时间步进")

    def integrate(self, c0, t_span, dt_max=60.0, transport_source=None):
        c = np.asarray(c0, dtype=float).copy()
        t = 0.0
        t_history = [0.0]
        c_history = [c.copy()]

        while t < t_span:
            dt = dt_max
            if t + dt > t_span:
                dt = t_span - t

            if callable(transport_source):
                S = transport_source(t)
            else:
                S = transport_source



            c_new = self.step_rosenbrock(c, dt, S)
            c_half1 = self.step_rosenbrock(c, dt * 0.5, S)
            c_half2 = self.step_rosenbrock(c_half1, dt * 0.5, S)
            err = np.linalg.norm(c_new - c_half2) / (np.linalg.norm(c_new) + 1e-20)
            if err < 0.2:
                c = c_half2
                t += dt
                t_history.append(t)
                c_history.append(c.copy())
            else:
                dt_max = max(dt * 0.5, 1.0)

        return c, t_history, c_history
