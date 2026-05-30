import numpy as np
from combustion_utils import (
    check_positive, check_interval,
    arrhenius_rate, sound_speed_from_prho,
    R_UNIVERSAL, DEFAULT_GAMMA, DEFAULT_Q,
    DEFAULT_E_A, DEFAULT_A_PRE, DEFAULT_RHO_0,
    DEFAULT_P_0, DEFAULT_T_0, DEFAULT_W_MOL
)


class ZNDSolver:

    def __init__(self, gamma=DEFAULT_GAMMA, Q=DEFAULT_Q,
                 A=DEFAULT_A_PRE, Ea=DEFAULT_E_A,
                 n_order=1.0, rho0=DEFAULT_RHO_0,
                 p0=DEFAULT_P_0, T0=DEFAULT_T_0,
                 W_mol=DEFAULT_W_MOL):
        self.gamma = gamma
        self.Q = Q
        self.A = A
        self.Ea = Ea
        self.n_order = n_order
        self.rho0 = rho0
        self.p0 = p0
        self.T0 = T0
        self.W_mol = W_mol
        self.R_specific = R_UNIVERSAL / W_mol

    def cj_velocity(self):
        a0_sq = self.gamma * self.p0 / self.rho0
        D_cj_sq = 2.0 * (self.gamma ** 2 - 1.0) * self.Q + a0_sq
        if D_cj_sq <= 0.0:
            raise ValueError("CJ velocity squared non-positive")
        return np.sqrt(D_cj_sq)

    def von_neumann_state(self, D=None):
        if D is None:
            D = self.cj_velocity()
        check_positive(D, "D")
        a0 = np.sqrt(self.gamma * self.p0 / self.rho0)
        M = D / a0
        p_ratio = 1.0 + 2.0 * self.gamma / (self.gamma + 1.0) * (M * M - 1.0)
        rho_ratio = (self.gamma + 1.0) * M * M / ((self.gamma - 1.0) * M * M + 2.0)
        p_vn = self.p0 * p_ratio
        rho_vn = self.rho0 * rho_ratio
        T_vn = p_vn / (rho_vn * self.R_specific)
        u_vn = D * (1.0 - self.rho0 / rho_vn)
        return rho_vn, p_vn, T_vn, u_vn

    def _rhs(self, y, D):
        rho, u, p, lam = y
        if rho <= 0.0 or p <= 0.0:
            return np.zeros(4)
        if lam < 0.0:
            lam = 0.0
        if lam > 1.0:
            lam = 1.0

        T = p / (rho * self.R_specific)
        if T <= 0.0:
            T = 1.0e-6






        raise NotImplementedError("Hole_3: 请实现 ZND ODE 右端项的守恒关系推导")

    def solve(self, D=None, ximax=1.0e-2, npts=2000):
        if D is None:
            D = self.cj_velocity()
        check_positive(D, "D")
        check_positive(ximax, "ximax")
        check_positive(npts, "npts")

        rho_vn, p_vn, T_vn, u_vn = self.von_neumann_state(D)
        self.rho_vn = rho_vn
        self.u_vn = u_vn


        y = np.array([rho_vn, u_vn, p_vn, 0.0])
        xi = np.linspace(0.0, ximax, npts)
        sol = np.zeros((npts, 4))
        sol[0] = y

        for i in range(1, npts):
            dx = xi[i] - xi[i - 1]
            k1 = self._rhs(y, D)
            y2 = y + dx * k1

            y2[0] = max(y2[0], 1.0e-6)
            y2[2] = max(y2[2], 1.0e-6)
            y2[3] = max(0.0, min(1.0, y2[3]))
            k2 = self._rhs(y2, D)
            y = y + 0.5 * dx * (k1 + k2)
            y[0] = max(y[0], 1.0e-6)
            y[2] = max(y[2], 1.0e-6)
            y[3] = max(0.0, min(1.0, y[3]))
            sol[i] = y

        return xi, sol

    def induction_length(self, xi, sol, threshold=0.95):
        lambda_profile = sol[:, 3]
        for i in range(len(xi)):
            if lambda_profile[i] >= threshold:
                return xi[i]
        return xi[-1]

    def half_reaction_length(self, xi, sol):
        return self.induction_length(xi, sol, threshold=0.5)
