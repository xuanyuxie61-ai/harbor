
import numpy as np
from typing import Tuple
from banded_linear_algebra import BandedMatrix, SymmetricToeplitzSolver
from numerical_toolkit import TemperatureDependentProperty, muller_root
from quadrature_special import gauss_legendre_nodes_weights





FARADAY = 96485.33212
R_GAS = 8.314462618






def make_default_diffusivity_spline() -> TemperatureDependentProperty:
    T = np.array([273.15, 283.15, 298.15, 313.15, 333.15, 353.15])
    D = np.array([1.0e-15, 3.0e-15, 1.0e-14, 2.5e-14, 5.0e-14, 8.0e-14])
    return TemperatureDependentProperty(T, D)


def make_default_kappa_electrolyte_spline() -> TemperatureDependentProperty:
    T = np.array([273.15, 283.15, 298.15, 313.15, 333.15, 353.15])
    k = np.array([0.2, 0.4, 0.8, 1.2, 1.8, 2.4])
    return TemperatureDependentProperty(T, k)


def make_default_diffusivity_electrolyte_spline() -> TemperatureDependentProperty:
    T = np.array([273.15, 283.15, 298.15, 313.15, 333.15, 353.15])
    D = np.array([1.0e-11, 2.0e-11, 4.0e-11, 7.0e-11, 1.1e-10, 1.5e-10])
    return TemperatureDependentProperty(T, D)






def ocp_graphite(sto: float) -> float:
    sto = np.clip(sto, 0.01, 0.99)
    return 0.05 + 0.2 * (1.0 - sto) + 0.05 * np.sin(3.0 * np.pi * sto)


def ocp_lco(sto: float) -> float:
    sto = np.clip(sto, 0.1, 0.99)
    return 3.8 + 0.4 * (1.0 - sto) + 0.1 * np.sin(2.0 * np.pi * sto)


def d_ocp_dT_graphite(sto: float) -> float:
    return -0.0002 * (1.0 - sto)


def d_ocp_dT_lco(sto: float) -> float:
    return 0.0001 * sto






def butler_volmer_flux(eta: float, j0: float, T: float,
                       alpha_a: float = 0.5, alpha_c: float = 0.5) -> float:
    RT_F = R_GAS * T / FARADAY
    exp_a = np.exp(alpha_a * eta / RT_F)
    exp_c = np.exp(-alpha_c * eta / RT_F)
    return j0 * (exp_a - exp_c)


def butler_volmer_inverse(j_target: float, j0: float, T: float,
                          alpha_a: float = 0.5, alpha_c: float = 0.5) -> float:
    def residual(eta):
        return butler_volmer_flux(eta, j0, T, alpha_a, alpha_c) - j_target

    eta = muller_root(residual, -0.5, 0.0, 0.5, tol=1e-12, max_iter=30)
    return eta


def exchange_current_density(C_e: float, C_s_surf: float, C_s_max: float,
                             k_ref: float, T: float, E_act: float = 5000.0) -> float:
    T_ref = 298.15
    conc_term = np.sqrt(max(C_e, 1e-6) * max(C_s_surf, 1e-6)
                        * max(C_s_max - C_s_surf, 1e-6))
    arrhenius = np.exp(-E_act / R_GAS * (1.0 / T - 1.0 / T_ref))
    return k_ref * conc_term * arrhenius






class SolidDiffusionSolver:

    def __init__(self, R: float, n_r: int, D_s: float):
        self.R = R
        self.n_r = n_r
        self.D_s = D_s
        self.dr = R / n_r
        self.r = np.linspace(0.5 * self.dr, R - 0.5 * self.dr, n_r)

        self.C = np.full(n_r, 0.5 * 30555.0)

    def _build_diffusion_matrix(self, dt: float) -> BandedMatrix:
        n = self.n_r
        D = self.D_s
        dr = self.dr
        bm = BandedMatrix(n, 1, 1)
        for i in range(n):
            r_plus = self.r[i] + 0.5 * dr
            r_minus = max(self.r[i] - 0.5 * dr, 1e-12)
            r_c = self.r[i]

            if i == 0:

                a_center = 1.0 + dt * D * r_plus ** 2 / (r_c ** 2 * dr ** 2)
                a_right = -dt * D * r_plus ** 2 / (r_c ** 2 * dr ** 2)
                bm.set_entry(i, i, a_center)
                bm.set_entry(i, i + 1, a_right)
            elif i == n - 1:

                a_left = -dt * D * r_minus ** 2 / (r_c ** 2 * dr ** 2)
                a_center = 1.0 + dt * D * r_minus ** 2 / (r_c ** 2 * dr ** 2)
                bm.set_entry(i, i - 1, a_left)
                bm.set_entry(i, i, a_center)
            else:
                a_left = -dt * D * r_minus ** 2 / (r_c ** 2 * dr ** 2)
                a_center = 1.0 + dt * D * (r_plus ** 2 + r_minus ** 2) / (r_c ** 2 * dr ** 2)
                a_right = -dt * D * r_plus ** 2 / (r_c ** 2 * dr ** 2)
                bm.set_entry(i, i - 1, a_left)
                bm.set_entry(i, i, a_center)
                bm.set_entry(i, i + 1, a_right)
        return bm

    def step(self, dt: float, j_flux: float, C_s_max: float = 30555.0) -> np.ndarray:
        bm = self._build_diffusion_matrix(dt)
        rhs = self.C.copy()


        n = self.n_r
        r_c = self.r[-1]
        R = self.R
        flux_source = dt * (R ** 2 / (r_c ** 2 + 1e-18)) * (j_flux / FARADAY) / self.dr
        rhs[-1] -= flux_source
        info = bm.plu_factor()
        if info != 0:

            A = np.zeros((n, n))
            for i in range(n):
                for j in range(max(0, i - 1), min(n, i + 2)):
                    A[i, j] = bm.get_entry(i, j)
            self.C = np.linalg.solve(A, rhs)
        else:
            self.C = bm.solve(rhs)

        self.C = np.clip(self.C, 0.0, C_s_max)
        return self.C.copy()

    def surface_concentration(self) -> float:
        return float(self.C[-1])

    def average_concentration(self) -> float:

        weights = self.r ** 2
        return float(np.sum(self.C * weights) / np.sum(weights))






class MacroscopicElectrochemicalSolver:

    def __init__(self,
                 L_neg: float = 50e-6,
                 L_sep: float = 25e-6,
                 L_pos: float = 50e-6,
                 n_neg: int = 20,
                 n_sep: int = 10,
                 n_pos: int = 20,
                 T0: float = 298.15):
        self.L_neg = L_neg
        self.L_sep = L_sep
        self.L_pos = L_pos
        self.n_neg = n_neg
        self.n_sep = n_sep
        self.n_pos = n_pos
        self.n_total = n_neg + n_sep + n_pos
        self.T = T0


        x_neg = np.linspace(0.5 * L_neg / n_neg, L_neg - 0.5 * L_neg / n_neg, n_neg)
        x_sep = np.linspace(L_neg + 0.5 * L_sep / n_sep, L_neg + L_sep - 0.5 * L_sep / n_sep, n_sep)
        x_pos = np.linspace(L_neg + L_sep + 0.5 * L_pos / n_pos, L_neg + L_sep + L_pos - 0.5 * L_pos / n_pos, n_pos)
        self.x = np.concatenate([x_neg, x_sep, x_pos])


        self.regions = np.array(["neg"] * n_neg + ["sep"] * n_sep + ["pos"] * n_pos)


        self.epsilon_e = np.where(self.regions == "neg", 0.385,
                         np.where(self.regions == "sep", 0.724, 0.485))
        self.sigma_s_eff = np.where(self.regions == "neg", 100.0,
                           np.where(self.regions == "sep", 0.0, 10.0))
        self.a_s = np.where(self.regions == "neg", 885000.0,
                   np.where(self.regions == "sep", 0.0, 173000.0))
        self.C_s_max = np.where(self.regions == "neg", 30555.0, 51554.0)


        self.D_s_spline = make_default_diffusivity_spline()
        self.kappa_spline = make_default_kappa_electrolyte_spline()
        self.D_e_spline = make_default_diffusivity_electrolyte_spline()


        R_neg = 2e-6
        R_pos = 2e-6
        D_s_neg = self.D_s_spline.eval(self.T)
        D_s_pos = self.D_s_spline.eval(self.T)
        self.solid_neg = SolidDiffusionSolver(R_neg, 15, D_s_neg)
        self.solid_pos = SolidDiffusionSolver(R_pos, 15, D_s_pos)


        self.solid_neg.C = np.full(self.solid_neg.n_r, 0.5 * 30555.0)
        self.solid_pos.C = np.full(self.solid_pos.n_r, 0.5 * 51554.0)


        self.solid_solvers = []
        for r in self.regions:
            if r == "neg":
                s = SolidDiffusionSolver(R_neg, 10, D_s_neg)
                s.C = np.full(s.n_r, 0.5 * 30555.0)
                self.solid_solvers.append(s)
            elif r == "pos":
                s = SolidDiffusionSolver(R_pos, 10, D_s_pos)
                s.C = np.full(s.n_r, 0.5 * 51554.0)
                self.solid_solvers.append(s)
            else:
                self.solid_solvers.append(None)


        self.C_e = np.full(self.n_total, 1000.0)
        self.phi_s = np.zeros(self.n_total)
        self.phi_e = np.zeros(self.n_total)

    def update_temperature(self, T_new: float):
        self.T = T_new
        D_s = self.D_s_spline.eval(T_new)
        for solver in self.solid_solvers:
            if solver is not None:
                solver.D_s = D_s
        self.solid_neg.D_s = D_s
        self.solid_pos.D_s = D_s

    def _build_electrolyte_matrix(self, dt: float) -> np.ndarray:
        n = self.n_total
        A = np.zeros((n, n))
        D_e = self.D_e_spline.eval(self.T)
        dx_arr = np.diff(self.x)
        dx_arr = np.concatenate([[dx_arr[0]], dx_arr, [dx_arr[-1]]])

        for i in range(n):
            eps = self.epsilon_e[i]
            A[i, i] = eps
            if i > 0:
                dx_avg = 0.5 * (dx_arr[i] + dx_arr[i - 1])
                A[i, i] += dt * D_e / dx_avg ** 2
                A[i, i - 1] = -dt * D_e / dx_avg ** 2
            if i < n - 1:
                dx_avg = 0.5 * (dx_arr[i] + dx_arr[i + 1])
                A[i, i] += dt * D_e / dx_avg ** 2
                A[i, i + 1] = -dt * D_e / dx_avg ** 2
        return A

    def solve_electrolyte(self, dt: float, j_BV: np.ndarray, t_plus: float = 0.38) -> np.ndarray:
        A = self._build_electrolyte_matrix(dt)
        rhs = self.epsilon_e * self.C_e + dt * (1.0 - t_plus) * self.a_s * j_BV / FARADAY


        self.C_e = np.linalg.solve(A + 1e-12 * np.eye(len(A)), rhs)
        self.C_e = np.clip(self.C_e, 10.0, 5000.0)
        return self.C_e.copy()

    def solve_charge_conservation(self, I_app: float) -> np.ndarray:
        n = self.n_total
        kappa = self.kappa_spline.eval(self.T)
        phi_s = np.zeros(n)
        phi_e = np.zeros(n)
        j_BV = np.zeros(n)



        neg_mask = self.regions == "neg"
        pos_mask = self.regions == "pos"
        if np.any(neg_mask):
            x_neg = self.x[neg_mask]
            phi_s[neg_mask] = -I_app * (x_neg - x_neg[0]) / max(self.sigma_s_eff[neg_mask][0], 1e-6)
        if np.any(pos_mask):
            x_pos = self.x[pos_mask]
            phi_s[pos_mask] = I_app * (x_pos - x_pos[-1]) / max(self.sigma_s_eff[pos_mask][0], 1e-6)


        phi_e = -I_app * self.x / max(kappa, 1e-6)



        for i in range(n):
            if self.regions[i] == "sep":
                continue
            cs_surf = self.solid_solvers[i].surface_concentration() if self.solid_solvers[i] is not None else self.C_s_max[i] * 0.5
            ce_local = self.C_e[i]
            j0 = exchange_current_density(ce_local, cs_surf, self.C_s_max[i], 10.0, self.T)
            sto = cs_surf / self.C_s_max[i]
            ocp = ocp_graphite(sto) if self.regions[i] == "neg" else ocp_lco(sto)


            dx = self.x[1] - self.x[0] if n > 1 else 1e-5
            j_target = I_app / max(self.a_s[i] * dx, 1e-6)

            try:
                eta = butler_volmer_inverse(j_target, j0, self.T)
            except Exception:
                eta = 0.01
            j_BV[i] = j_target

            phi_s[i] = eta + phi_e[i] + ocp

        self.phi_s = phi_s
        self.phi_e = phi_e
        return j_BV

    def step(self, dt: float, I_app: float, T_local: float = None) -> dict:
        if T_local is not None:
            self.update_temperature(T_local)


        j_BV = self.solve_charge_conservation(I_app)


        for i in range(self.n_total):
            if self.solid_solvers[i] is not None:
                self.solid_solvers[i].step(dt, j_BV[i], self.C_s_max[i])


        self.solve_electrolyte(dt, j_BV)



        neg_conc = self.solid_solvers[0].surface_concentration() if self.solid_solvers[0] is not None else 0.5 * 30555.0
        pos_conc = self.solid_solvers[-1].surface_concentration() if self.solid_solvers[-1] is not None else 0.5 * 51554.0
        sto_neg = np.clip(neg_conc / 30555.0, 0.01, 0.99)
        sto_pos = np.clip(pos_conc / 51554.0, 0.35, 0.99)
        U_neg = ocp_graphite(sto_neg)
        U_pos = ocp_lco(sto_pos)




        V_cell = 0.0

        return {
            "voltage": float(V_cell),
            "j_BV": j_BV.copy(),
            "C_e": self.C_e.copy(),
            "phi_s": self.phi_s.copy(),
            "phi_e": self.phi_e.copy(),
            "solid_surface_conc": np.array([
                s.surface_concentration() if s is not None else 0.0
                for s in self.solid_solvers
            ])
        }
