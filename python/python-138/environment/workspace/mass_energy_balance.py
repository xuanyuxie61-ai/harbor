
import numpy as np
from typing import Tuple, Optional, Callable


class MassEnergyBalanceSolver:

    def __init__(
        self,
        C0: float = 1000.0,
        T0: float = 300.0,
        Q: float = 1.0e-6,
        V: float = 1.0e-5,
        rho: float = 1000.0,
        cp: float = 4180.0,
        dH: float = -8.0e4,
        Ua: float = 0.5,
        Tc: float = 350.0,
        A_arr: float = 1.0e8,
        Ea: float = 50000.0,
        R_gas: float = 8.314,
        reaction_order: float = 1.0,
    ):
        if Q <= 0.0 or V <= 0.0:
            raise ValueError("流量 Q 与体积 V 必须为正")
        if C0 <= 0.0 or T0 <= 0.0:
            raise ValueError("入口浓度与温度必须为正")

        self.C0 = C0
        self.T0 = T0
        self.Q = Q
        self.V = V
        self.rho = rho
        self.cp = cp
        self.dH = dH
        self.Ua = Ua
        self.Tc = Tc
        self.A_arr = A_arr
        self.Ea = Ea
        self.R_gas = R_gas
        self.n_order = reaction_order
        self.tau = V / Q

    def rate_constant(self, T: float) -> float:
        if T <= 0.0:
            return 0.0


        k = 0.0
        return min(k, 1.0e12)

    def concentration_from_T(self, T: float) -> float:
        k = self.rate_constant(T)
        if self.n_order == 1.0:
            C = self.C0 / (1.0 + k * self.tau)
        else:

            C = self.C0 / (1.0 + k * self.tau)
            for _ in range(20):
                f = self.C0 - C - k * (C ** self.n_order) * self.tau
                df = -1.0 - k * self.n_order * (C ** max(self.n_order - 1.0, 0.0)) * self.tau
                if abs(df) < 1.0e-14:
                    break
                delta = -f / df
                C_new = max(C + delta, 0.0)
                if abs(C_new - C) < 1.0e-12:
                    break
                C = C_new
        return max(C, 0.0)

    def G_map(self, T: float) -> float:
        C = self.concentration_from_T(T)
        numerator = (
            self.rho * self.cp * self.Q * self.T0
            + (-self.dH) * self.Q * self.C0
            + self.Ua * self.Tc
            - (-self.dH) * self.Q * C
        )
        denominator = self.rho * self.cp * self.Q + self.Ua
        if abs(denominator) < 1.0e-14:
            return T
        return numerator / denominator

    def solve_fixed_point(
        self,
        T_guess: Optional[float] = None,
        max_iter: int = 1000,
        tol: float = 1.0e-10,
    ) -> Tuple[float, float, int, bool]:
        if T_guess is None:
            T_guess = 0.5 * (self.T0 + self.Tc)
        T = max(T_guess, 200.0)

        for it in range(1, max_iter + 1):
            T_new = self.G_map(T)

            T_new = 0.7 * T_new + 0.3 * T
            T_new = max(T_new, 200.0)
            T_new = min(T_new, 2000.0)

            diff = abs(T_new - T)
            T = T_new
            if diff < tol:
                C_ss = self.concentration_from_T(T)
                return T, C_ss, it, True

        C_ss = self.concentration_from_T(T)
        return T, C_ss, max_iter, False

    def multi_start_solve(
        self, n_starts: int = 5
    ) -> Tuple[float, float, bool]:
        T_starts = np.linspace(self.T0, self.Tc + 100.0, n_starts)
        best_residual = float("inf")
        best_result = (self.T0, self.C0, False)

        for T0_g in T_starts:
            T_ss, C_ss, it, conv = self.solve_fixed_point(T_guess=T0_g)

            k = self.rate_constant(T_ss)
            r = k * (C_ss ** self.n_order)
            res_mass = abs(self.Q * (self.C0 - C_ss) - self.V * r)
            res_energy = abs(
                self.rho * self.cp * self.Q * (self.T0 - T_ss)
                + (-self.dH) * self.V * r
                - self.Ua * (T_ss - self.Tc)
            )
            residual = res_mass + res_energy
            if residual < best_residual:
                best_residual = residual
                best_result = (T_ss, C_ss, conv)

        return best_result

    def compute_multiplicity_diagram(
        self,
        Tc_range: Tuple[float, float] = (280.0, 400.0),
        n_points: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Tc_vals = np.linspace(Tc_range[0], Tc_range[1], n_points)
        T_ss_vals = np.zeros(n_points)
        C_ss_vals = np.zeros(n_points)
        orig_Tc = self.Tc
        for i, Tc in enumerate(Tc_vals):
            self.Tc = Tc
            T_ss, C_ss, _ = self.multi_start_solve(n_starts=7)
            T_ss_vals[i] = T_ss
            C_ss_vals[i] = C_ss
        self.Tc = orig_Tc
        return Tc_vals, T_ss_vals, C_ss_vals

    def bifurcation_indicator(self) -> float:
        T_ss, _, _, _ = self.solve_fixed_point()
        dT = 0.01
        Gp = self.G_map(T_ss + dT)
        Gm = self.G_map(T_ss - dT)
        dGdT = abs((Gp - Gm) / (2.0 * dT))
        return dGdT
