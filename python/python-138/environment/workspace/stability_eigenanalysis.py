
import numpy as np
from typing import Tuple, Optional


class ReactorStabilityAnalyzer:

    def __init__(
        self,
        Nx: int = 100,
        L: float = 0.1,
        D_m: float = 1e-9,
        alpha: float = 1.43e-7,
        u: float = 0.01,
        dH: float = -8.0e4,
        rho: float = 1000.0,
        cp: float = 4180.0,
        h_wall: float = 500.0,
        d_h: float = 5.0e-4,
        R_gas: float = 8.314,
    ):
        if Nx < 4:
            raise ValueError("Nx 至少为 4")
        self.Nx = Nx
        self.L = L
        self.dx = L / (Nx - 1)
        self.D_m = D_m
        self.alpha = alpha
        self.u = u
        self.dH = dH
        self.rho = rho
        self.cp = cp
        self.h_wall = h_wall
        self.d_h = d_h
        self.R_gas = R_gas

    def _build_advection_diffusion_operator(
        self, diff_coeff: float
    ) -> np.ndarray:
        n = self.Nx
        A = np.zeros((n, n))
        dx = self.dx
        Pe_local = self.u * dx / diff_coeff

        for i in range(1, n - 1):

            adv = -self.u / dx
            A[i, i] += adv
            A[i, i - 1] += -adv


            diff = diff_coeff / (dx ** 2)
            A[i, i - 1] += diff
            A[i, i] += -2.0 * diff
            A[i, i + 1] += diff


        A[0, :] = 0.0
        A[0, 0] = -1.0


        A[n - 1, :] = 0.0
        A[n - 1, n - 1] = -1.0
        A[n - 1, n - 2] = 1.0

        return A

    def compute_jacobian(
        self,
        C_steady: np.ndarray,
        T_steady: np.ndarray,
        A_arr: float,
        Ea: float,
        n_order: float,
    ) -> np.ndarray:
        if len(C_steady) != self.Nx or len(T_steady) != self.Nx:
            raise ValueError("稳态场维度与 Nx 不匹配")

        n = self.Nx
        A_c = self._build_advection_diffusion_operator(self.D_m)
        A_t = self._build_advection_diffusion_operator(self.alpha)


        C_safe = np.maximum(C_steady, 1.0e-12)
        T_safe = np.maximum(T_steady, 200.0)




        r_base = np.zeros(n)
        dr_dC = np.zeros(n)
        dr_dT = np.zeros(n)


        beta_wall = 4.0 * self.h_wall / (self.rho * self.cp * self.d_h)


        J = np.zeros((2 * n, 2 * n))

        J[:n, :n] = A_c - np.diag(dr_dC)

        J[:n, n:] = -np.diag(dr_dT)

        factor = -self.dH / (self.rho * self.cp)
        J[n:, :n] = factor * np.diag(dr_dC)

        J[n:, n:] = A_t + factor * np.diag(dr_dT) - np.diag(np.full(n, beta_wall))

        return J

    def analyze_stability(
        self,
        C_steady: np.ndarray,
        T_steady: np.ndarray,
        A_arr: float,
        Ea: float,
        n_order: float,
    ) -> Tuple[np.ndarray, float, bool]:
        J = self.compute_jacobian(C_steady, T_steady, A_arr, Ea, n_order)
        eigenvalues = np.linalg.eigvals(J)
        max_real = np.max(np.real(eigenvalues))
        is_stable = max_real < -1.0e-10
        return eigenvalues, max_real, is_stable

    def compute_critical_damkohler_bracket(
        self,
        C_steady_base: np.ndarray,
        T_steady_base: np.ndarray,
        A_arr_base: float,
        Ea: float,
        n_order: float,
        da_min: float = 0.01,
        da_max: float = 10.0,
        n_scan: int = 20,
    ) -> Tuple[float, float, float]:
        if da_min >= da_max or da_min <= 0.0:
            raise ValueError("da_min 必须为正且小于 da_max")

        Da_vals = np.logspace(np.log10(da_min), np.log10(da_max), n_scan)
        max_reals = np.zeros(n_scan)

        for i, Da in enumerate(Da_vals):
            A_scaled = Da * A_arr_base / (A_arr_base * self.L / self.u)
            A_scaled = A_arr_base * Da / (A_arr_base * self.L / self.u)

            A_scaled = A_arr_base * Da
            _, max_reals[i], _ = self.analyze_stability(
                C_steady_base, T_steady_base, A_scaled, Ea, n_order
            )


        stable_mask = max_reals < 0
        if np.all(stable_mask):
            return Da_vals[-1], Da_vals[-1] * 10.0, max_reals[-1]
        if not np.any(stable_mask):
            return Da_vals[0] / 10.0, Da_vals[0], max_reals[0]

        idx = np.where(~stable_mask)[0][0]
        if idx == 0:
            return Da_vals[0], Da_vals[1], max_reals[0]
        return Da_vals[idx - 1], Da_vals[idx], max_reals[idx]

    def compute_thermal_explosion_index(self, max_real: float) -> float:
        tau_res = self.L / max(self.u, 1.0e-12)
        risk = np.tanh(max(0.0, max_real) * tau_res)
        return risk
