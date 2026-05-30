
import numpy as np
from typing import Tuple, Callable, Optional


class MicroreactorPDESolver:

    def __init__(
        self,
        L: float = 0.1,
        Nx: int = 200,
        D_m: float = 1e-9,
        u: float = 0.01,
        A_arr: float = 1.0e8,
        Ea: float = 50000.0,
        R_gas: float = 8.314,
        reaction_order: float = 1.0,
        rho: float = 1000.0,
        cp: float = 4180.0,
        lam: float = 0.6,
        dH: float = -8.0e4,
        T_wall: float = 350.0,
        h_wall: float = 500.0,
        hydraulic_diameter: float = 5.0e-4,
        C_in: float = 1000.0,
        T_in: float = 300.0,
    ):

        if L <= 0.0:
            raise ValueError("通道长度 L 必须大于 0")
        if Nx < 3:
            raise ValueError("空间网格数 Nx 至少为 3")
        if D_m <= 0.0 or u <= 0.0:
            raise ValueError("扩散系数 D_m 与流速 u 必须为正")
        if Ea <= 0.0 or A_arr <= 0.0:
            raise ValueError("Arrhenius 参数必须为正")

        self.L = L
        self.Nx = Nx
        self.dx = L / (Nx - 1)
        self.D_m = D_m
        self.u = u
        self.A_arr = A_arr
        self.Ea = Ea
        self.R_gas = R_gas
        self.n_order = reaction_order
        self.rho = rho
        self.cp = cp
        self.lam = lam
        self.dH = dH
        self.T_wall = T_wall
        self.h_wall = h_wall
        self.d_h = hydraulic_diameter
        self.C_in = C_in
        self.T_in = T_in


        self.Pe_m = u * self.dx / D_m
        self.Pe_t = u * self.dx / (lam / (rho * cp))


        self.dt_adv = 0.5 * self.dx / u
        self.dt_diff = 0.25 * self.dx**2 / max(D_m, lam / (rho * cp))

    def reaction_rate(self, C: np.ndarray, T: np.ndarray) -> np.ndarray:
        C_safe = np.where(C > 0.0, C, 0.0)
        T_safe = np.where(T > 0.0, T, 1.0)



        rate = np.zeros_like(C_safe)
        rate = np.where(rate < 1.0e6, rate, 1.0e6)
        return rate

    def _build_tridiagonal_matrix(
        self, a: float, b: float, c: float, n: int
    ) -> np.ndarray:
        mat = np.zeros((n, n))
        np.fill_diagonal(mat, b)
        np.fill_diagonal(mat[1:, :-1], a)
        np.fill_diagonal(mat[:-1, 1:], c)
        return mat

    def _solve_tridiagonal(
        self, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
    ) -> np.ndarray:
        n = len(diag)
        if n == 0:
            return rhs.copy()
        cp = upper.copy()
        dp = rhs.copy()
        cp[0] /= diag[0]
        dp[0] /= diag[0]
        for i in range(1, n):
            denom = diag[i] - lower[i] * cp[i - 1]
            if abs(denom) < 1.0e-14:
                denom = 1.0e-14
            cp[i] = (upper[i] / denom) if i < n - 1 else 0.0
            dp[i] = (dp[i] - lower[i] * dp[i - 1]) / denom
        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def solve_steady_state(
        self, max_iter: int = 5000, tol: float = 1.0e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        C = np.full(self.Nx, self.C_in, dtype=float)
        T = np.full(self.Nx, self.T_in, dtype=float)


        C[0] = self.C_in
        T[0] = self.T_in

        alpha_m = self.D_m / (self.u * self.dx)
        alpha_t = self.lam / (self.rho * self.cp * self.u * self.dx)
        beta = 4.0 * self.h_wall / (self.rho * self.cp * self.u * self.d_h)

        for it in range(max_iter):
            C_old = C.copy()
            T_old = T.copy()


            for i in range(1, self.Nx - 1):
                r = self.reaction_rate(np.array([C_old[i]]), np.array([T_old[i]]))[0]







                conv_c = C_old[i] - C_old[i - 1]
                diff_c = alpha_m * (C_old[i + 1] - 2.0 * C_old[i] + C_old[i - 1])
                source_c = -r * self.dx / self.u

                C[i] = C_old[i] + 0.5 * (
                    conv_c + diff_c + source_c
                    + (C_old[i] - C_old[i - 1])
                    + alpha_m * (C_old[i + 1] - 2.0 * C_old[i] + C_old[i - 1])
                    + (-self.reaction_rate(np.array([C[i]]), np.array([T[i]]))[0] * self.dx / self.u)
                )



            for i in range(1, self.Nx - 1):





                C_prev_iter = C[i]
                T_prev_iter = T[i]
                for inner in range(10):
                    r_val = self.reaction_rate(
                        np.array([C_prev_iter]), np.array([T_prev_iter])
                    )[0]

                    conv = self.u * (C_prev_iter - C[i - 1]) / self.dx
                    diff = -self.D_m * (C[i + 1] - 2.0 * C_prev_iter + C[i - 1]) / (self.dx ** 2)
                    f_c = conv + diff + r_val

                    dr_dc = r_val * self.n_order / max(C_prev_iter, 1.0e-12)
                    jac_c = self.u / self.dx + 2.0 * self.D_m / (self.dx ** 2) + dr_dc
                    delta_c = -f_c / max(jac_c, 1.0e-12)
                    C_prev_iter = max(C_prev_iter + delta_c, 0.0)


                    conv_t = self.rho * self.cp * self.u * (T_prev_iter - T[i - 1]) / self.dx
                    diff_t = -self.lam * (T[i + 1] - 2.0 * T_prev_iter + T[i - 1]) / (self.dx ** 2)
                    source_t = -self.dH * r_val
                    wall_loss = 4.0 * self.h_wall * (T_prev_iter - self.T_wall) / self.d_h
                    f_t = conv_t + diff_t + source_t + wall_loss
                    dr_dt = r_val * self.Ea / (self.R_gas * T_prev_iter ** 2)
                    jac_t = (
                        self.rho * self.cp * self.u / self.dx
                        + 2.0 * self.lam / (self.dx ** 2)
                        - self.dH * dr_dt
                        + 4.0 * self.h_wall / self.d_h
                    )
                    delta_t = -f_t / max(jac_t, 1.0e-12)
                    T_prev_iter = max(T_prev_iter + delta_t, 200.0)

                    if abs(delta_c) < 1.0e-10 and abs(delta_t) < 1.0e-10:
                        break

                C[i] = C_prev_iter
                T[i] = T_prev_iter


            C[-1] = C[-2]
            T[-1] = T[-2]


            diff_norm = np.max(np.abs(C - C_old)) + np.max(np.abs(T - T_old))
            if diff_norm < tol:
                break

        return C, T

    def compute_conversion_and_yield(self, C: np.ndarray) -> Tuple[float, float]:
        C_out = C[-1]
        X = (self.C_in - C_out) / max(self.C_in, 1.0e-12)
        X = max(0.0, min(1.0, X))
        return X, C_out

    def compute_peclet_damkohler(self, C: np.ndarray, T: np.ndarray) -> Tuple[float, float]:
        Pe = self.u * self.L / self.D_m
        T_avg = np.mean(T)
        Da = self.A_arr * np.exp(-self.Ea / (self.R_gas * T_avg)) * self.L / self.u
        return Pe, Da
