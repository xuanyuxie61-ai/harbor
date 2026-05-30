
import numpy as np
from typing import Callable


class PDE1DHeatExplicit:

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def solve(self, x_grid: np.ndarray, u0: np.ndarray,
              dt: float, n_steps: int,
              source: Callable | None = None) -> np.ndarray:
        n = len(x_grid)
        if len(u0) != n:
            raise ValueError("u0 length must match x_grid length.")
        if n < 3:
            raise ValueError("Grid must have at least 3 points.")


        h = np.diff(x_grid)
        h_min = h.min()


        cfl_limit = h_min ** 2 / (2.0 * self.kappa)
        if dt > cfl_limit:

            dt = cfl_limit * 0.9
            n_steps = max(int(n_steps * cfl_limit / dt), 1)


        M_diag = np.zeros(n)
        M_diag[0] = h[0] / 2.0
        M_diag[-1] = h[-1] / 2.0
        for i in range(1, n - 1):
            M_diag[i] = (h[i - 1] + h[i]) / 2.0


        K_diag = np.zeros(n)
        K_off = np.zeros(n - 1)
        for i in range(n - 1):
            inv_h = 1.0 / h[i]
            K_off[i] = -self.kappa * inv_h
            K_diag[i] += self.kappa * inv_h
            K_diag[i + 1] += self.kappa * inv_h

        u = u0.copy()
        for step in range(n_steps):

            Ku = np.zeros(n)
            Ku[0] = K_diag[0] * u[0] + K_off[0] * u[1]
            for i in range(1, n - 1):
                Ku[i] = K_off[i - 1] * u[i - 1] + K_diag[i] * u[i] + K_off[i] * u[i + 1]
            Ku[-1] = K_off[-1] * u[-2] + K_diag[-1] * u[-1]

            rhs = -Ku
            if source is not None:
                rhs += source(x_grid, step * dt, u)


            du = dt * rhs / M_diag
            u = u + du

        return u


class ReactionDiffusion1D:

    def __init__(self, D: float = 0.1, rho: float = 1.0, K: float = 1.0,
                 mu: float = 0.5, c_s: float = 0.1):
        self.D = D
        self.rho = rho
        self.K = K
        self.mu = mu
        self.c_s = c_s

    def reaction(self, u: np.ndarray) -> np.ndarray:
        logistic = self.rho * u * (1.0 - u / self.K)

        consumption = self.mu * u / (self.c_s + np.abs(u) + 1e-12)
        return logistic - consumption

    def solve(self, x_grid: np.ndarray, u0: np.ndarray,
              dt: float, n_steps: int, scheme: str = "heun") -> np.ndarray:
        n = len(x_grid)
        if len(u0) != n:
            raise ValueError("u0 length mismatch.")
        h = np.diff(x_grid)
        h_min = h.min()


        cfl_diff = h_min ** 2 / (2.0 * self.D) if self.D > 0 else np.inf

        cfl_react = 1.0 / abs(self.rho) if self.rho != 0 else np.inf
        dt_limit = min(cfl_diff, cfl_react)
        if dt > dt_limit:
            dt = dt_limit * 0.5

        def laplacian(u):
            Lu = np.zeros(n)

            for i in range(1, n - 1):
                hp = x_grid[i + 1] - x_grid[i]
                hm = x_grid[i] - x_grid[i - 1]

                Lu[i] = 2.0 / (hp + hm) * ((u[i + 1] - u[i]) / hp - (u[i] - u[i - 1]) / hm)

            Lu[0] = Lu[1]
            Lu[-1] = Lu[-2]
            return Lu

        u = u0.copy()
        if scheme == "euler":
            for _ in range(n_steps):
                Lu = laplacian(u)
                R = self.reaction(u)
                u = u + dt * (self.D * Lu + R)
        elif scheme == "heun":
            for _ in range(n_steps):
                Lu = laplacian(u)
                R = self.reaction(u)
                k1 = dt * (self.D * Lu + R)
                u_temp = u + k1
                Lu2 = laplacian(u_temp)
                R2 = self.reaction(u_temp)
                k2 = dt * (self.D * Lu2 + R2)
                u = u + 0.5 * (k1 + k2)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        return u
