
import numpy as np
from typing import Tuple, Optional


class SpatialDiscretization1D:

    def __init__(self,
                 x: np.ndarray,
                 epsilon: float = 0.01,
                 velocity: float = 0.5,
                 reaction_rate: float = 1.0,
                 carrying_capacity: float = 1.0):
        if x.ndim != 1 or len(x) < 3:
            raise ValueError("x must be 1D with length >= 3")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        self.x = x.copy()
        self.nx = len(x)
        self.dx = np.diff(x)
        self.epsilon = epsilon
        self.v = velocity
        self.r = reaction_rate
        self.K = carrying_capacity


        self.pe_local = self._compute_local_peclet()

    def _compute_local_peclet(self) -> np.ndarray:
        h = np.zeros(self.nx, dtype=np.float64)
        h[0] = self.dx[0]
        h[-1] = self.dx[-1]
        h[1:-1] = 0.5 * (self.dx[:-1] + self.dx[1:])
        pe = np.abs(self.v) * h / self.epsilon
        return pe

    def diffusion_operator(self, u: np.ndarray) -> np.ndarray:
        if len(u) != self.nx:
            raise ValueError("u length mismatch")
        d2u = np.zeros(self.nx, dtype=np.float64)
        for i in range(1, self.nx - 1):
            hp = self.x[i + 1] - self.x[i]
            hm = self.x[i] - self.x[i - 1]
            if hp <= 0 or hm <= 0:
                raise ValueError("Non-positive mesh spacing detected")
            d2u[i] = 2.0 / (hp + hm) * ((u[i + 1] - u[i]) / hp - (u[i] - u[i - 1]) / hm)

        d2u[0] = d2u[1]
        d2u[-1] = d2u[-2]
        return self.epsilon * d2u

    def advection_operator(self, u: np.ndarray, scheme: str = "auto") -> np.ndarray:
        if len(u) != self.nx:
            raise ValueError("u length mismatch")
        adv = np.zeros(self.nx, dtype=np.float64)

        for i in range(1, self.nx - 1):
            hp = self.x[i + 1] - self.x[i]
            hm = self.x[i] - self.x[i - 1]
            pe = self.pe_local[i]
            sel = scheme
            if sel == "auto":
                sel = "centered" if pe < 2.0 else "lax_wendroff"

            if sel == "centered":
                du = (u[i + 1] - u[i - 1]) / (hp + hm)
            elif sel == "upwind":
                if self.v > 0:
                    du = (u[i] - u[i - 1]) / hm
                else:
                    du = (u[i + 1] - u[i]) / hp
            elif sel == "lax_wendroff":

                du_center = (u[i + 1] - u[i - 1]) / (hp + hm)

                nu = np.abs(self.v) * max(hp, hm) * 0.5
                diff_artificial = nu * ((u[i + 1] - u[i]) / hp - (u[i] - u[i - 1]) / hm) / (0.5 * (hp + hm))
                du = du_center - diff_artificial / self.v if self.v != 0 else du_center
            else:
                du = (u[i + 1] - u[i - 1]) / (hp + hm)
            adv[i] = self.v * du


        adv[0] = adv[1]
        adv[-1] = adv[-2]
        return adv

    def reaction_operator(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, None)
        return self.r * u * (1.0 - u / self.K)

    def full_rhs_deterministic(self, u: np.ndarray, scheme: str = "auto") -> np.ndarray:
        return self.diffusion_operator(u) - self.advection_operator(u, scheme=scheme) + self.reaction_operator(u)

    def assemble_fem_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.nx
        M = np.zeros(n, dtype=np.float64)
        K = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            if i == 0:
                h = self.dx[0]
                M[i] = 0.5 * h
                K[i, i] = self.epsilon / h + abs(self.v) / 2.0
                K[i, i + 1] = -self.epsilon / h - self.v / 2.0 * (1.0 + np.sign(self.v))
            elif i == n - 1:
                h = self.dx[-1]
                M[i] = 0.5 * h
                K[i, i] = self.epsilon / h + abs(self.v) / 2.0
                K[i, i - 1] = -self.epsilon / h + self.v / 2.0 * (1.0 - np.sign(self.v))
            else:
                hm = self.dx[i - 1]
                hp = self.dx[i]
                M[i] = 0.5 * (hm + hp)
                K[i, i] = self.epsilon * (1.0 / hm + 1.0 / hp) + abs(self.v) / 2.0
                K[i, i + 1] = -self.epsilon / hp - self.v / 2.0 * (1.0 + np.sign(self.v))
                K[i, i - 1] = -self.epsilon / hm + self.v / 2.0 * (1.0 - np.sign(self.v))

        return M, K

    def dg_numerical_flux(self, u_left: float, u_right: float) -> float:
        f_L = self.v * u_left
        f_R = self.v * u_right
        alpha = abs(self.v)
        flux = 0.5 * (f_L + f_R) - 0.5 * alpha * (u_right - u_left)
        return flux
