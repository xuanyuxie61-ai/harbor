
import numpy as np
from typing import Callable, Tuple, Optional


class ReactionDiffusionFTCS:

    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                 D: float = 1.0, dt: Optional[float] = None):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.D = D


        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        self.dt_max = 0.5 / (D * (1.0 / dx2 + 1.0 / dy2))
        if dt is None:
            self.dt = 0.25 * self.dt_max
        else:
            if dt > self.dt_max:
                raise ValueError(
                    f"时间步长 dt={dt} 超过 FTCS 稳定性限制 dt_max={self.dt_max:.6e}"
                )
            self.dt = dt


        self.cx = D * self.dt / dx2
        self.cy = D * self.dt / dy2

    def laplacian_2d(self, u: np.ndarray) -> np.ndarray:
        if u.shape != (self.ny, self.nx):
            raise ValueError(f"u 形状应为 ({self.ny}, {self.nx})")

        Lu = np.zeros_like(u)

        Lu[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dx ** 2 +
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dy ** 2
        )



        Lu[:, 0] = (2.0 * u[:, 1] - 2.0 * u[:, 0]) / self.dx ** 2 + (
            np.concatenate([[0], (u[2:, 0] - 2.0 * u[1:-1, 0] + u[:-2, 0]), [0]]) / self.dy ** 2
        )

        Lu[:, -1] = (2.0 * u[:, -2] - 2.0 * u[:, -1]) / self.dx ** 2 + (
            np.concatenate([[0], (u[2:, -1] - 2.0 * u[1:-1, -1] + u[:-2, -1]), [0]]) / self.dy ** 2
        )

        Lu[0, :] = (np.concatenate([[0], (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]), [0]]) / self.dx ** 2 +
                    (2.0 * u[1, :] - 2.0 * u[0, :]) / self.dy ** 2)

        Lu[-1, :] = (np.concatenate([[0], (u[-1, 2:] - 2.0 * u[-1, 1:-1] + u[-1, :-2]), [0]]) / self.dx ** 2 +
                     (2.0 * u[-2, :] - 2.0 * u[-1, :]) / self.dy ** 2)

        return Lu

    def step(self, u: np.ndarray, reaction: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        Lu = self.laplacian_2d(u)
        R = reaction(u)

        R[0, :] = 0.0
        R[-1, :] = 0.0
        R[:, 0] = 0.0
        R[:, -1] = 0.0

        u_new = u + self.dt * (self.D * Lu + R)


        u_max = np.max(np.abs(u))
        if u_max > 0:
            clip_val = u_max * 1e6
            u_new = np.clip(u_new, -clip_val, clip_val)


        if not np.all(np.isfinite(u_new)):

            mask = ~np.isfinite(u_new)
            u_new[mask] = u[mask] * 0.9

        return u_new

    def solve(self, u0: np.ndarray, reaction: Callable[[np.ndarray], np.ndarray],
              nsteps: int, callback: Optional[Callable] = None) -> np.ndarray:
        u = u0.copy()
        for step in range(nsteps):
            u = self.step(u, reaction)
            if callback is not None:
                callback(step, u)
        return u


def fisher_kpp_reaction(u: np.ndarray, r: float = 1.0, K: float = 1.0) -> np.ndarray:
    return r * u * (1.0 - u / K)


def allen_cahn_reaction(u: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    return (u - u ** 3) / (epsilon ** 2 + 1e-20)


def coupled_reaction_diffusion_step(
    P: np.ndarray, M: np.ndarray,
    solver_P: ReactionDiffusionFTCS,
    solver_M: ReactionDiffusionFTCS,
    reaction_P: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reaction_M: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    P_new = solver_P.step(P, lambda u: reaction_P(u, M))
    M_new = solver_M.step(M, lambda u: reaction_M(P, u))
    return P_new, M_new
