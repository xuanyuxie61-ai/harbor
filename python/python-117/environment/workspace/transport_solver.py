
import numpy as np
from typing import Tuple


class AdvectionDiffusionSolver:

    def __init__(self, L: float = 20.0, v: float = 0.05, D: float = 0.1,
                 c0: float = 0.1, nx: int = 201):
        self.L = float(L)
        self.v = float(v)
        self.D = float(D)
        self.c0 = float(c0)
        self.nx = int(nx)
        self.dx = L / (nx - 1)

        dt_adv = self.dx / (abs(v) + 1e-12)
        dt_diff = self.dx ** 2 / (2.0 * D + 1e-12)
        self.dt = 0.4 * min(dt_adv, dt_diff)
        self.x = np.linspace(0.0, L, nx)

    def initial_condition(self, depletion_width: float = 2.0) -> np.ndarray:
        alpha = 0.3
        c = self.c0 * (1.0 - alpha * np.exp(-self.x / depletion_width))
        c = np.clip(c, 0.0, self.c0)
        return c

    def source_term(self, c: np.ndarray, t: float,
                    sink_strength: float = 0.01) -> np.ndarray:
        delta_sink = 1.0
        S = -sink_strength * c * np.exp(-self.x / delta_sink)
        return S

    def step(self, c: np.ndarray) -> np.ndarray:
        nx = self.nx
        dx = self.dx
        dt = self.dt
        v = self.v
        D = self.D
        cnew = np.zeros(nx, dtype=np.float64)

        for i in range(1, nx - 1):
            adv = -v * dt / dx * (c[i] - c[i - 1])
            diff = D * dt / dx ** 2 * (c[i - 1] - 2.0 * c[i] + c[i + 1])
            cnew[i] = c[i] + adv + diff

        S = self.source_term(c, 0.0)
        cnew[1:nx - 1] += dt * S[1:nx - 1]

        cnew[0] = self.c0 * 0.7
        cnew[-1] = self.c0

        cnew = np.clip(cnew, 0.0, None)
        return cnew

    def solve(self, n_steps: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        c = self.initial_condition()
        snap_every = max(1, n_steps // 50)
        history = []
        for step in range(n_steps):
            c = self.step(c)
            if step % snap_every == 0:
                history.append(c.copy())
        history = np.array(history)
        return c, history

    def compute_flux(self, c: np.ndarray) -> float:
        dc_dx = (c[1] - c[0]) / self.dx
        J = -self.D * dc_dx + self.v * c[0]
        return float(J)
