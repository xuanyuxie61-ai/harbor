
import math
from typing import List, Tuple


class ArteryFlowModel:

    def __init__(self, alpha: float, beta: float, gamma: float,
                 a: float, b: float, omega: float, x: float = 1.0, dp_dx: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.omega = omega
        self.x = x
        self.dp_dx = dp_dx

    def forcing(self, t: float) -> float:
        return self.gamma * self.x * self.dp_dx * (self.a + self.b * math.cos(self.omega * t))

    def rhs(self, t: float, state: List[float]) -> List[float]:
        u, v = state
        force = self.forcing(t)
        return [v, -self.alpha * u - self.beta * v + force]

    def analytical_amplitude(self) -> float:
        denom = math.sqrt((self.alpha - self.omega ** 2) ** 2 + (self.beta * self.omega) ** 2)
        if denom < 1e-15:
            return 0.0
        return abs(self.gamma * self.x * self.dp_dx * self.b) / denom

    def simulate_euler(self, u0: float, v0: float, t_end: float, n_steps: int) -> Tuple[List[float], List[float], List[float]]:
        h = t_end / n_steps
        t_vals = [i * h for i in range(n_steps + 1)]
        u_vals = [u0]
        v_vals = [v0]
        u, v = u0, v0
        for i in range(n_steps):
            t = t_vals[i]
            du, dv = self.rhs(t, [u, v])
            u += h * du
            v += h * dv

            u = max(u, 0.0)
            u_vals.append(u)
            v_vals.append(v)
        return t_vals, u_vals, v_vals


class LaxWendroffBuffer:

    def __init__(self, nx: int, c: float, dx: float, dt: float):
        self.nx = nx
        self.c = c
        self.dx = dx
        self.dt = dt
        self.nu = c * dt / dx
        if abs(self.nu) > 1.0:

            self.nu = math.copysign(1.0, self.nu) * 0.95
            self.dt = self.nu * dx / c if abs(c) > 1e-15 else dt

    def step(self, rho: List[float]) -> List[float]:
        if len(rho) != self.nx:
            raise ValueError(f"rho length {len(rho)} != nx {self.nx}")
        rho_new = [0.0] * self.nx
        nu = self.nu
        nu2 = nu * nu
        nx = self.nx

        for j in range(nx):
            jm = (j - 1) % nx
            jp = (j + 1) % nx
            rho_new[j] = (
                rho[j]
                - 0.5 * nu * (rho[jp] - rho[jm])
                + 0.5 * nu2 * (rho[jp] - 2.0 * rho[j] + rho[jm])
            )
        return rho_new

    def simulate(self, rho0: List[float], n_steps: int) -> List[List[float]]:
        history = [list(rho0)]
        rho = list(rho0)
        for _ in range(n_steps):
            rho = self.step(rho)
            history.append(list(rho))
        return history

    def compute_courant_number(self) -> float:
        return abs(self.c) * self.dt / self.dx


def predict_optimal_buffer_size(alpha: float, beta: float, gamma: float,
                                omega: float, safety_factor: float = 1.5) -> float:
    if alpha < 1e-15:
        return 1e6
    steady_offset = gamma / alpha
    model = ArteryFlowModel(alpha, beta, gamma, 1.0, 1.0, omega)
    amp = model.analytical_amplitude()
    return safety_factor * (steady_offset + amp)
