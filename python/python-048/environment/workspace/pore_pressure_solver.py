
import numpy as np
from typing import Tuple, Callable


def laplacian_1d_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    N = u.size
    if N < 2:
        return np.zeros_like(u)
    uxx = np.empty_like(u)
    uxx[0] = (u[1] - u[0]) / (dx * dx)
    uxx[1:N - 1] = (u[0:N - 2] - 2.0 * u[1:N - 1] + u[2:N]) / (dx * dx)
    uxx[N - 1] = (u[N - 2] - u[N - 1]) / (dx * dx)
    return uxx


def euler_integrate(dydt: Callable, tspan: Tuple[float, float],
                    y0: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    t0, tstop = tspan
    dt = (tstop - t0) / n_steps

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    t[0] = t0
    y[0, :] = y0

    for k in range(n_steps):
        t[k + 1] = t[k] + dt
        deriv = np.asarray(dydt(t[k], y[k, :]), dtype=float)
        if deriv.size != m:
            raise ValueError(f"dydt 返回维度 {deriv.size} 与状态维度 {m} 不匹配")
        y[k + 1, :] = y[k, :] + dt * deriv

    return t, y


def pore_pressure_diffusion_rhs(t: float, p: np.ndarray,
                                 D: float, dx: float,
                                 source_idx: int,
                                 source_rate: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    dpdt = D * laplacian_1d_neumann(p, dx)

    if 0 <= source_idx < p.size:
        dpdt[source_idx] += source_rate
    return dpdt


class PorePressureSolver:

    def __init__(self, x_min: float, x_max: float, nx: int,
                 D: float, source_rate: float, source_x: float):
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
        self.D = D
        self.source_rate = source_rate
        self.source_idx = int(np.argmin(np.abs(self.x - source_x)))
        self.p0 = np.zeros(nx)

    def rhs(self, t: float, p: np.ndarray) -> np.ndarray:
        return pore_pressure_diffusion_rhs(t, p, self.D, self.dx,
                                            self.source_idx, self.source_rate)

    def solve(self, tspan: Tuple[float, float], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        dt = (tspan[1] - tspan[0]) / n_steps
        cfl_limit = self.dx ** 2 / (2.0 * self.D) if self.D > 0 else np.inf
        if dt > cfl_limit:

            n_steps = max(n_steps, int(np.ceil((tspan[1] - tspan[0]) / (0.9 * cfl_limit))))
        return euler_integrate(self.rhs, tspan, self.p0, n_steps)

    def cavity_flow_velocity(self, p: np.ndarray, w_aperture: float,
                              mu_fluid: float = 1.0e-3) -> np.ndarray:
        if p.size != self.x.size:
            raise ValueError("压力场维度与空间网格不匹配")
        grad_p = np.zeros_like(p)
        nx = p.size
        if nx > 1:

            grad_p[1:nx - 1] = (p[2:nx] - p[0:nx - 2]) / (2.0 * self.dx)
            grad_p[0] = (p[1] - p[0]) / self.dx
            grad_p[nx - 1] = (p[nx - 1] - p[nx - 2]) / self.dx

        coeff = (w_aperture ** 2) / (12.0 * mu_fluid)
        v = coeff * np.abs(grad_p)

        v_max_phys = 15.0
        return np.clip(v, 0.0, v_max_phys)

    def coulomb_failure_stress(self, p: np.ndarray,
                                sigma_n: float,
                                mu_fric: float = 0.6,
                                cohesion: float = 2.0e6) -> np.ndarray:
        tau_c = mu_fric * (sigma_n - p) + cohesion

        tau = mu_fric * sigma_n + cohesion
        return tau - tau_c
