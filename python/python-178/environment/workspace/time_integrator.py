
import numpy as np
from typing import Callable, Tuple






def rk4_step(y: np.ndarray, t: float, dt: float,
             rhs: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ssp_rk3_step(y: np.ndarray, t: float, dt: float,
                 rhs: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    y1 = y + dt * rhs(y, t)
    y2 = 0.75 * y + 0.25 * y1 + 0.25 * dt * rhs(y1, t + dt)
    y3 = (1.0 / 3.0) * y + (2.0 / 3.0) * y2 + (2.0 / 3.0) * dt * rhs(y2, t + 0.5 * dt)
    return y3


def low_storage_rk45_step(y: np.ndarray, t: float, dt: float,
                          rhs: Callable[[np.ndarray, float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:

    a = np.array([0.0, -567301805773.0/1357537059087.0,
                  -2404267992013.0/2016746695238.0,
                  -3550918686646.0/2091501179385.0,
                  -1275806237668.0/842570457699.0], dtype=np.float64)
    b = np.array([1432997174477.0/9575080441755.0,
                  5161836677717.0/13612068292357.0,
                  1720146321549.0/2090206949498.0,
                  3134564353537.0/4481467310338.0,
                  2277821191437.0/14882151754819.0], dtype=np.float64)
    c = np.array([0.0, 1432997174477.0/9575080441755.0,
                  2526269341429.0/6820363962896.0,
                  2006345519317.0/3224310063776.0,
                  2802321613138.0/2924317926251.0], dtype=np.float64)

    b_hat = np.array([0.0,
                      567301805773.0/1357537059087.0,
                      2404267992013.0/2016746695238.0,
                      3550918686646.0/2091501179385.0,
                      1275806237668.0/842570457699.0], dtype=np.float64)
    y_tmp = y.copy()
    k = np.zeros_like(y)
    for i in range(5):
        ti = t + c[i] * dt
        k = rhs(y_tmp, ti)
        if i < 4:
            y_tmp = y + a[i + 1] * dt * k if i > 0 else y + dt * k
    y_next = y + dt * np.sum([b[i] * k for i in range(5)], axis=0)
    y_emb = y + dt * np.sum([b_hat[i] * k for i in range(5)], axis=0)
    return y_next, y_emb






def symplectic_euler_step(q: np.ndarray, p: np.ndarray, dt: float,
                          grad_Hq: Callable, grad_Hp: Callable) -> Tuple[np.ndarray, np.ndarray]:
    q_next = q + dt * grad_Hp(q, p)
    p_next = p - dt * grad_Hq(q_next, p)
    return q_next, p_next


def stormer_verlet_step(q: np.ndarray, p: np.ndarray, dt: float,
                        grad_V: Callable) -> Tuple[np.ndarray, np.ndarray]:
    p_half = p - 0.5 * dt * grad_V(q)
    q_next = q + dt * p_half
    p_next = p_half - 0.5 * dt * grad_V(q_next)
    return q_next, p_next






class AdaptiveTimeIntegrator:
    def __init__(self, rhs: Callable, y0: np.ndarray, t0: float,
                 atol: float = 1e-6, rtol: float = 1e-4,
                 dt_init: float = 1e-3, dt_min: float = 1e-8, dt_max: float = 1.0):
        self.rhs = rhs
        self.y = np.asarray(y0, dtype=np.float64).copy()
        self.t = float(t0)
        self.atol = atol
        self.rtol = rtol
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.n_steps = 0
        self.n_rejected = 0

    def step(self) -> Tuple[np.ndarray, float, bool]:
        while self.dt >= self.dt_min:
            try:
                y_next, y_emb = low_storage_rk45_step(self.y, self.t, self.dt, self.rhs)
            except Exception:
                self.dt *= 0.5
                self.n_rejected += 1
                continue

            err = np.abs(y_next - y_emb)
            scale = self.atol + self.rtol * np.maximum(np.abs(self.y), np.abs(y_next))
            err_norm = np.max(err / (scale + 1e-30))
            if err_norm <= 1.0:

                self.y = y_next
                self.t += self.dt
                self.n_steps += 1

                fac = min(2.0, max(0.5, 0.9 * (1.0 / err_norm) ** 0.2))
                self.dt = min(fac * self.dt, self.dt_max)
                return self.y.copy(), self.dt / fac, True
            else:

                self.dt *= max(0.5, 0.9 * (1.0 / err_norm) ** 0.25)
                self.n_rejected += 1

        self.y = rk4_step(self.y, self.t, self.dt_min, self.rhs)
        self.t += self.dt_min
        self.n_steps += 1
        return self.y.copy(), self.dt_min, True






def gyroscope_rhs(state: np.ndarray, t: float,
                  A1: float = 1.0, A2: float = 1.5, A3: float = 2.0,
                  m: float = 1.0) -> np.ndarray:
    psi, theta, phi, w1, w2, w3 = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    if abs(sin_theta) < 1e-10:
        sin_theta = 1e-10 if sin_theta >= 0 else -1e-10
    dpsi = (w1 * sin_phi + w2 * cos_phi) / sin_theta
    dtheta = w1 * cos_phi - w2 * sin_phi
    dphi = w3 - cos_theta * dpsi

    M1 = -m * A1 * sin_theta * cos_phi
    M2 = m * A2 * sin_theta * sin_phi
    M3 = 0.0
    dw1 = ((A2 - A3) * w2 * w3 + M1) / A1
    dw2 = ((A3 - A1) * w3 * w1 + M2) / A2
    dw3 = ((A1 - A2) * w1 * w2 + M3) / A3
    return np.array([dpsi, dtheta, dphi, dw1, dw2, dw3], dtype=np.float64)






def kepler_perturbed_rhs(state: np.ndarray, t: float, delta: float = 0.015) -> np.ndarray:
    q1, q2, p1, p2 = state
    r = np.sqrt(q1 * q1 + q2 * q2)
    r3 = r ** 3
    r5 = r ** 5
    dq1 = p1
    dq2 = p2
    dp1 = -q1 / r3 - delta * q1 / r5
    dp2 = -q2 / r3 - delta * q2 / r5
    return np.array([dq1, dq2, dp1, dp2], dtype=np.float64)


def kepler_hamiltonian(state: np.ndarray, delta: float = 0.015) -> float:
    q1, q2, p1, p2 = state
    r = np.sqrt(q1 * q1 + q2 * q2)
    return 0.5 * (p1 * p1 + p2 * p2) - 1.0 / r - delta / (2.0 * r * r * r)
