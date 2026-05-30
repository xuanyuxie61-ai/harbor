
import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Callable






def forward_backward_sweep(state_rhs: Callable, costate_rhs: Callable,
                           control_update: Callable,
                           r0: float, lambda_T: float,
                           time: np.ndarray,
                           u_guess: np.ndarray,
                           max_iter: int = 100,
                           tol: float = 1e-6,
                           alpha_step: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(time)
    u = u_guess.copy()
    r = np.zeros(n)
    lam = np.zeros(n)

    for it in range(max_iter):

        def fwd_ode(y, t_idx):

            idx = min(int(t_idx), n - 1)
            return state_rhs(time[idx], y, u[idx])


        r[0] = r0
        for i in range(1, n):
            dt = time[i] - time[i - 1]
            r[i] = r[i - 1] + dt * state_rhs(time[i - 1], r[i - 1], u[i - 1])

            if r[i] < 0.1e-3:
                r[i] = 0.1e-3


        lam[-1] = lambda_T
        for i in range(n - 2, -1, -1):
            dt = time[i + 1] - time[i]
            lam[i] = lam[i + 1] + dt * costate_rhs(time[i + 1], r[i + 1], lam[i + 1], u[i + 1])


        u_new = np.zeros(n)
        for i in range(n):
            u_new[i] = control_update(r[i], lam[i])

            u_new[i] = np.clip(u_new[i], 0.0, 1.0)


        u_old_norm = np.linalg.norm(u, ord=1) + 1e-15
        diff = np.linalg.norm(u_new - u, ord=1) / u_old_norm
        u = (1.0 - alpha_step) * u + alpha_step * u_new

        if diff < tol:
            break

    return r, lam, u






class WSSOptimalControl:
    def __init__(self, equilibrium_radius: float = 0.005,
                 target_wss_pa: float = 2.5,
                 blood_viscosity_pa_s: float = 0.0035,
                 flow_rate_m3_s: float = 5.0e-5,
                 k_growth: float = 0.5,
                 k_drug: float = 0.3,
                 control_penalty: float = 0.1):
        self.r_eq = equilibrium_radius
        self.wss_target = target_wss_pa
        self.mu = blood_viscosity_pa_s
        self.Q = flow_rate_m3_s
        self.k_g = k_growth
        self.k_u = k_drug
        self.B = control_penalty

    def wss_from_radius(self, r: float) -> float:
        if r < 1e-6:
            return 0.0
        return 4.0 * self.mu * self.Q / (np.pi * r ** 3)

    def state_rhs(self, t: float, r: float, u: float) -> float:
        return self.k_g * (self.r_eq - r) + self.k_u * u * r

    def costate_rhs(self, t: float, r: float, lam: float, u: float) -> float:
        wss = self.wss_from_radius(r)
        dwss_dr = -12.0 * self.mu * self.Q / (np.pi * r ** 4 + 1e-20)
        dH_dr = (wss - self.wss_target) * dwss_dr + lam * (-self.k_g + self.k_u * u)
        return -dH_dr

    def control_update(self, r: float, lam: float) -> float:
        u_star = -lam * self.k_u * r / (self.B + 1e-15)
        return u_star

    def solve(self, r0: float, time: np.ndarray,
              u_guess: np.ndarray = None,
              max_iter: int = 100) -> dict:
        n = len(time)
        if u_guess is None:
            u_guess = np.zeros(n)

        r, lam, u = forward_backward_sweep(
            self.state_rhs,
            self.costate_rhs,
            self.control_update,
            r0, 0.0, time, u_guess,
            max_iter=max_iter, tol=1e-5, alpha_step=0.3
        )

        wss = np.array([self.wss_from_radius(ri) for ri in r])

        return {
            "radius": r,
            "costate": lam,
            "control": u,
            "wss": wss,
            "target_wss": self.wss_target,
            "time": time
        }


def compute_control_cost(wss_trajectory: np.ndarray,
                         target_wss: float,
                         control_trajectory: np.ndarray,
                         B: float = 0.1) -> float:
    n = len(wss_trajectory)
    if n < 2:
        return 0.0

    integrand = 0.5 * (wss_trajectory - target_wss) ** 2 + 0.5 * B * control_trajectory ** 2

    J = np.trapezoid(integrand)
    return float(J)


def wss_physiological_score(wss_pa: float) -> float:
    if 1.0 <= wss_pa <= 7.0:
        return 1.0
    elif wss_pa < 0.5 or wss_pa > 10.0:
        return 0.0
    elif wss_pa < 1.0:
        return (wss_pa - 0.5) / 0.5
    else:
        return (10.0 - wss_pa) / 3.0
