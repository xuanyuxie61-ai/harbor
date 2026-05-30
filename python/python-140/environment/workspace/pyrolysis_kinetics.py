
import numpy as np
from utils import safe_exp, check_bounds, finite_diff_jacobian


class BiomassPyrolysisKinetics:

    def __init__(self, R_gas=8.314):
        self.R = R_gas


        self.A1 = 2.8e19
        self.Ea1 = 242.4e3
        self.A2 = 3.27e14
        self.Ea2 = 196.5e3
        self.A3 = 1.30e10
        self.Ea3 = 150.5e3

        self.A4 = 2.1e16
        self.Ea4 = 186.7e3

        self.A5 = 1.05e15
        self.Ea5 = 179.8e3

        self.y0 = np.array([0.40, 0.30, 0.30, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


    def reaction_rates(self, T):

        raise NotImplementedError("Hole 1: 请补全 reaction_rates 方法")

    def deriv(self, t, y, T_func):
        y = np.asarray(y, dtype=np.float64)

        y = np.clip(y, 0.0, 1.0)

        y_sum = np.sum(y[:3]) + y[3] + y[4] + y[5] + y[6]
        if y_sum > 1e-10 and abs(y_sum - 1.0) > 1e-6:
            y = y / y_sum

        T = T_func(t)
        k = self.reaction_rates(T)
        k1, k2, k3, k4, k5 = k[0], k[1], k[2], k[3], k[4]


        alpha_V2, alpha_C2 = 0.75, 0.25
        alpha_V4, alpha_C4 = 0.65, 0.35
        alpha_V5, alpha_C5 = 0.55, 0.45

        dydt = np.zeros(7, dtype=np.float64)
        dydt[0] = -k1 * y[0]
        dydt[1] = -k4 * y[1]
        dydt[2] = -k5 * y[2]
        dydt[3] =  k1 * y[0] - (k2 + k3) * y[3]
        dydt[4] =  k2 * y[3] * alpha_V2 + k4 * y[1] * alpha_V4 + k5 * y[2] * alpha_V5
        dydt[5] =  k2 * y[3] * alpha_C2 + k4 * y[1] * alpha_C4 + k5 * y[2] * alpha_C5
        dydt[6] =  k3 * y[3]
        return dydt

    def solve_rk4(self, tspan, y0, n_steps, T_func):
        y0 = np.asarray(y0, dtype=np.float64)
        t0, tf = tspan
        dt = (tf - t0) / n_steps
        m = len(y0)
        t = np.zeros(n_steps + 1, dtype=np.float64)
        y = np.zeros((n_steps + 1, m), dtype=np.float64)
        t[0] = t0
        y[0, :] = y0

        for i in range(n_steps):
            ti = t[i]
            yi = y[i, :]
            f1 = self.deriv(ti, yi, T_func)
            f2 = self.deriv(ti + dt / 2.0, yi + dt * f1 / 2.0, T_func)
            f3 = self.deriv(ti + dt / 2.0, yi + dt * f2 / 2.0, T_func)
            f4 = self.deriv(ti + dt, yi + dt * f3, T_func)
            t[i + 1] = ti + dt
            y[i + 1, :] = yi + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0

            y[i + 1, :] = np.maximum(y[i + 1, :], 0.0)

            total = np.sum(y[i + 1, :])
            if total > 1e-10:
                y[i + 1, :] /= total
        return t, y

    def solve_midpoint(self, tspan, y0, n_steps, T_func, theta=0.5, max_iter=10):
        y0 = np.asarray(y0, dtype=np.float64)
        t0, tf = tspan
        dt = (tf - t0) / n_steps
        m = len(y0)
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, m), dtype=np.float64)
        y[0, :] = y0

        for i in range(n_steps):
            tm = t[i] + theta * dt
            ym = y[i, :].copy()

            for _ in range(max_iter):
                fm = self.deriv(tm, ym, T_func)
                ym_new = y[i, :] + theta * dt * fm
                if np.linalg.norm(ym_new - ym) < 1e-12:
                    ym = ym_new
                    break
                ym = ym_new
            y[i + 1, :] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y[i, :]
            y[i + 1, :] = np.maximum(y[i + 1, :], 0.0)
            total = np.sum(y[i + 1, :])
            if total > 1e-10:
                y[i + 1, :] /= total
        return t, y


def doughnut_pyrolysis_flow(t, y, m_param=1.0, n_param=0.5):
    y = np.asarray(y, dtype=np.float64)
    dy1dt = -m_param * y[1] + n_param * y[0] * y[2]
    dy2dt =  m_param * y[0] + n_param * y[1] * y[2]
    dy3dt =  0.5 * n_param * (1.0 - y[0] ** 2 - y[1] ** 2 + y[2] ** 2)
    return np.array([dy1dt, dy2dt, dy3dt], dtype=np.float64)


def solve_doughnut_flow_rk4(tspan, y0, n_steps, m_param=1.0, n_param=0.5):
    y0 = np.asarray(y0, dtype=np.float64)
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.zeros(n_steps + 1, dtype=np.float64)
    y = np.zeros((n_steps + 1, m), dtype=np.float64)
    t[0] = t0
    y[0, :] = y0

    for i in range(n_steps):
        f1 = doughnut_pyrolysis_flow(t[i], y[i, :], m_param, n_param)
        f2 = doughnut_pyrolysis_flow(t[i] + dt / 2.0, y[i, :] + dt * f1 / 2.0, m_param, n_param)
        f3 = doughnut_pyrolysis_flow(t[i] + dt / 2.0, y[i, :] + dt * f2 / 2.0, m_param, n_param)
        f4 = doughnut_pyrolysis_flow(t[i] + dt, y[i, :] + dt * f3, m_param, n_param)
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
    return t, y
