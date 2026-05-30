# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple
from utils import robust_divide, clip_to_unit





class GyroscopeDynamics:

    def __init__(self, A1: float = 1.0, A2: float = 1.0, A3: float = 0.5,
                 m: float = 1.0):
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.m = m

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        psi, theta, phi, w1, w2, w3 = y
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)

        sin_t_reg = sin_t if abs(sin_t) > 1e-8 else np.copysign(1e-8, sin_t)


        dpsi = (w1 * sin_p + w2 * cos_p) / sin_t_reg
        dtheta = w1 * cos_p - w2 * sin_p
        dphi = w3 - cos_t * dpsi


        M1 = -self.m * self.A1 * sin_t * cos_p
        M2 = self.m * self.A2 * sin_t * sin_p
        M3 = 0.0


        dw1 = ((self.A2 - self.A3) * w2 * w3 + M1) / self.A1
        dw2 = ((self.A3 - self.A1) * w3 * w1 + M2) / self.A2
        dw3 = ((self.A1 - self.A2) * w1 * w2 + M3) / self.A3

        return np.array([dpsi, dtheta, dphi, dw1, dw2, dw3])

    def rk4_step(self, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(t, y)
        k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self.rhs(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, y0: np.ndarray, t_span: Tuple[float, float],
                  n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        t_arr = np.linspace(t0, t1, n_steps + 1)
        y_arr = np.zeros((n_steps + 1, len(y0)))
        y_arr[0] = y0
        for i in range(n_steps):
            y_arr[i + 1] = self.rk4_step(t_arr[i], y_arr[i], dt)
        return t_arr, y_arr





def generate_scanning_trajectory(n_steps: int = 2000,
                                  spin_period_min: float = 60.0,
                                  precession_period_min: float = 192.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_steps)
    omega_spin = 2.0 * np.pi * n_steps / spin_period_min
    omega_prec = 2.0 * np.pi * n_steps / precession_period_min

    theta_ax = np.radians(45.0)
    phi_ax = omega_prec * t


    phi_spin = omega_spin * t


    boresight_tilt = np.radians(85.0)


    theta_p = np.zeros(n_steps)
    phi_p = np.zeros(n_steps)

    for i in range(n_steps):

        bx = np.sin(boresight_tilt) * np.cos(phi_spin[i])
        by = np.sin(boresight_tilt) * np.sin(phi_spin[i])
        bz = np.cos(boresight_tilt)

        ct = np.cos(theta_ax)
        st = np.sin(theta_ax)
        cp = np.cos(phi_ax[i])
        sp = np.sin(phi_ax[i])

        x = cp * (ct * bx + st * bz) - sp * by
        y = sp * (ct * bx + st * bz) + cp * by
        z = -st * bx + ct * bz
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r < 1e-12:
            theta_p[i] = 0.0
            phi_p[i] = 0.0
        else:
            theta_p[i] = np.arccos(clip_to_unit(z / r))
            phi_p[i] = np.arctan2(y, x)
            if phi_p[i] < 0:
                phi_p[i] += 2.0 * np.pi

    return t, theta_p, phi_p





def compute_hit_map(theta: np.ndarray, phi: np.ndarray,
                    n_theta: int = 36, n_phi: int = 72) -> np.ndarray:
    hits = np.zeros((n_theta, n_phi), dtype=int)
    dtheta = np.pi / n_theta
    dphi = 2.0 * np.pi / n_phi
    for th, ph in zip(theta, phi):
        it = min(int(th / dtheta), n_theta - 1)
        ip = min(int(ph / dphi), n_phi - 1)
        hits[it, ip] += 1
    return hits


def compute_coverage_uniformity(hits: np.ndarray) -> float:
    mean_h = np.mean(hits)
    if mean_h < 1e-12:
        return 0.0
    std_h = np.std(hits)
    return max(0.0, 1.0 - std_h / mean_h)
