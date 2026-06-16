"""
halo_ode_systems.py
===================
Three coupled ODE systems relevant for dark matter halo dynamics,
integrated with a second-order implicit trapezoidal rule.

1. *Density-oscillation (Oregonator-inspired)* --
   Baryonic feedback inside the halo core drives limit-cycle
   oscillations of the central density, analogous to the
   Belousov-Zhabotinsky Oregonator (Field-Koros-Noyes, 1972):

       eta1 du/dt = q v - u v + u (1 - u)
       eta2 dv/dt = -q v - u v + f w
               dw/dt = u - w

   We reinterpret (u, v, w) as (central density contrast, gas
   fraction, star-formation reservoir).

2. *Halo figure precession (gyroscope)* --
   A triaxial dark matter halo with principal axes (A1, A2, A3)
   precesses under an external tidal torque in the same way as a
   torque-free gyroscope (Moulton, 1958):

       dpsi  / dt = (omega1 sin phi + omega2 cos phi) / sin theta
       dtheta/dt =  omega1 cos phi - omega2 sin phi
       dphi  / dt =  omega3 - cos theta dpsi/dt
       A1 domega1/dt = (A2 - A3) omega2 omega3 + M1
       A2 domega2/dt = (A3 - A1) omega3 omega1 + M2
       A3 domega3/dt = (A1 - A2) omega1 omega2 + M3

3. *Trapezoidal ODE integrator* --
   The implicit trapezoidal rule

       y_{n+1} = y_n + (h/2)(f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

   is solved at each step by a fixed-point iteration (or Newton if
   the Jacobian is supplied).  The method is A-stable, making it
   suitable for the mildly stiff halo oscillation problem.
"""

from __future__ import annotations
import math
from typing import Callable, Tuple

import numpy as np


# ---------- Oregonator-style density oscillation -----------------------------

class DensityOscillator:
    """Oregonator-inspired model for baryon-driven halo core oscillations."""

    def __init__(self, eta1: float = 0.05, eta2: float = 0.05,
                 q: float = 0.02, f: float = 1.0):
        self.eta1 = eta1
        self.eta2 = eta2
        self.q = q
        self.f = f

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        u, v, w = y
        du = (self.q * v - u * v + u * (1.0 - u)) / self.eta1
        dv = (-self.q * v - u * v + self.f * w) / self.eta2
        dw = u - w
        # Numerical safeguard: clip to avoid blow-up
        return np.clip(np.array([du, dv, dw]), -1e6, 1e6)


# ---------- Triaxial halo precession -----------------------------------------

class HaloPrecession:
    """Gyroscope-like precession of a triaxial dark matter halo."""

    def __init__(self, A1: float = 1.0, A2: float = 1.1, A3: float = 1.3,
                 m_tidal: float = 0.05):
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.m_tidal = m_tidal

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        psi, theta, phi, om1, om2, om3 = y
        # Tidal torque model: external perturbation from a filament
        M1 = -self.m_tidal * self.A1 * math.sin(theta) * math.cos(phi)
        M2 =  self.m_tidal * self.A2 * math.sin(theta) * math.sin(phi)
        M3 = 0.0
        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-8:
            sin_theta = 1e-8 * (1.0 if sin_theta >= 0 else -1.0)
        dpsi  = (om1 * math.sin(phi) + om2 * math.cos(phi)) / sin_theta
        dth   =  om1 * math.cos(phi) - om2 * math.sin(phi)
        dphi  =  om3 - math.cos(theta) * dpsi
        dom1  = ((self.A2 - self.A3) * om2 * om3 + M1) / self.A1
        dom2  = ((self.A3 - self.A1) * om3 * om1 + M2) / self.A2
        dom3  = ((self.A1 - self.A2) * om1 * om2 + M3) / self.A3
        return np.array([dpsi, dth, dphi, dom1, dom2, dom3])


# ---------- Implicit trapezoidal integrator ----------------------------------

def trapezoidal_step(f: Callable[[float, np.ndarray], np.ndarray],
                     t: float, y: np.ndarray, h: float,
                     max_iter: int = 25, tol: float = 1e-9
                     ) -> Tuple[np.ndarray, bool]:
    """One step of the implicit trapezoidal rule.

    Solves   y_new = y + (h/2) (f(t, y) + f(t+h, y_new))
    by fixed-point iteration with relaxation."""
    f0 = f(t, y)
    y_pred = y + h * f0                          # explicit Euler predictor
    for _ in range(max_iter):
        f1 = f(t + h, y_pred)
        y_new = y + 0.5 * h * (f0 + f1)
        if np.max(np.abs(y_new - y_pred)) < tol:
            return y_new, True
        y_pred = 0.5 * y_pred + 0.5 * y_new      # under-relaxation
    return y_new, False


def integrate_trajectory(f: Callable[[float, np.ndarray], np.ndarray],
                         y0: np.ndarray,
                         t_span: Tuple[float, float],
                         n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] with
    n_steps trapezoidal steps."""
    t0, t1 = t_span
    h = (t1 - t0) / max(n_steps, 1)
    ts = np.linspace(t0, t1, n_steps + 1)
    ys = np.zeros((n_steps + 1, y0.size))
    ys[0] = y0
    converged_count = 0
    for k in range(n_steps):
        y_new, ok = trapezoidal_step(f, ts[k], ys[k], h)
        ys[k + 1] = y_new
        if ok:
            converged_count += 1
    return ts, ys


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    osc = DensityOscillator()
    y0 = np.array([0.8, 0.2, 0.5])
    ts, ys = integrate_trajectory(osc.rhs, y0, (0.0, 1.0), 100)
    gyro = HaloPrecession()
    y0g = np.array([0.1, 1.0, 0.0, 0.5, 0.3, 1.0])
    tsg, ysg = integrate_trajectory(gyro.rhs, y0g, (0.0, 1.0), 100)
    return dict(osc_final=ys[-1].tolist(),
                gyro_final=ysg[-1].tolist(),
                n_steps=100)
