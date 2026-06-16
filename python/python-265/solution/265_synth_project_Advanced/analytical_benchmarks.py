# -*- coding: utf-8 -*-
"""
analytical_benchmarks.py
------------------------
Closed-form or semi-analytical benchmark solutions for the
cosmic-ray transport equation, used to validate the numerical
schemes.

Included benchmarks
-------------------
1. ``force_field_model`` -- the Gleeson-Axford force-field modulation
   model giving the modulated LIS as a function of the modulation
   potential phi.
2. ``convection_diffusion_steady`` -- 1-D steady solution of
      V_sw df/dr = kappa d^2 f / dr^2  with f(0)=f0, f(L)=0.
3. ``barenblatt_cosmicray`` -- self-similar solution of the nonlinear
   porous-medium-like equation  df/dt = div( f^m kappa grad f ) that
   arises when kappa depends on f (nonlinear scattering).
4. ``constant_kappa_analytic`` -- time-dependent Green's function for
   constant kappa and uniform solar wind in an infinite domain.
5. ``compute_residual`` -- evaluate the residual of the transport
   equation against a candidate solution.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple

import cosmic_ray_physics as crp


# =====================================================================
# 1. Force-field model
# =====================================================================
def force_field_model(Ek_GeV: np.ndarray, phi_MV: float = 500.0,
                      A: int = 1, Z: int = 1) -> np.ndarray:
    """Return the modulated proton LIS.

    J_mod(E) = ((E + m_p)^2 - m_p^2) / ((E + phi)^2 - m_p^2 + 2 m_p phi)
               * J_LIS(E + phi |Z| / A)

    with  E, phi in GV-equivalent (multiply by q_e).
    """
    phi_V = phi_MV * 1.0e6
    Ek = Ek_GeV * 1.0e9 * crp.q_e
    phi = phi_V * crp.q_e
    rest = crp.m_p * crp.c_light ** 2
    EL = Ek + phi * abs(Z) / max(A, 1)
    num = Ek * Ek + 2.0 * Ek * rest
    den = EL * EL + 2.0 * EL * rest
    mod_ratio = np.where(den > 0.0, num / den, 0.0)
    J_lis = np.array([crp.lis_proton_vladimir2015(E / (1.0e9 * crp.q_e))
                      for E in EL])
    return mod_ratio * J_lis


# =====================================================================
# 2. Steady convection-diffusion
# =====================================================================
def convection_diffusion_steady(r: np.ndarray, V_sw: float,
                                kappa: float, f0: float,
                                L: float) -> np.ndarray:
    """Steady solution of  V df/dr = kappa d^2 f / dr^2  with f(0)=f0,
    f(L) = 0.

    Exact form:
        f(r) = f0 * (exp(Pe (1 - r/L)) - 1) / (exp(Pe) - 1)
    where  Pe = V L / kappa  is the cell Peclet number.
    """
    if kappa <= 0.0:
        return np.where(r < L, f0, 0.0)
    Pe = V_sw * L / kappa
    if abs(Pe) < 1.0e-8:
        return f0 * (1.0 - r / L)
    return f0 * (np.exp(Pe * (1.0 - r / L)) - 1.0) / (math.exp(Pe) - 1.0)


# =====================================================================
# 3. Barenblatt-like self-similar solution
# =====================================================================
class BarenblattCosmicRay:
    """Self-similar solution of

        df/dt = d/dr[ D_0 f^{m-1} df/dr ]     (nonlinear diffusion)

    which arises when  kappa depends on f  via nonlinear scattering
    (e.g. streaming instability:  kappa ~ f^{-1}).

    The Barenblatt solution is

        f(r, t) = t^{-alpha} [ C - beta (r / t^gamma)^2 ]_{+}^{1/(m-1)}

    with
        alpha = 1 / (m + 1),  beta = (m-1) / (2 m (m + 1) D_0),
        gamma = beta.
    """

    def __init__(self, m: float = 2.0, D0: float = 1.0e22,
                 C: float = 1.0, t0: float = 1.0):
        if m <= 1.0:
            raise ValueError("Barenblatt: m > 1 required.")
        self.m = m
        self.D0 = D0
        self.C = C
        self.t0 = t0
        self.alpha = 1.0 / (m + 1.0)
        self.beta = (m - 1.0) / (2.0 * m * (m + 1.0) * D0)
        self.gamma = self.beta

    def __call__(self, r: np.ndarray, t: float) -> np.ndarray:
        bot = (t + self.t0) ** self.gamma
        arg = self.C - self.beta * (r / bot) ** 2
        f = np.where(arg > 0.0,
                     (1.0 / (t + self.t0) ** self.alpha)
                     * np.power(arg, 1.0 / (self.m - 1.0)),
                     0.0)
        return f

    def time_derivative(self, r: np.ndarray, t: float) -> np.ndarray:
        """Return df/dt of the Barenblatt solution."""
        m = self.m
        bot = (t + self.t0) ** self.gamma
        arg = self.C - self.beta * (r / bot) ** 2
        safe = arg > 0.0
        out = np.zeros_like(r)
        tpb = t + self.t0
        fac = np.where(safe, np.power(np.maximum(arg, 0.0),
                                      1.0 / (m - 1.0) - 1.0), 0.0)
        term1 = (-self.alpha / tpb) * np.where(safe, np.power(
            np.maximum(arg, 0.0), 1.0 / (m - 1.0)), 0.0)
        term2 = np.where(safe, -2.0 * self.beta * (r ** 2) * self.gamma
                         / (tpb ** (2 * self.gamma + 1))
                         * (1.0 / (m - 1.0)) * fac, 0.0)
        return term1 + term2

    def residual(self, r: np.ndarray, t: float,
                 dr: float = 1.0e10) -> np.ndarray:
        """Return  df/dt - d/dr[ D_0 f^{m-1} df/dr ]  (numerical d/dr)."""
        f = self(r, t)
        ft = self.time_derivative(r, t)
        r_p = r + dr
        r_m = r - dr
        f_p = self(r_p, t)
        f_m = self(r_m, t)
        # D f^{m-1} df/dr evaluated at r+dr/2 and r-dr/2
        def flux(x, fx):
            return np.where(fx > 0.0,
                            self.D0 * np.power(fx, self.m - 1.0) * 0.0,
                            0.0)
        # central second-order derivative of  F(f) = D_0 f^m / m
        F = np.where(f > 0.0, self.D0 * np.power(f, self.m) / self.m, 0.0)
        F_p = np.where(f_p > 0.0,
                       self.D0 * np.power(f_p, self.m) / self.m, 0.0)
        F_m = np.where(f_m > 0.0,
                       self.D0 * np.power(f_m, self.m) / self.m, 0.0)
        lap = (F_p - 2.0 * F + F_m) / (dr * dr)
        return ft - lap


# =====================================================================
# 4. Time-dependent Green's function (constant kappa, uniform V)
# =====================================================================
def greens_function_constant(r: np.ndarray, t: float, V_sw: float,
                             kappa: float, r_src: float,
                             Q0: float = 1.0) -> np.ndarray:
    """Return the Green's function solution of

        df/dt + V df/dr = kappa d^2 f / dr^2,
        f(r, 0) = Q0 * delta(r - r_src).

    Closed form:
        f(r, t) = Q0 / sqrt(4 pi kappa t)
                  * exp(-(r - r_src - V t)^2 / (4 kappa t)).
    """
    if t <= 0.0 or kappa <= 0.0:
        out = np.zeros_like(r)
        idx = np.argmin(np.abs(r - r_src))
        out[idx] = Q0
        return out
    sigma2 = 2.0 * kappa * t
    centre = r_src + V_sw * t
    return Q0 / math.sqrt(2.0 * math.pi * sigma2) * np.exp(
        -(r - centre) ** 2 / (2.0 * sigma2))


# =====================================================================
# 5. Residual
# =====================================================================
def compute_residual(f_func: Callable[[np.ndarray, float], np.ndarray],
                     r: np.ndarray, t: float, V_sw: float,
                     kappa_func: Callable[[float], float],
                     dr: float = 1.0e10, dt: float = 1.0) -> np.ndarray:
    """Return the residual  R = df/dt + V df/dr - (1/r^2) d/dr[r^2
    kappa df/dr]  of a candidate solution.
    """
    f = f_func(r, t)
    ft = (f_func(r, t + 0.5 * dt) - f_func(r, t - 0.5 * dt)) / dt
    f_p = f_func(r + dr, t)
    f_m = f_func(r - dr, t)
    fr = (f_p - f_m) / (2.0 * dr)
    # diffusion term
    kappa = np.array([kappa_func(ri) for ri in r])
    k_p = np.array([kappa_func(ri + dr) for ri in r])
    k_m = np.array([kappa_func(ri - dr) for ri in r])
    fr_p = (f_func(r + dr, t) - f) / dr
    fr_m = (f - f_func(r - dr, t)) / dr
    flux_p = k_p * 0.5 * (kappa + k_p) * 0.5 * fr_p
    flux_m = k_m * 0.5 * (kappa + k_m) * 0.5 * fr_m
    lap = (flux_p - flux_m) / (2.0 * dr)
    return ft + V_sw * fr - lap


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    Ek = np.logspace(-1, 2, 16)
    J = force_field_model(Ek, phi_MV=500.0)
    print(f"[analytical_benchmarks] Force-field modulated J at Ek=[{Ek[0]:.2f},"
          f" {Ek[-1]:.1f}] GeV:  min={J.min():.3e}  max={J.max():.3e}")
    r = np.linspace(0.01, 1.0, 32)
    f_steady = convection_diffusion_steady(r, 4.0e5, 1.0e22, 1.0, 1.0)
    print(f"[analytical_benchmarks] Steady conv-diff f(0)={f_steady[0]:.3e}, "
          f"f(1)={f_steady[-1]:.3e}")
    b = BarenblattCosmicRay(m=2.0, D0=1.0)
    print(f"[analytical_benchmarks] Barenblatt peak at t=1: {b(np.array([0.0]), 1.0)}")


if __name__ == "__main__":
    _demo()
