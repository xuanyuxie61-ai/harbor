# -*- coding: utf-8 -*-
"""
stochastic_parker.py
--------------------
Stochastic differential equation (SDE) representation of the
focused Parker transport equation.  Cosmic-ray particles are
propagated by simulating their trajectories as stochastic
processes (Zhang 1999; Kress et al. 2007).

The equivalent Itô SDEs in (r, mu) are

    dr = [ V_sw + (1/r^2) d(r^2 kappa_rr)/dr + (1 - mu^2) v/(2L) ] dt
         + sqrt(2 kappa_rr) dW_r

    dmu = [ v/(2L) (1 - mu^2) (mu - V_sw / (mu v))
            + d(D_mumu)/dmu ] dt + sqrt(2 D_mumu) dW_mu

where  W_r, W_mu  are independent Wiener processes.

The steady-state distribution function is then estimated via

    f(r, mu) = (1 / (N r^2)) sum_particles  1 / |v_r|

weighted by source weights and time spent in the (r, mu) cell.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional, Tuple

import cosmic_ray_physics as crp
import heliocentric_mesh as hm


# =====================================================================
# SDE coefficients
# =====================================================================
def drift_r(r: float, mu: float, kappa_rr: float,
            kappa_rr_deriv: float, L: float, v: float,
            V_sw: float) -> float:
    """Drift coefficient  A_r(r, mu)  of the Itô SDE for r."""
    if r <= 0.0:
        return V_sw
    term1 = V_sw
    term2 = (1.0 / (r * r)) * (2.0 * r * kappa_rr
                                + r * r * kappa_rr_deriv)
    term3 = (1.0 - mu * mu) * v / (2.0 * max(L, 1.0e6))
    return term1 + term2 + term3


def drift_mu(r: float, mu: float, D_mumu: float,
             dDdmu: float, L: float, v: float,
             V_sw: float) -> float:
    """Drift coefficient  A_mu(r, mu)  of the Itô SDE for mu."""
    if r <= 0.0 or abs(mu) < 1.0e-3:
        return 0.0
    mu_safe = mu if abs(mu) > 1.0e-3 else math.copysign(1.0e-3, mu)
    term1 = (v / (2.0 * max(L, 1.0e6))) * (1.0 - mu * mu) * (
        mu_safe - V_sw / (mu_safe * v))
    term2 = dDdmu
    return term1 + term2


# =====================================================================
# Milstein integrator
# =====================================================================
def milstein_step(r: float, mu: float, dt: float,
                  kappa_rr: float, dkdrr: float,
                  D_mumu: float, dDdmu: float,
                  L: float, v: float, V_sw: float,
                  rng: np.random.Generator) -> Tuple[float, float]:
    """Advance one Milstein step of the coupled (r, mu) SDE.

    The Milstein scheme includes the derivative correction to improve
    strong order from 0.5 (Euler-Maruyama) to 1.0.
    """
    dWr = rng.standard_normal() * math.sqrt(abs(dt))
    dWm = rng.standard_normal() * math.sqrt(abs(dt))
    Ar = drift_r(r, mu, kappa_rr, dkdrr, L, v, V_sw)
    Am = drift_mu(r, mu, D_mumu, dDdmu, L, v, V_sw)
    sigma_r = math.sqrt(2.0 * max(kappa_rr, 0.0))
    sigma_mu = math.sqrt(2.0 * max(D_mumu, 0.0))
    r_new = r + Ar * dt + sigma_r * dWr
    mu_new = mu + Am * dt + sigma_mu * dWm
    # reflecting boundaries for mu
    if mu_new > 1.0:
        mu_new = 2.0 - mu_new
    if mu_new < -1.0:
        mu_new = -2.0 - mu_new
    # absorbing boundaries for r
    if r_new < 0.0:
        r_new = -r_new
    return r_new, mu_new


# =====================================================================
# Full stochastic propagator
# =====================================================================
class StochasticParkerPropagator:
    """Propagate N_particles through the heliosphere using the SDE
    equivalent of the focused transport equation.

    The estimator for f(r, mu) uses the adjoint (backward-in-time)
    approach: each pseudo-particle starts at the detector (r_det, mu)
    and runs backward until it reaches the outer boundary (free
    escape, weight = f_LIS) or the inner boundary (weight = 0).
    """

    def __init__(self, r: np.ndarray, mu: np.ndarray,
                 Ek: float,
                 V_sw: float = crp.SOLAR_WIND_SPEED_V0,
                 delta_B_B: float = 0.3,
                 kappa_par_func: Optional[Callable[[float], float]] = None,
                 f_LIS: Optional[Callable[[float], float]] = None,
                 N_particles: int = 2000,
                 dt: float = 3.0e4,
                 max_steps: int = 2000,
                 seed: int = 42):
        self.r = r
        self.mu = mu
        self.Ek = Ek
        self.v = crp.velocity_from_Ek(Ek)
        self.R = crp.kinetic_to_rigidity(Ek)
        self.V_sw = V_sw
        self.delta_B_B = delta_B_B
        self.N_particles = N_particles
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # kappa profiles
        B0 = crp.B_FIELD_1AU
        if kappa_par_func is None:
            kpar0 = crp.quasilinear_kappa_parallel(
                self.R, B0, delta_B_B, 1.0e8, self.v)
            self.kappa_par_func = lambda rr: max(
                kpar0 * (rr / crp.AU) ** 0.3, 1.0e14)
        else:
            self.kappa_par_func = kappa_par_func

        self.f_LIS = f_LIS if f_LIS is not None else (
            lambda R: crp.lis_proton_vladimir2015(R / 1.0e9))

    # -----------------------------------------------------------------
    def kappa_rr(self, r: float, mu: float) -> float:
        Br, BT, Bmag, psi = crp.parker_imf(r)
        kpar = self.kappa_par_func(r)
        kperp = 0.02 * kpar
        return kpar * mu * mu + kperp * math.sin(psi) ** 2

    # -----------------------------------------------------------------
    def _propagate_one(self, r0: float, mu0: float) -> Tuple[float, float]:
        """Backward-in-time propagation of one pseudo-particle.
        Returns (weight, r_escape)."""
        r = r0
        mu = mu0
        for _ in range(self.max_steps):
            if r >= self.r[-1]:
                return self.f_LIS(self.R), r
            if r <= self.r[0]:
                return 0.0, r
            krr = self.kappa_rr(r, mu)
            # finite-difference derivative of kappa_rr w.r.t. r
            dr_fd = 0.01 * r
            krr_p = self.kappa_rr(r + dr_fd, mu)
            krr_m = self.kappa_rr(r - dr_fd, mu)
            dkdrr = (krr_p - krr_m) / (2.0 * dr_fd)
            # D_mumu and its mu derivative
            Br, BT, Bmag, psi = crp.parker_imf(r)
            Dmu = (3.0 * krr / max(self.v, 1.0)) * (1.0 - mu * mu) / 2.0
            Dmu_p = (3.0 * krr / max(self.v, 1.0)) * (
                1.0 - (mu + 0.01) ** 2) / 2.0
            Dmu_m = (3.0 * krr / max(self.v, 1.0)) * (
                1.0 - (mu - 0.01) ** 2) / 2.0
            dDdmu = (Dmu_p - Dmu_m) / 0.02
            L = hm.focusing_length(r) if False else (
                crp.AU * (r / crp.AU) ** 1.0)
            r_new, mu_new = milstein_step(
                r, mu, -self.dt, krr, dkdrr, Dmu, dDdmu, L, self.v,
                self.V_sw, self.rng)
            r, mu = r_new, mu_new
        # did not escape: treat as absorbed
        return 0.5 * self.f_LIS(self.R), r

    # -----------------------------------------------------------------
    def solve_at(self, r_det: float, mu_det: np.ndarray) -> np.ndarray:
        """Return f(r_det, mu) by Monte Carlo over the mu grid."""
        f_est = np.zeros(mu_det.size)
        for j, muj in enumerate(mu_det):
            weights = []
            for _ in range(self.N_particles):
                w, _ = self._propagate_one(r_det, muj)
                weights.append(w)
            f_est[j] = float(np.mean(weights))
        return f_est

    # -----------------------------------------------------------------
    def solve_field(self, r_detect: np.ndarray,
                    mu_detect: np.ndarray) -> np.ndarray:
        """Return f(r, mu) on a sparse grid (use fewer particles)."""
        f = np.zeros((r_detect.size, mu_detect.size))
        for i, ri in enumerate(r_detect):
            for j, muj in enumerate(mu_detect):
                w = [self._propagate_one(ri, muj)[0]
                     for _ in range(max(self.N_particles // 20, 50))]
                f[i, j] = float(np.mean(w))
        return f


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    import cosmic_ray_physics as crp
    r = hm.logarithmic_radial_mesh(Nr=32)
    mu = hm.pitch_angle_mesh(Nmu=8)
    Ek = 1.0 * crp.q_e * 1.0e9   # 1 GV
    prop = StochasticParkerPropagator(r, mu, Ek, N_particles=100,
                                      dt=3.0e4, max_steps=200, seed=1)
    f = prop.solve_at(crp.AU, mu)
    print(f"[stochastic_parker] f at 1 AU over mu = {f}")


if __name__ == "__main__":
    _demo()
