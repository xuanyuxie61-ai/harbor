# -*- coding: utf-8 -*-
"""
focused_transport_eq.py
-----------------------
Discrete representation of the focused Parker transport equation
(Parker 1965; Ruffolo 1995; Schlickeiser 2002) for galactic cosmic
rays propagating in the expanding solar wind.

The equation in (r, mu, p) coordinates reads:

  df/dt + (mu v + V_sw) df/dr
      = (1 / r^2) d/dr [ r^2 kappa_rr df/dr ]
      + d/dmu [ D_mumu df/dmu ]
      + d/dmu [ G_mu f ]
      + (1/3) (nabla . V_sw) p df/dp
      + G_p f
      + Q(r, mu, p, t)

where
    kappa_rr  = kappa_par mu^2 + kappa_perp sin^2(psi)
              is the radial diffusion coefficient,
    D_mumu    is the pitch-angle diffusion coefficient,
    G_mu      = (v / (2 L)) (1 - mu^2)(mu - V_sw / (mu v))
              is the focusing term (L = B / (nabla || B) is the
              focusing length),
    G_p       is the adiabatic cooling term.

This module implements each term as a callable that returns its
contribution to df/dt.  The spatial discretisation uses the high-
order stencils defined in ``high_order_stencils``.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional

import cosmic_ray_physics as crp


# =====================================================================
# Focusing length L = B / (nabla_b . B)
# =====================================================================
def focusing_length(r: float, theta: float = crp.THETA_HELIO) -> float:
    """Parker-spiral focusing length  L = |B| / (b . nabla |B|).

    For an Archimedean spiral the analytic expression is

        1/L = d/dr [ ln |B| ] = - (2/r) (B_r^2 / B^2) - (B_T^2 / (r B^2))

    so that

        L = - r / (2 cos^2 psi + sin^2 psi)

    where tan psi = B_T / B_r.  We return the absolute value.
    """
    Br, BT, _, _ = crp.parker_imf(r, theta)
    B2 = Br * Br + BT * BT
    if B2 <= 0.0:
        return math.inf
    invL = (2.0 * Br * Br + BT * BT) / (r * B2)
    return abs(1.0 / invL) if invL != 0.0 else math.inf


def divergence_solar_wind(r: float,
                          V_sw: float = crp.SOLAR_WIND_SPEED_V0) -> float:
    """Divergence of a constant-speed radial solar wind:  2 V_sw / r."""
    if r <= 0.0:
        return math.inf
    return 2.0 * V_sw / r


# =====================================================================
# Pitch-angle scattering
# =====================================================================
def pitch_angle_diffusion(mu: float, R: float, B: float,
                          delta_B_B: float, slab_fraction: float = 0.2,
                          l_c: float = 1.0e8) -> float:
    """Quasi-linear pitch-angle diffusion coefficient D_mumu.

    A standard model (Schlickeiser 2002) is

        D_mumu(r, mu, R) = (Omega r_L / 4) (1 - mu^2)
                           [ (1 - mu^2) delta_B_slab^2 / B^2 * R(mu)
                             + 2 mu^2 delta_B_2D^2 / B^2 * T(mu) ]

    We use a simplified isotropic form for small-scale experiments:

        D_mumu = nu_0 * (1 - mu^2) / 2
    where  nu_0 = v / lambda_mfp,  lambda_mfp = 3 kappa_|| / v.
    """
    nu_0 = 3.0 * crp.quasilinear_kappa_parallel(
        R, B, delta_B_B, l_c, crp.c_light) / max(crp.c_light, 1.0)
    nu_0 = nu_0 / max(crp.c_light, 1.0)
    # Rescale by turbulence level
    nu_0 *= (delta_B_B * delta_B_B + 1.0e-3)
    return max(nu_0, 1.0e-30) * (1.0 - mu * mu) / 2.0


# =====================================================================
# The focused transport operator (semidiscrete in r, mu)
# =====================================================================
class FocusedTransportOperator:
    """Evaluate df/dt from the focused Parker transport equation.

    The state is stored on a tensor grid  f[i_r, i_mu]  with radial
    coordinate  r[i_r]  and pitch-angle cosine  mu[i_mu].  The momentum
    coordinate is treated as a parameter (mono-energetic experiment).

    Attributes
    ----------
    r : ndarray (Nr,)
        Radial grid [m].
    mu : ndarray (Nmu,)
        Pitch-angle cosine grid in (-1, 1).
    Ek : float
        Particle kinetic energy [J] held fixed during evaluation.
    R  : float
        Rigidity [V] corresponding to Ek.
    V_sw : float
        Solar-wind speed [m/s].
    delta_B_B : float
        Relative turbulence amplitude  delta B / B.
    kappa_par_func, kappa_perp_func : callables
        Radial profiles of kappa_||(r) and kappa_perp(r).
    source_func : callable or None
        Source term  Q(r, mu).
    stencil_order : str
        ``'compact4'``, ``'weno5'``, or ``'upwind2'``.
    """

    def __init__(
        self,
        r: np.ndarray,
        mu: np.ndarray,
        Ek: float,
        *,
        V_sw: float = crp.SOLAR_WIND_SPEED_V0,
        delta_B_B: float = 0.3,
        kappa_par_func: Optional[Callable[[float], float]] = None,
        kappa_perp_func: Optional[Callable[[float], float]] = None,
        source_func: Optional[Callable[[float, float], float]] = None,
        stencil_order: str = "upwind2",
    ) -> None:
        if r.ndim != 1 or mu.ndim != 1:
            raise ValueError("FocusedTransportOperator: r and mu must be 1D.")
        if r.size < 3 or mu.size < 3:
            raise ValueError("FocusedTransportOperator: need >= 3 points "
                             "in each coordinate.")
        self.r = r
        self.mu = mu
        self.dr = np.diff(r)
        self.dmu = np.diff(mu)
        self.Nr = r.size
        self.Nmu = mu.size
        self.Ek = Ek
        self.R = crp.kinetic_to_rigidity(Ek)
        self.v = crp.velocity_from_Ek(Ek)
        self.V_sw = float(V_sw)
        self.delta_B_B = float(delta_B_B)
        self.stencil_order = stencil_order
        self.source_func = source_func

        # Default kappa profiles
        if kappa_par_func is None:
            B0 = crp.B_FIELD_1AU
            kpar0 = crp.quasilinear_kappa_parallel(
                self.R, B0, delta_B_B, 1.0e8, self.v)
            # kappa || ~ r^0.3 ( Palmer 1982 )
            self.kappa_par_func = lambda rr: max(kpar0 * (rr / crp.AU) ** 0.3,
                                                 1.0e14)
        else:
            self.kappa_par_func = kappa_par_func
        if kappa_perp_func is None:
            kperp0 = 0.02 * self.kappa_par_func(crp.AU)
            self.kappa_perp_func = lambda rr: kperp0 * (rr / crp.AU) ** 0.5
        else:
            self.kappa_perp_func = kappa_perp_func

        # Precompute B(r), focusing length, div V_sw, kappa_rr, D_mumu
        self._precompute_background()

    # -----------------------------------------------------------------
    def _precompute_background(self) -> None:
        Br, BT, Bmag, psi = zip(*[crp.parker_imf(ri) for ri in self.r])
        self.Br = np.array(Br)
        self.Bmag = np.array(Bmag)
        self.psi = np.array(psi)
        self.L_focus = np.array([focusing_length(ri) for ri in self.r])
        self.divV = np.array([divergence_solar_wind(ri) for ri in self.r])
        self.kpar_r = np.array([self.kappa_par_func(ri) for ri in self.r])
        self.kperp_r = np.array([self.kappa_perp_func(ri) for ri in self.r])
        # kappa_rr(r, mu) = kpar mu^2 + kperp sin^2 psi
        # stored as (Nr, Nmu)
        mu2 = self.mu * self.mu
        sin2 = math.sin(self.psi[0]) ** 2  # psi varies slowly; use avg
        # use mean psi to keep array 2-D cheap
        sin2_mean = float(np.mean(np.sin(self.psi) ** 2))
        Krr = np.zeros((self.Nr, self.Nmu))
        for j, muj in enumerate(mu2):
            Krr[:, j] = self.kpar_r * muj + self.kperp_r * sin2_mean
        self.Krr = Krr

        # D_mumu(r, mu)
        Dmu = np.zeros((self.Nr, self.Nmu))
        for i, ri in enumerate(self.r):
            B = self.Bmag[i]
            for j, muj in enumerate(self.mu):
                Dmu[i, j] = pitch_angle_diffusion(
                    muj, self.R, B, self.delta_B_B)
        self.Dmumu = Dmu

    # -----------------------------------------------------------------
    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Return df/dt evaluated at the state  f of shape (Nr, Nmu)."""
        if f.shape != (self.Nr, self.Nmu):
            raise ValueError("shape mismatch.")
        dfdt = np.zeros_like(f)
        dfdt += self._radial_transport(f)
        dfdt += self._pitch_angle_transport(f)
        dfdt += self._adiabatic_cooling(f)
        dfdt += self._focusing(f)
        dfdt += self._source_term()
        return dfdt

    # ----- radial transport  d/dr [ kappa_rr df/dr ] - (mu v + Vsw) df/dr
    def _radial_transport(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(f)
        # Import stencils lazily to avoid cycles.
        import high_order_stencils as hos
        for j in range(self.Nmu):
            fj = f[:, j]
            # Diffusion:  (1/r^2) d/dr [ r^2 kappa_rr d f / dr ]
            Flux = np.zeros(self.Nr + 1)
            # interior faces i+1/2
            for i in range(1, self.Nr - 1):
                drf = self.dr[i]
                kface = 0.5 * (self.Krr[i, j] + self.Krr[i + 1, j])
                rface = 0.5 * (self.r[i] + self.r[i + 1])
                dfdr = (fj[i + 1] - fj[i]) / drf
                Flux[i + 1] = -kface * rface * rface * dfdr
            Flux[1] = Flux[2]
            Flux[-1] = Flux[-2]
            for i in range(1, self.Nr - 1):
                r2 = self.r[i] * self.r[i]
                out[i, j] += -(Flux[i + 1] - Flux[i]) / (r2 * self.dr[i])

            # Advection by streaming (mu v + V_sw) df/dr
            stream = self.mu[j] * self.v + self.V_sw
            if self.stencil_order == "weno5":
                dfdr = hos.weno5_first_derivative(fj, self.dr, jacobian=False)
            elif self.stencil_order == "compact4":
                dfdr = hos.compact4_first_derivative(fj, self.dr)
            else:
                dfdr = hos.upwind_first_derivative(
                    fj, self.dr, speed=stream)
            out[1:-1, j] += -stream * dfdr[1:-1]
        return out

    # ----- pitch-angle diffusion d/dmu [ D_mumu df/dmu ]
    def _pitch_angle_transport(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(f)
        for i in range(1, self.Nr - 1):
            fi = f[i, :]
            Flux = np.zeros(self.Nmu + 1)
            for j in range(1, self.Nmu - 1):
                dmu = self.dmu[j]
                dface = 0.5 * (self.Dmumu[i, j] + self.Dmumu[i, j + 1])
                dfdmu = (fi[j + 1] - fi[j]) / dmu
                Flux[j + 1] = -dface * dfdmu
            for j in range(1, self.Nmu - 1):
                out[i, j] += -(Flux[j + 1] - Flux[j]) / self.dmu[j]
        return out

    # ----- adiabatic cooling  (1/3)(div V) p df/dp   (mono-energetic: 0)
    def _adiabatic_cooling(self, f: np.ndarray) -> np.ndarray:
        # With fixed Ek this term vanishes; retained for completeness.
        return np.zeros_like(f)

    # ----- focusing term   d/dmu [ G_mu f ]
    def _focusing(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(f)
        for i in range(1, self.Nr - 1):
            Li = self.L_focus[i]
            if Li == 0.0 or math.isinf(Li):
                continue
            vi = self.v
            for j in range(1, self.Nmu - 1):
                muj = self.mu[j]
                # G_mu = (v / (2L))(1 - mu^2) (mu - V_sw / (mu v))
                # regularised near mu = 0
                if abs(muj) < 1.0e-3:
                    mu_safe = math.copysign(1.0e-3, muj) if muj != 0.0 else 1.0e-3
                else:
                    mu_safe = muj
                Gmu = (vi / (2.0 * Li)) * (1.0 - muj * muj) * (
                    mu_safe - self.V_sw / (mu_safe * vi))
                # upwind flux
                if Gmu >= 0.0:
                    flux_right = Gmu * f[i, j]
                    flux_left = Gmu * f[i, j - 1]
                else:
                    flux_right = Gmu * f[i, j + 1]
                    flux_left = Gmu * f[i, j]
                out[i, j] += -(flux_right - flux_left) / self.dmu[j]
        return out

    # ----- source term
    def _source_term(self) -> np.ndarray:
        if self.source_func is None:
            return np.zeros((self.Nr, self.Nmu))
        Q = np.zeros((self.Nr, self.Nmu))
        for i, ri in enumerate(self.r):
            for j, muj in enumerate(self.mu):
                Q[i, j] = self.source_func(ri, muj)
        return Q

    # -----------------------------------------------------------------
    def cfl_timestep(self) -> float:
        """Return an explicit CFL-restricted time step."""
        import stability_analysis as sa
        return sa.explicit_cfl_timestep(
            self.r, self.mu, self.Krr, self.Dmumu,
            streaming_speed_max=abs(self.v) + self.V_sw,
            safety=0.4)
