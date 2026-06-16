"""
eos_gas.py
==========

Equation of state for a multi-phase, self-gravitating interstellar/
intergalactic gas mixture, with characteristic-structure analysis
inspired by the ellipse geometry routines (328_ellipse).

Key mappings from the seed project 328_ellipse:
  * ellipse_area1(A, R)     -> characteristic ellipse area in velocity space
  * ellipse_perimeter(a, b) -> circumference of the Riemann-invariant ellipse
  * ellipse_eccentricity    -> kinematic anisotropy of the pressure tensor
  * elliptic_inc_fm, fm     -> incomplete elliptic integrals for Riemann
                               invariants along the characteristics of the
                               2-D shallow-water analogue of the Euler system.

Physical content
----------------
For an ideal gas with ratio of specific heats gamma and mean molecular
weight mu the basic relations are:

    P     = (gamma - 1) * rho * e_th
    c_s   = sqrt(gamma * P / rho) = sqrt(gamma k_B T / (mu m_p))
    c_s,eff = sqrt(c_s^2 + (1/3) sigma_turb^2)

where sigma_turb is the 1-D turbulent velocity dispersion.  The *effective*
sound speed replaces c_s in the CFL and Jeans-length formulae when the gas
is turbulent (e.g. McKee & Ostriker 1977).

Riemann invariants
------------------
For isentropic flow in 1-D the two Riemann invariants are

    J_{\\pm} = v \\pm \\frac{2 c_s}{\\gamma - 1}

and the characteristic ellipse in the (x, t) plane (for a wave train with
phase speed c_s and dispersion k^2 nu) has semi-axes

    a = c_s * T / sqrt(1 + (k^2 nu / c_s^2))
    b = c_s * T * sqrt(k^2 nu / c_s^2) / sqrt(1 + (k^2 nu / c_s^2))

whose perimeter P(a, b) is computed via Ramanujan's approximation using
the complete elliptic integral of the second kind E(e).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from astro_constants import (
    BOLTZMANN_CGS, PROTON_MASS_CGS, GRAVITATIONAL_CGS,
    MEAN_MOLECULAR_WEIGHT_NEUTRAL, MEAN_MOLECULAR_WEIGHT_IONIZED,
)


# =====================================================================
#                         IDEAL-GAS EOS
# =====================================================================

class IdealGasEOS:
    """
    Standard gamma-law EOS with support for partial ionisation
    via a temperature-dependent mean molecular weight mu(T).

    The Saha-equilibrium approximation for a primordial H/He mixture is

        x_e(T, n) ~ sqrt((2 pi m_e k T / h^2)^{3/2} / n) *
                    exp(-chi_H / (2 k T))

    and mu = (1 + 4 y) / (1 + 3 y + x_e) where y = Y_p / (4 X_p).
    """

    def __init__(
        self,
        gamma: float = 5.0 / 3.0,
        mu_neutral: float = MEAN_MOLECULAR_WEIGHT_NEUTRAL,
        mu_ionized: float = MEAN_MOLECULAR_WEIGHT_IONIZED,
        ionisation_transition_k: float = 1.0e4,
    ) -> None:
        if not (1.0 < gamma <= 2.0):
            raise ValueError("gamma must lie in (1, 2]")
        self.gamma = gamma
        self.gamma_m1 = gamma - 1.0
        self.mu_neutral = mu_neutral
        self.mu_ionized = mu_ionized
        self._k_trans = ionisation_transition_k

    # -----------------------------------------------------------------
    #  mu(T) via tanh switching (smooth approximation to Saha jump)
    # -----------------------------------------------------------------
    def mu(self, temperature_k: float):
        """
        Temperature-dependent mean molecular weight.  We approximate
        the Saha ionisation transition by a smooth tanh switch:

            mu(T) = (mu_n + mu_i)/2  -  (mu_n - mu_i)/2 * tanh((T - T_t)/dT)

        where dT ~ T_t / 4 controls the steepness.
        """
        dt = self._k_trans / 4.0
        x = (temperature_k - self._k_trans) / dt
        try:
            import numpy as np
            th = np.tanh(x)
        except Exception:
            th = math.tanh(x)
        return 0.5 * (self.mu_neutral + self.mu_ionized) \
               - 0.5 * (self.mu_neutral - self.mu_ionized) * th

    # -----------------------------------------------------------------
    #  core thermodynamic quantities
    # -----------------------------------------------------------------
    def pressure(self, rho_cgs: float, e_th_cgs: float) -> float:
        """P = (gamma - 1) rho e_th."""
        return max(self.gamma_m1 * rho_cgs * e_th_cgs, 0.0)

    def temperature(self, e_th_cgs: float, mu: Optional[float] = None) -> float:
        """T = mu m_p e_th / k_B."""
        mu_use = mu if mu is not None else self.mu_neutral
        return e_th_cgs * mu_use * PROTON_MASS_CGS / BOLTZMANN_CGS

    def specific_energy_from_T(self, T_k: float, mu: Optional[float] = None) -> float:
        """e_th = k_B T / ((gamma-1) mu m_p)."""
        mu_use = mu if mu is not None else self.mu(T_k)
        return BOLTZMANN_CGS * T_k / (self.gamma_m1 * mu_use * PROTON_MASS_CGS)

    def sound_speed(self, rho_cgs: float, p_cgs: float) -> float:
        """Adiabatic sound speed  c_s = sqrt(gamma P / rho)."""
        ratio = p_cgs / max(rho_cgs, 1.0e-60)
        return math.sqrt(self.gamma * max(ratio, 0.0))

    def effective_sound_speed(self, rho_cgs: float, p_cgs: float,
                              sigma_turb_cgs: float = 0.0) -> float:
        """
        Effective sound speed including turbulent pressure support:

            c_{s,eff} = sqrt(c_s^2 + sigma_turb^2 / 3).

        The factor 1/3 comes from the isotropic Reynolds stress
        decomposition for isotropic turbulence.
        """
        cs2 = self.gamma * p_cgs / max(rho_cgs, 1.0e-60)
        ct2 = sigma_turb_cgs ** 2 / 3.0
        return math.sqrt(max(cs2 + ct2, 0.0))

    def isentropic_pressure(self, rho_cgs: float, entropy_cgs: float) -> float:
        """
        P = K rho^gamma where K = (gamma-1) * A,
        with A the specific entropy per unit mass.
        """
        return self.gamma_m1 * entropy_cgs * (rho_cgs ** self.gamma)

    # -----------------------------------------------------------------
    #  characteristic ellipse in velocity space
    #  (maps ellipse_area1, ellipse_eccentricity from 328_ellipse)
    # -----------------------------------------------------------------
    def hodograph_ellipse_axes(self, vx_cgs: float, vy_cgs: float,
                                cs_cgs: float) -> Tuple[float, float]:
        """
        In the 2-D velocity hodograph plane (vx, vy) the local
        characteristic ellipse has semi-major axis

            a = |v| + c_s
            b = |v| - c_s        (if |v| > c_s; otherwise b = c_s - |v|)

        This is the standard epicycle construction for the Euler
        characteristics in the hodograph plane.
        """
        v_abs = math.sqrt(vx_cgs ** 2 + vy_cgs ** 2)
        a = v_abs + cs_cgs
        b = abs(v_abs - cs_cgs)
        return (a, b)

    def hodograph_ellipse_eccentricity(self, vx_cgs: float, vy_cgs: float,
                                        cs_cgs: float) -> float:
        """
        Eccentricity of the hodograph ellipse:

            e = sqrt(1 - (b/a)^2)

        e = 0 for a subsonic stagnant flow (|v| = 0) and e -> 1 for
        a highly supersonic unidirectional flow.
        """
        a, b = self.hodograph_ellipse_axes(vx_cgs, vy_cgs, cs_cgs)
        if a < 1.0e-60:
            return 0.0
        ratio = b / a
        return math.sqrt(max(1.0 - ratio * ratio, 0.0))

    def hodograph_ellipse_area(self, vx_cgs: float, vy_cgs: float,
                                cs_cgs: float) -> float:
        """
        Area = pi a b of the hodograph ellipse.
        (Direct analogue of ellipse_area1(A, R).)
        """
        a, b = self.hodograph_ellipse_axes(vx_cgs, vy_cgs, cs_cgs)
        return math.pi * a * b

    def hodograph_ellipse_perimeter(self, vx_cgs: float, vy_cgs: float,
                                     cs_cgs: float) -> float:
        """
        Perimeter via Ramanujan's approximation:

            P ~ pi (a + b) [ 1 + 3 h / (10 + sqrt(4 - 3 h)) ]
            where h = ((a - b)/(a + b))^2.

        (Direct analogue of ellipse_perimeter(a, b).)
        """
        a, b = self.hodograph_ellipse_axes(vx_cgs, vy_cgs, cs_cgs)
        if a + b < 1.0e-60:
            return 0.0
        h = ((a - b) / (a + b)) ** 2
        return math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 0.0))))


# =====================================================================
#                 INCOMPLETE ELLIPTIC INTEGRAL (for Riemann inv.)
# =====================================================================

def incomplete_elliptic_F(phi: float, k: float, n_terms: int = 16) -> float:
    """
    Incomplete elliptic integral of the first kind:

        F(phi | m) = int_0^phi dt / sqrt(1 - m sin^2 t),   m = k^2.

    Computed via the series expansion (Abramowitz & Stegun 17.4.1):

        F(phi | m) = phi + (m/6) phi^3 + (3m^2/40) phi^5 + ...

    For |phi| <= pi/2 and |m| < 1 the series converges rapidly; we take
    `n_terms` = 16 which gives ~14 digits of accuracy for |m| < 0.5.
    """
    if abs(phi) > math.pi / 2.0:
        raise ValueError("phi must be in [-pi/2, pi/2]")
    if k * k >= 1.0:
        # asymptotic log form for k -> 1
        if k >= 1.0 - 1.0e-12:
            return math.log(math.tan(abs(phi) / 2.0 + math.pi / 4.0))
        k = min(k, 1.0 - 1.0e-12)
    m = k * k
    # series coefficients via recurrence
    s = phi
    power = phi
    coef = 1.0
    for n in range(1, n_terms + 1):
        coef *= (2 * n - 1) * (2 * n - 1) / (2 * n * (2 * n + 1))
        power *= m * phi * phi
        s += coef * power
        if abs(coef * power) < 1.0e-15 * abs(s):
            break
    return s


def incomplete_elliptic_E(phi: float, k: float, n_terms: int = 16) -> float:
    """
    Incomplete elliptic integral of the second kind:

        E(phi | m) = int_0^phi sqrt(1 - m sin^2 t) dt.

    Series (Abramowitz & Stegun 17.4.2):

        E(phi | m) = phi - (m/6) phi^3 - (3m^2/40) phi^5 - ...

    Used below to compute the Riemann-invariant ellipse perimeter.
    """
    if abs(phi) > math.pi / 2.0:
        raise ValueError("phi must be in [-pi/2, pi/2]")
    if k * k >= 1.0:
        if k >= 1.0 - 1.0e-12:
            return abs(math.sin(phi))
        k = min(k, 1.0 - 1.0e-12)
    m = k * k
    s = phi
    power = phi
    coef = -1.0
    for n in range(1, n_terms + 1):
        coef *= (2 * n - 1) * (2 * n - 3) / (2 * n * (2 * n + 1))
        power *= m * phi * phi
        s += coef * power
        if abs(coef * power) < 1.0e-15 * abs(s):
            break
    return s


# =====================================================================
#                RIEMANN-INVARIANT ELLIPTIC PERIMETER
# =====================================================================

def riemann_ellipse_perimeter(cs_cgs: float, dv_cgs: float,
                               kappa_dispersion: float = 0.0) -> float:
    """
    Perimeter of the Riemann-invariant ellipse in the (x, t) plane for a
    dispersive sound wave.  With dispersion wavenumber kappa, the effective
    phase speed is

        c_eff = c_s / sqrt(1 + (kappa c_s / omega_0)^2),

    and the semi-axes of the characteristic ellipse after time T are

        a = (|dv| + c_eff) * T,    b = |c_eff - |dv|| * T.

    With T = 1 (normalised), the perimeter is 4 * a * E(e) where
    e = sqrt(1 - b^2 / a^2) is the eccentricity and E is the complete
    elliptic integral of the second kind.
    """
    if kappa_dispersion > 0.0:
        c_eff = cs_cgs / math.sqrt(1.0 + (kappa_dispersion * cs_cgs) ** 2)
    else:
        c_eff = cs_cgs
    a = abs(dv_cgs) + c_eff
    b = abs(c_eff - abs(dv_cgs))
    if a < 1.0e-60:
        return 0.0
    e = math.sqrt(max(1.0 - (b / a) ** 2, 0.0))
    return 4.0 * a * incomplete_elliptic_E(math.pi / 2.0, e)


# =====================================================================
#                        EOS WRAPPER
# =====================================================================

def make_default_eos() -> IdealGasEOS:
    """Factory for the standard simulation EOS."""
    return IdealGasEOS()
