"""
waveform_template.py — Post-Newtonian gravitational-wave phase and strain.

Implements the stationary-phase approximation (SPA) and a time-domain
post-Newtonian (PN) model for the dominant (ell, m) = (2, 2) mode of a
quasi-circular compact binary.

Key equations
-------------
1. PN orbital frequency evolution (leading-order chirp):
       df/dt = (96/5) pi^{8/3} (G M_c / c^3)^{5/3} f^{11/3}
   with solution
       f(t) = f_0 (1 - t / t_merge)^{-3/8}
   where
       t_merge - t = (5/256) (G M_c / c^3)^{-5/3} (pi f)^{-8/3}.

2. SPA phase (TaylorF2) to 3.5 PN order:
       Psi(f) = 2 pi f t_c - phi_c - pi/4
                + (3/128) (pi M f)^{-5/3} sum_{k=0}^{7} phi_k (pi M f)^{k/3}
   with leading coefficients:
       phi_0 = 1,
       phi_2 = (3715/756 + 55 nu/9) - (11/6) chi_s,
       ...   (see Blanchet 2014, Living Rev. Rel. 17, 2).

3. Strain at Earth:
       h(t) = A(t) cos(Phi(t)),
       A(t) = (4 / D_L) (G M_c / c^2)^{5/6} (pi f(t) / c)^{2/3}.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Dict

from physics_constants import (PI, C_SI, G_SI, MSUN_SI, MPC_SI,
                               BinaryParameters)


# ---------------------------------------------------------------------------
#  PN coefficients (non-spinning, up to 3.5 PN)
# ---------------------------------------------------------------------------
def pn_coefficients(nu: float, chi_s: float = 0.0, chi_a: float = 0.0) -> Dict[int, float]:
    """Return phi_k coefficients in the TaylorF2 SPA phase.

    Parameters
    ----------
    nu    : symmetric mass ratio in (0, 1/4].
    chi_s : (chi1 + chi2)/2  (symmetric combination).
    chi_a : (chi1 - chi2)/2  (antisymmetric combination).

    The coefficients multiply (pi M f)^{k/3} relative to the leading order.
    """
    phi = {}
    phi[0] = 1.0
    phi[1] = 0.0
    phi[2] = (3715.0 / 756.0 + 55.0 * nu / 9.0) - 11.0 * chi_s
    phi[3] = -16.0 * PI - 113.0 / 3.0 * chi_a * (nu ** 0)
    phi[4] = (15293365.0 / 508032.0
              + 27145.0 * nu / 504.0
              + 3085.0 * nu ** 2 / 72.0)
    phi[5] = PI * (38645.0 / 756.0 - 65.0 * nu / 9.0) * (1.0 + 3.0 * math.log(1.0))
    phi[6] = (11583231236531.0 / 4694215680.0
              - 640.0 * PI ** 2 / 3.0
              - 6848.0 * (math.euler_gamma if hasattr(math, "euler_gamma") else 0.5772156649) / 21.0
              + nu * (-15737765635.0 / 3048192.0 + 2255.0 * PI ** 2 / 12.0)
              + 76055.0 * nu ** 2 / 1728.0
              - 127825.0 * nu ** 3 / 1296.0)
    phi[7] = PI * (77096675.0 / 254016.0 + 378515.0 * nu / 1512.0 - 74045.0 * nu ** 2 / 756.0)
    return phi


# ---------------------------------------------------------------------------
#  Time-domain leading-order chirp
# ---------------------------------------------------------------------------
def chirp_time(f: float, f_ref: float, Mc_sec: float) -> float:
    """Time for the GW frequency to sweep from f to f_ref (leading order).

        Delta t = (5/256) (pi M_c f)^{-8/3}  (in geometric seconds).
    """
    return (5.0 / 256.0) * (PI * Mc_sec * f) ** (-8.0 / 3.0)


def frequency_at_time(t_before_merger: float, Mc_sec: float) -> float:
    """Instantaneous GW frequency  f(t)  during the inspiral.

        f(t) = (5/256)^{3/8} (pi M_c)^{-5/8} (t_merge - t)^{-3/8}.
    """
    tau = max(t_before_merger, 1.0e-300)
    return (5.0 / 256.0) ** 0.375 * (PI * Mc_sec) ** (-0.625) * tau ** (-0.375)


def orbital_phase_at_time(t_before_merger: float, Mc_sec: float,
                          phi_c: float = 0.0) -> float:
    """Leading-order orbital phase phi(t) during inspiral.

        phi(t) = phi_c - (1/32) (pi M_c)^{-5/3} (t_merge - t)^{5/8} * (5/256)^{-5/8}
    simplified to  phi(t) = phi_c - (1/eta) ... with  eta = 5 (t_merge - t) / (256 Mc).
    """
    tau = max(t_before_merger, 1.0e-300)
    # phi(t) = phi_c - (5/256)^{-5/8} (pi M_c)^{-5/3} tau^{5/8} / 32
    prefactor = (5.0 / 256.0) ** (-5.0 / 8.0) * (PI * Mc_sec) ** (-5.0 / 3.0) / 32.0
    return phi_c - prefactor * tau ** (5.0 / 8.0)


# ---------------------------------------------------------------------------
#  Strain amplitude
# ---------------------------------------------------------------------------
def strain_amplitude(f: float, Mc_si: float, D_mpc: float) -> float:
    """Leading-order GW strain amplitude at luminosity distance D [Mpc].

        h(t) = 4 (G M_c / c^2)^{5/6} (pi f / c)^{2/3} / D.
    """
    D_m = D_mpc * MPC_SI
    Mc_m = G_SI * Mc_si / C_SI ** 2
    return 4.0 * Mc_m ** (5.0 / 6.0) * (PI * f / C_SI) ** (2.0 / 3.0) / D_m


# ---------------------------------------------------------------------------
#  Time-domain waveform generator
# ---------------------------------------------------------------------------
def generate_td_waveform(params: BinaryParameters,
                         n_points: int = 4096,
                         f_max_factor: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a leading-order time-domain waveform  h(t) = A(t) cos(Phi(t)).

    Returns
    -------
    t      : (n_points,) time array (seconds, ending at t_merge).
    h      : (n_points,) dimensionless strain.
    f_gw   : (n_points,) instantaneous GW frequency [Hz].
    """
    Mc_kg = params.chirp_mass * MSUN_SI
    Mc_sec = G_SI * Mc_kg / C_SI ** 3
    M_total_sec = G_SI * params.M_kg / C_SI ** 3
    f_low = params.f_low_hz
    f_max = f_max_factor * params.f_isco_hz
    # total duration
    T = chirp_time(f_low, f_max, Mc_sec)
    t = np.linspace(0.0, T, n_points)
    tau = T - t  # time before merger (decreasing)
    f_gw = np.array([frequency_at_time(tau_i, Mc_sec) for tau_i in tau])
    f_gw = np.clip(f_gw, f_low, f_max)
    A = np.array([strain_amplitude(f, Mc_kg, params.distance_mpc) for f in f_gw])
    # orbital phase by quadrature: Phi(t) = 2 pi int_0^t f(t') dt'
    # closed form:  Phi(t) = phi_c (1 - (tau/T)^{5/8})
    phi_c = 0.0
    Phi = 2.0 * phi_c * (1.0 - (tau / max(T, 1.0e-30)) ** (5.0 / 8.0))
    # use cumulative trapezoid for robustness
    dt = T / (n_points - 1)
    Phi_integral = np.cumsum(2.0 * PI * f_gw) * dt
    h = A * np.cos(Phi_integral)
    return t, h, f_gw


# ---------------------------------------------------------------------------
#  SPA waveform in the frequency domain
# ---------------------------------------------------------------------------
def generate_spa_waveform(params: BinaryParameters,
                          f_arr: np.ndarray) -> np.ndarray:
    """Return the complex SPA waveform  h_tilde(f) = A_tilde(f) exp(i Psi(f))."""
    Mc_kg = params.chirp_mass * MSUN_SI
    Mc_sec = G_SI * Mc_kg / C_SI ** 3
    M_total_sec = G_SI * params.M_kg / C_SI ** 3
    nu = params.symmetric_ratio
    phi = pn_coefficients(nu, params.chi_eff, 0.0)
    D_m = params.distance_mpc * MPC_SI
    # amplitude  A_tilde(f) = sqrt(5/24) pi^{-2/3} D^{-1} (G M_c / c^2)^{5/6} f^{-7/6}
    Mc_m = G_SI * Mc_kg / C_SI ** 2
    A_tilde = (math.sqrt(5.0 / 24.0) * PI ** (-2.0 / 3.0)
               / D_m * Mc_m ** (5.0 / 6.0) * f_arr ** (-7.0 / 6.0))
    x_arr = (PI * M_total_sec * f_arr) ** (1.0 / 3.0)
    Psi = np.zeros_like(f_arr)
    for k, coeff in phi.items():
        Psi += coeff * x_arr ** k
    Psi *= (3.0 / 128.0) * (PI * M_total_sec * f_arr) ** (-5.0 / 3.0)
    Psi += -PI / 4.0
    return A_tilde * np.exp(1j * Psi)
