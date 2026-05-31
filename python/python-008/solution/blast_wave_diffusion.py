"""
Blast Wave Diffusion Module
===========================
Based on seed project 901_porous_medium_exact:
- porous_medium_exact.m      →  nonlinear diffusion for blast wave expansion
- porous_medium_parameters.m →  parameter management

Physics:
--------
The expansion of a relativistic GRB blast wave into the ISM can be
modelled by a nonlinear diffusion equation analogous to the porous
medium equation (PME):

    ∂u/∂t = Δ(u^m)

For the GRB afterglow, we interpret u as the comoving energy density
e(t,r) of the shocked fluid, and m > 1 as a polytropic index related
to the adiabatic index of the relativistic gas.

The Barenblatt self-similar solution gives the exact energy-density
profile of the blast wave:

    α = 1 / (m - 1)
    β = 1 / (m + 1)
    γ_pme = (m - 1) / (2 m (m + 1))

    u(r,t) = (t + δ)^{-β} · [ C - γ_pme · (r / (t + δ)^β )² ]_{+}^{α}

where [x]_{+} = max(x, 0).  This describes a blast wave with a sharp
front at:

    r_f(t) = √(C / γ_pme) · (t + δ)^β

The total energy inside the blast wave is:

    E(t) = ∫_0^{r_f(t)} 4π r² u(r,t) dr
         ∝ (t + δ)^{-β(3α-1)} = (t + δ)^{-β(3/(m-1)-1)}

For m = 3, we have α = 1/2, β = 1/4, and the Sedov-like energy
decay E ∝ t^{-1/4} in this nonlinear diffusion framework.
"""

import numpy as np


# Module-level defaults (mimicking persistent MATLAB defaults)
_DEFAULTS = {
    "c": np.sqrt(3.0) / 15.0,
    "delta": 1.0 / 75.0,
    "m": 3.0,
    "t0": 0.0,
    "tstop": 4.0,
}


def porous_medium_parameters(c_user=None, delta_user=None, m_user=None,
                             t0_user=None, tstop_user=None):
    """
    Returns parameters for the porous-medium blast-wave model.

    Any supplied arguments override the defaults.
    """
    params = dict(_DEFAULTS)
    if c_user is not None:
        params["c"] = float(c_user)
    if delta_user is not None:
        params["delta"] = float(delta_user)
    if m_user is not None:
        params["m"] = float(m_user)
    if t0_user is not None:
        params["t0"] = float(t0_user)
    if tstop_user is not None:
        params["tstop"] = float(tstop_user)
    return params


def porous_medium_exact(x, t, params=None):
    """
    Evaluates the Barenblatt self-similar solution of the porous medium
    equation and its derivatives.

    Parameters
    ----------
    x : float or ndarray
        Radial coordinate (normalized).
    t : float or ndarray
        Time coordinate.
    params : dict, optional
        PME parameters.

    Returns
    -------
    u, ut, ux, uxx : ndarray
        Solution and derivatives.
    """
    if params is None:
        params = porous_medium_parameters()

    c = params["c"]
    delta = params["delta"]
    m = params["m"]

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    scalar_input = (x.ndim == 0 and t.ndim == 0)
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    # Broadcast to common shape
    x, t = np.broadcast_arrays(x, t)

    bot = (t + delta) ** beta
    factor = c - gamma * (x / bot) ** 2

    # Initialize outputs
    u = np.zeros_like(factor)
    ut = np.zeros_like(factor)
    ux = np.zeros_like(factor)
    uxx = np.zeros_like(factor)

    mask = factor > 0.0
    if np.any(mask):
        f = factor[mask]
        u[mask] = (t[mask] + delta) ** (-beta) * f ** alpha

        ut[mask] = (2.0 * alpha * beta * gamma
                    * (t[mask] + delta) ** (-1.0 - 3.0 * beta)
                    * x[mask] ** 2 * f ** (alpha - 1.0)
                    - beta * (t[mask] + delta) ** (-1.0 - beta) * f ** alpha)

        ux[mask] = (-2.0 * alpha * gamma
                    * (t[mask] + delta) ** (-3.0 * beta)
                    * x[mask] * f ** (alpha - 1.0))

        uxx[mask] = (4.0 * (alpha - 1.0) * alpha * gamma ** 2
                     * (t[mask] + delta) ** (-5.0 * beta)
                     * x[mask] ** 2 * f ** (alpha - 2.0)
                     - 2.0 * alpha * gamma
                     * (t[mask] + delta) ** (-3.0 * beta)
                     * f ** (alpha - 1.0))

    if scalar_input:
        return u.item(), ut.item(), ux.item(), uxx.item()
    return u, ut, ux, uxx


def blast_wave_energy_density_profile(r_cm, t_s, E_iso=1e53,
                                      n_ism=1.0, gamma_ad=4.0 / 3.0):
    """
    Computes the post-shock energy density of a GRB blast wave using
    the PME self-similar solution mapped to physical units.

    The mapping is:

        ε(r,t) = E_iso / (4π r_dec³) · u(ξ, τ)

    where ξ = r / r_dec, τ = t / t_dec, and the deceleration radius
    and time are:

        r_dec = (3 E_iso / (4π n_ism m_p c² Γ_0²) )^{1/3}
        t_dec = r_dec / (2 Γ_0² c)

    Parameters
    ----------
    r_cm : ndarray
        Radial coordinate in cm.
    t_s : float
        Time in seconds.
    E_iso : float
        Isotropic equivalent energy in erg.
    n_ism : float
        ISM number density in cm^{-3}.
    gamma_ad : float
        Adiabatic index.

    Returns
    -------
    eps : ndarray
        Post-shock energy density in erg cm^{-3}.
    """
    m_p = 1.6726219e-24  # g
    c = 2.99792458e10    # cm/s
    Gamma_0 = 300.0

    r_dec = ((3.0 * E_iso) / (4.0 * np.pi * n_ism * m_p * c ** 2 * Gamma_0 ** 2)) ** (1.0 / 3.0)
    t_dec = r_dec / (2.0 * Gamma_0 ** 2 * c)

    # Robustness: ensure t_dec > 0
    if t_dec <= 0.0:
        t_dec = 1.0
    if r_dec <= 0.0:
        r_dec = 1.0

    xi = r_cm / r_dec
    tau = t_s / t_dec

    # Map adiabatic index to PME exponent: m = (γ_ad + 1) / (γ_ad - 1)
    m_pme = (gamma_ad + 1.0) / (gamma_ad - 1.0)
    params = porous_medium_parameters(m_user=m_pme)

    u, _, _, _ = porous_medium_exact(xi, tau, params=params)

    scale = E_iso / (4.0 * np.pi * r_dec ** 3)
    eps = scale * u
    eps = np.clip(eps, 0.0, None)
    return eps
