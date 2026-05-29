"""
Particle Acceleration Module
============================
Based on seed project 1171_stochastic_rk:
- rk4_ti_step.m  →  stochastic Runge-Kutta for Fokker-Planck

Physics:
--------
The energy evolution of relativistic electrons in GRB shocks is
described by the Fokker-Planck equation:

    ∂N(γ,t)/∂t = ∂/∂γ [ D(γ) ∂N/∂γ ] - ∂/∂γ [ A(γ) N ] + Q(γ,t) - N/τ_esc

where:
    γ      = electron Lorentz factor
    D(γ)   = diffusion coefficient in energy space
    A(γ)   = advection (systematic acceleration) coefficient
    Q(γ,t) = injection rate
    τ_esc  = escape time scale

In the test-particle limit of diffusive shock acceleration (DSA),
the systematic energy gain rate is:

    A(γ) = (4/3) (u₁ - u₂)/c · γ

with u₁, u₂ the upstream and downstream flow velocities.
The diffusion coefficient scales as Bohm diffusion:

    D(γ) = (1/3) r_L c = (1/3) (γ m_e c² / (e B)) c

This module implements a 4th-order stochastic Runge-Kutta scheme
(Kasdin 1995) for the SDE:

    dγ/dt = A(γ) + √(2D(γ)) · η(t)

where η(t) is white noise with ⟨η(t)η(t')⟩ = δ(t-t').
"""

import numpy as np


# Kasdin RK4-TI coefficients (time-invariant)
_A21 = 2.71644396264860
_A31 = -6.95653259006152
_A32 = 0.78313689457981
_A41 = 0.0
_A42 = 0.48257353309214
_A43 = 0.26171080165848
_A51 = 0.47012396888046
_A52 = 0.36597075368373
_A53 = 0.08906615686702
_A54 = 0.07483912056879

_Q1 = 2.12709852335625
_Q2 = 2.73245878238737
_Q3 = 11.22760917474960
_Q4 = 13.36199560336697


def bohm_diffusion(gamma, B):
    """
    Bohm diffusion coefficient in energy space:

        D(γ) = (1/3) · r_L · c = (1/3) · (γ m_e c² / (e B)) · c

    Units: erg² / s  (after multiplying by (m_e c²)²).
    """
    m_e = 9.10938356e-28   # g
    c = 2.99792458e10      # cm/s
    e = 4.80320427e-10     # statcoulomb

    r_L = gamma * m_e * c ** 2 / (e * B)
    D = (1.0 / 3.0) * r_L * c
    # Scale to dimensionless energy-space diffusion
    return D


def acceleration_coefficient(gamma, u1, u2):
    """
    Systematic acceleration rate in diffusive shock acceleration:

        A(γ) = (4/3) · (u1 - u2)/c · γ

    For a strong relativistic shock with compression ratio r = u1/u2 = 4,
    A(γ) ≈ (1/4) γ / t_cross.
    """
    c = 2.99792458e10
    beta_sh = (u1 - u2) / c
    beta_sh = np.clip(beta_sh, 0.0, 1.0)
    return (4.0 / 3.0) * beta_sh * gamma


def fi_dsa(gamma, u1, u2):
    """Deterministic drift term dγ/dt = A(γ)."""
    return acceleration_coefficient(gamma, u1, u2)


def gi_dsa(gamma, B):
    """Stochastic diffusion amplitude √(2D(γ))."""
    D = bohm_diffusion(gamma, B)
    D = max(D, 0.0)
    return np.sqrt(2.0 * D)


def rk4_ti_step(x, t, h, q, fi, gi):
    """
    One step of a 4th-order stochastic Runge-Kutta scheme
    for time-invariant SDEs (Kasdin 1995).

    dX/dt = F(X) + G(X) · w(t)

    Parameters
    ----------
    x : float
        Current state (e.g., log γ).
    t : float
        Current time.
    h : float
        Time step.
    q : float
        Spectral density of input white noise.
    fi : callable
        Deterministic drift F(X).
    gi : callable
        Stochastic amplitude G(X).

    Returns
    -------
    xstar : float
        State at t+h.
    """
    n1 = np.random.randn()
    w1 = n1 * np.sqrt(_Q1 * q / h)
    k1 = h * fi(x) + h * gi(x) * w1

    t2 = t + _A21 * h
    x2 = x + _A21 * k1
    n2 = np.random.randn()
    w2 = n2 * np.sqrt(_Q2 * q / h)
    k2 = h * fi(x2) + h * gi(x2) * w2

    t3 = t + (_A31 + _A32) * h
    x3 = x + _A31 * k1 + _A32 * k2
    n3 = np.random.randn()
    w3 = n3 * np.sqrt(_Q3 * q / h)
    k3 = h * fi(x3) + h * gi(x3) * w3

    t4 = t + (_A41 + _A42 + _A43) * h
    x4 = x + _A41 * k1 + _A42 * k2  # Note: original has this pattern
    n4 = np.random.randn()
    w4 = n4 * np.sqrt(_Q4 * q / h)
    k4 = h * fi(x4) + h * gi(x4) * w4

    xstar = x + _A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4
    return xstar


def accelerate_electrons(gamma_0, n_particles, t_max, dt, B, u1, u2,
                         q_noise=1.0, gamma_max=1e8):
    """
    Stochastically accelerate a population of electrons via DSA.

    Parameters
    ----------
    gamma_0 : float
        Initial Lorentz factor.
    n_particles : int
        Number of Monte-Carlo particles.
    t_max : float
        Total integration time (s).
    dt : float
        Time step (s).
    B : float
        Magnetic field strength (Gauss).
    u1, u2 : float
        Upstream / downstream velocities (cm/s).
    q_noise : float
        Noise spectral density.
    gamma_max : float
        Maximum Lorentz factor (cutoff).

    Returns
    -------
    gamma_final : ndarray, shape (n_particles,)
        Final Lorentz factors.
    """
    n_steps = max(1, int(t_max / dt))
    gamma = np.full(n_particles, float(gamma_0))

    for _ in range(n_steps):
        for i in range(n_particles):
            gamma[i] = rk4_ti_step(
                gamma[i], 0.0, dt, q_noise,
                lambda x: fi_dsa(x, u1, u2),
                lambda x: gi_dsa(x, B)
            )
        # Hard cutoff at gamma_max
        gamma = np.clip(gamma, 1.0, gamma_max)

    return gamma
