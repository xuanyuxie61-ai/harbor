# -*- coding: utf-8 -*-
"""
================================================================================
LTP Wave Propagation Module (Fisher-KPP Reaction-Diffusion)
================================================================================

This module models the propagation of Long-Term Potentiation (LTP) waves
through neural tissue using the Fisher-Kolmogorov-Petrovsky-Piskunov (Fisher-KPP)
reaction-diffusion equation.

Biological Motivation:
----------------------
LTP can propagate as a regenerative wave of synaptic strengthening along
dendrites and through local networks. The wave front represents a transition
from a baseline synaptic weight state (w ≈ w_min) to a potentiated state
(w ≈ w_max).

Mathematical Model:
-------------------
The normalized synaptic weight u(x,t) ∈ [0,1] satisfies:

    ∂u/∂t = D · ∂²u/∂x² + r · u · (1 - u)

where:
    D = diffusion coefficient [μm²/ms] (spreads via diffusion of PRPs)
    r = growth rate [1/ms]       (local positive feedback)

Exact Traveling Wave Solution:
------------------------------
For the canonical form with D=1, r=1, an exact traveling wave is:

    u(x,t) = 1 / (1 + a · exp(k·(x - c·t)))²

where the wave parameters are:
    c = 5 / sqrt(6)   ≈ 2.0412   [wave speed]
    k = 1 / sqrt(6)   ≈ 0.4082   [wave steepness]
    a = 2.0                     [amplitude parameter]

The exact solution satisfies:
    u_t  =  2·c·a·k·exp(k·z) / (1 + a·exp(k·z))³
    u_x  = -2·a·k·exp(k·z) / (1 + a·exp(k·z))³
    u_xx =  6·a²·k²·exp(2k·z) / (1 + a·exp(k·z))⁴
           - 2·a·k²·exp(k·z) / (1 + a·exp(k·z))³

where z = x - c·t.

Method of Lines Discretization:
-------------------------------
Spatial discretization with finite differences converts the PDE to a system
of ODEs:

    du_i/dt = D·(u_{i-1} - 2u_i + u_{i+1})/h² + r·u_i·(1 - u_i)

This ODE system is then integrated using RK1 or RK4 from numerical_integrator.

Minimum Wave Speed:
-------------------
For the Fisher-KPP equation, any initial condition with compact support
evolves into a traveling wave with speed:

    c_min = 2 · sqrt(D·r)

This is known as the linear spreading speed or pulled front speed.

================================================================================
"""

import numpy as np
from typing import Tuple
from numerical_integrator import rk1_integrate, rk4_integrate


def fisher_exact_solution(
    x: np.ndarray,
    t: float,
    a: float = 2.0,
    c: float = None,
    k: float = None,
    D: float = 1.0,
    r: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the exact traveling wave solution of the Fisher-KPP equation.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates.
    t : float
        Time.
    a : float
        Amplitude parameter.
    c : float, optional
        Wave speed. Defaults to 5/sqrt(6) when D=r=1.
    k : float, optional
        Wave steepness. Defaults to 1/sqrt(6) when D=r=1.
    D : float
        Diffusion coefficient.
    r : float
        Growth rate.

    Returns
    -------
    u : np.ndarray
        Solution u(x,t).
    ut : np.ndarray
        Time derivative u_t(x,t).
    ux : np.ndarray
        Spatial derivative u_x(x,t).
    uxx : np.ndarray
        Second spatial derivative u_xx(x,t).
    """
    if c is None:
        c = 5.0 / np.sqrt(6.0)
    if k is None:
        k = 1.0 / np.sqrt(6.0)

    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")
    if a <= 0.0:
        raise ValueError("a must be positive.")

    z = x - c * t
    exp_kz = np.exp(k * z)
    denom = 1.0 + a * exp_kz
    denom2 = denom ** 2
    denom3 = denom ** 3
    denom4 = denom ** 4

    u = 1.0 / denom2
    ut = 2.0 * c * a * k * exp_kz / denom3
    ux = -2.0 * a * k * exp_kz / denom3
    uxx = 6.0 * (a ** 2) * (k ** 2) * np.exp(2.0 * k * z) / denom4 \
          - 2.0 * a * (k ** 2) * exp_kz / denom3

    # Clip u to [0,1] for numerical safety
    u = np.clip(u, 0.0, 1.0)

    return u, ut, ux, uxx


def fisher_wave_speed(D: float, r: float) -> float:
    """
    Compute the minimum Fisher-KPP wave speed.

    Formula: c_min = 2 · sqrt(D · r)

    Parameters
    ----------
    D : float
        Diffusion coefficient. Must be positive.
    r : float
        Growth rate. Must be positive.

    Returns
    -------
    c_min : float
        Minimum wave speed.
    """
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")
    return 2.0 * np.sqrt(D * r)


def build_fisher_rhs(
    n: int,
    h: float,
    D: float,
    r: float,
    bc: str = "NN",
) -> callable:
    """
    Build the right-hand side function for the method-of-lines ODE system.

    du_i/dt = D·(u_{i-1} - 2u_i + u_{i+1})/h² + r·u_i·(1 - u_i)

    Parameters
    ----------
    n : int
        Number of spatial points.
    h : float
        Grid spacing.
    D : float
        Diffusion coefficient.
    r : float
        Growth rate.
    bc : str
        Boundary condition type ('NN' for no-flux is natural for LTP waves).

    Returns
    -------
    rhs : callable
        Function f(t, u) -> du/dt.
    """
    if n < 3:
        raise ValueError("n must be >= 3.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")

    inv_h2 = 1.0 / (h * h)

    def rhs(t, u):
        # TODO: Implement the Fisher-KPP reaction-diffusion RHS.
        # This function must compute:
        #   du_i/dt = D * (discrete Laplacian of u)_i + r * u_i * (1 - u_i)
        # with boundary conditions: 'NN' (Neumann-Neumann), 'DD' (Dirichlet-Dirichlet),
        # or 'PP' (Periodic).
        # The discrete Laplacian depends on the boundary condition type.
        # Raise ValueError for unsupported bc.
        pass

    return rhs


def simulate_ltp_wave(
    n: int = 128,
    length: float = 200.0,
    D: float = 1.0,
    r: float = 1.0,
    t_final: float = 20.0,
    n_steps: int = 2000,
    bc: str = "NN",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simulate LTP wave propagation using the Fisher-KPP equation.

    Initial condition: localized region of high u (potentiated synapses)
    near the center, decaying to baseline.

    Parameters
    ----------
    n : int
        Number of spatial points.
    length : float
        Domain length [μm].
    D : float
        Diffusion coefficient [μm²/ms].
    r : float
        Growth rate [1/ms].
    t_final : float
        Final time [ms].
    n_steps : int
        Number of time steps.
    bc : str
        Boundary condition.

    Returns
    -------
    x : np.ndarray
        Spatial grid.
    t : np.ndarray
        Time points.
    u_history : np.ndarray
        Solution history, shape (n_steps+1, n).
    c_min : float
        Theoretical minimum wave speed.
    """
    if n < 3:
        raise ValueError("n must be >= 3.")
    if length <= 0.0:
        raise ValueError("length must be positive.")
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")

    h = length / n
    x = np.linspace(0.0, length, n)

    # Initial condition: step-like profile
    u0 = np.zeros(n)
    center_idx = n // 2
    width = n // 10
    u0[center_idx - width:center_idx + width] = 0.8
    # Smooth transition
    for i in range(n):
        u0[i] = 0.8 / (1.0 + np.exp(0.5 * (abs(i - center_idx) - width)))

    c_min = fisher_wave_speed(D, r)

    # Check CFL-like condition for explicit integration
    dt = t_final / n_steps
    dx2 = h * h
    if D * dt / dx2 > 0.5:
        n_steps = int(np.ceil(2.0 * D * t_final / dx2))
        dt = t_final / n_steps
        print(f"[plasticity_wave] n_steps adjusted to {n_steps} for stability")

    rhs = build_fisher_rhs(n, h, D, r, bc)
    t, u_history = rk4_integrate(rhs, (0.0, t_final), u0, n_steps)

    return x, t, u_history, c_min


def verify_fisher_exact(
    n: int = 64,
    length: float = 20.0,
    t_test: float = 2.0,
) -> float:
    """
    Verify the numerical solution against the exact traveling wave.

    Computes the L2 relative error:

        err = ||u_num - u_exact||₂ / ||u_exact||₂

    Parameters
    ----------
    n : int
        Number of spatial points.
    length : float
        Domain length.
    t_test : float
        Test time.

    Returns
    -------
    error : float
        Relative L2 error.
    """
    h = length / n
    x = np.linspace(0.0, length, n)

    # Exact solution at t=0 as initial condition
    u_exact_0, _, _, _ = fisher_exact_solution(x, 0.0)

    # Integrate to t_test
    D = 1.0
    r = 1.0
    rhs = build_fisher_rhs(n, h, D, r, bc="NN")
    n_steps = max(100, int(t_test * 500))
    _, u_history = rk4_integrate(rhs, (0.0, t_test), u_exact_0, n_steps)
    u_num = u_history[-1, :]

    u_exact, _, _, _ = fisher_exact_solution(x, t_test)

    denom = np.linalg.norm(u_exact)
    if denom < 1e-15:
        return np.linalg.norm(u_num - u_exact)
    return np.linalg.norm(u_num - u_exact) / denom


if __name__ == "__main__":
    x, t, u_hist, c = simulate_ltp_wave()
    print(f"LTP wave speed (theoretical min): {c:.4f} μm/ms")
    print(f"Wave front position at t={t[-1]}: ~{np.argmax(u_hist[-1] > 0.5) * (x[1]-x[0]):.2f} μm")
    err = verify_fisher_exact()
    print(f"Exact solution verification error: {err:.6e}")
