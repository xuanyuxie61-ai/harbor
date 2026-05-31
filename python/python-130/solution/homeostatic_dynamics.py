# -*- coding: utf-8 -*-
"""
================================================================================
Homeostatic Synaptic Weight Regulation Module
================================================================================

This module models homeostatic regulation of synaptic weights using:
1. Damped harmonic oscillator (spring-mass-damper system)
2. Nonlinear pendulum dynamics for network synchronization

Biological Motivation:
----------------------
Synaptic weights are subject to homeostatic plasticity that maintains
network activity within a functional range. When weights deviate from
target values, compensatory mechanisms push them back.

The homeostatic force can be modeled as:

    F_homeo = -k·(w - w_target) - b·(dw/dt)

where:
    k = homeostatic stiffness
    b = adaptation rate (damping)
    w_target = target synaptic weight

1. Damped Harmonic Oscillator Model:
------------------------------------
    m·d²w/dt² + b·dw/dt + k·(w - w_target) = F_plasticity(t)

where F_plasticity(t) represents Hebbian/LTP driving forces.

The characteristic equation is:
    m·λ² + b·λ + k = 0

Roots:
    λ = (-b ± sqrt(b² - 4mk)) / (2m)

- Underdamped (b² < 4mk): oscillatory return to target
- Critically damped (b² = 4mk): fastest non-oscillatory return
- Overdamped (b² > 4mk): slow monotonic return

2. Nonlinear Pendulum Model:
----------------------------
For network-level phase synchronization, we model each neuron as a
nonlinear pendulum:

    d²θ/dt² + (g/l)·sin(θ) = I_ext(t)

where θ is the phase difference between coupled oscillators.

Exact Solution (for undamped, zero external drive):
---------------------------------------------------
With initial conditions θ(0) = θ₀, dθ/dt(0) = 0:

    sin(θ₀/2) = k₀
    ω = sqrt(g/l)
    k = k₀    (elliptic modulus)

    θ(t) = 2·arcsin(k·sn(ωt, k))

dθ/dt(t) = 2·k·ω·cn(ωt, k)

where sn(u,k) and cn(u,k) are Jacobi elliptic functions.

The exact period is:
    T = 4·K(k) / ω

where K(k) is the complete elliptic integral of the first kind.

================================================================================
"""

import numpy as np
from scipy.special import ellipk, ellipj
from typing import Tuple, Optional
from numerical_integrator import rk1_integrate, rk4_integrate


def spring_deriv(t: float, y: np.ndarray, m: float, b: float, k: float, F_ext: float = 0.0) -> np.ndarray:
    """
    Right-hand side of the damped spring ODE for synaptic homeostasis.

    State vector: y = [w, v] where w = weight, v = dw/dt

    dy/dt = [v, -(k/m)·(w - w_target) - (b/m)·v + F_ext/m]

    Parameters
    ----------
    t : float
        Current time.
    y : np.ndarray
        State vector [w, v].
    m : float
        Inertial parameter (must be positive).
    b : float
        Damping coefficient (must be non-negative).
    k : float
        Stiffness (must be non-negative).
    F_ext : float
        External plasticity force.

    Returns
    -------
    dydt : np.ndarray
        Time derivatives.
    """
    if m <= 0.0:
        raise ValueError("m must be positive.")
    if b < 0.0 or k < 0.0:
        raise ValueError("b and k must be non-negative.")

    w = y[0]
    v = y[1]

    dudt = v
    dvdt = -(k / m) * w - (b / m) * v + F_ext / m

    return np.array([dudt, dvdt])


def classify_damping(m: float, b: float, k: float) -> str:
    """
    Classify the damping regime of the harmonic oscillator.

    discriminant = b² - 4mk

    Returns
    -------
    regime : str
        'underdamped', 'critically_damped', or 'overdamped'.
    """
    if m <= 0.0 or k < 0.0 or b < 0.0:
        raise ValueError("Invalid parameters.")

    disc = b * b - 4.0 * m * k
    if disc < 0.0:
        return "underdamped"
    elif np.isclose(disc, 0.0):
        return "critically_damped"
    else:
        return "overdamped"


def spring_parameters(
    m: float = 1.0,
    b: float = 0.5,
    k: float = 1.0,
    w_target: float = 0.5,
) -> dict:
    """
    Compute derived spring parameters.

    Parameters
    ----------
    m, b, k : float
        Mass, damping, stiffness.
    w_target : float
        Target weight.

    Returns
    -------
    params : dict
        Derived parameters: omega_n, zeta, tau, regime.
    """
    omega_n = np.sqrt(k / m)
    zeta = b / (2.0 * np.sqrt(m * k)) if k > 0 else np.inf
    tau = 2.0 * m / b if b > 0 else np.inf
    regime = classify_damping(m, b, k)

    return {
        "omega_n": omega_n,
        "zeta": zeta,
        "tau": tau,
        "regime": regime,
        "w_target": w_target,
    }


def simulate_homeostatic_response(
    w0: float = 0.2,
    v0: float = 0.0,
    m: float = 1.0,
    b: float = 0.5,
    k: float = 1.0,
    F_ext: float = 0.0,
    t_final: float = 50.0,
    n_steps: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate homeostatic return of synaptic weight to target.

    Parameters
    ----------
    w0 : float
        Initial weight deviation.
    v0 : float
        Initial rate of change.
    m, b, k : float
        Spring parameters.
    F_ext : float
        Constant external plasticity force.
    t_final : float
        Simulation time.
    n_steps : int
        Number of steps.

    Returns
    -------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Solution [w, v].
    params : dict
        Derived parameters.
    """
    params = spring_parameters(m, b, k)

    def rhs(t, y):
        return spring_deriv(t, y, m, b, k, F_ext)

    t, y = rk4_integrate(rhs, (0.0, t_final), np.array([w0, v0]), n_steps)
    return t, y, params


def pendulum_nonlinear_exact(
    t: np.ndarray,
    g: float = 9.81,
    l: float = 1.0,
    theta0: float = np.pi / 3.0,
    thetadot0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact solution of the nonlinear pendulum using Jacobi elliptic functions.

    The ODE is:
        d²θ/dt² + (g/l)·sin(θ) = 0

    Exact solution (Ochs 2011):
        k₀ = sin(θ₀/2)
        ω = sqrt(g/l)
        E₀ = θ̇₀² + 4ω²k₀²
        k = sqrt(E₀) / (2ω)   [elliptic modulus]

        If k ≤ 1 (librations):
            θ(t) = 2·arcsin(k·sn(ωt + φ₀, k))
            θ̇(t) = 2kω·cn(ωt + φ₀, k)
        If k > 1 (rotations):
            Use reciprocal modulus transformation.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    g : float
        Gravitational acceleration analog.
    l : float
        Length parameter.
    theta0 : float
        Initial angle.
    thetadot0 : float
        Initial angular velocity.

    Returns
    -------
    theta : np.ndarray
        Angular displacement.
    thetadot : np.ndarray
        Angular velocity.
    """
    if l <= 0.0:
        raise ValueError("l must be positive.")

    omega = np.sqrt(g / l)
    k0 = np.sin(theta0 / 2.0)
    ep = 4.0 * g / l
    e0 = thetadot0 ** 2 + ep * k0 ** 2

    if e0 < 1e-15:
        return np.zeros_like(t), np.zeros_like(t)

    k = np.sqrt(e0 / ep)

    # For the exact solution, we use the amplitude formulation
    # sn(u, m) where m = k²
    m = min(k ** 2, 0.999999)  # clamp for numerical stability

    # Phase offset
    if abs(k0) > 1e-15:
        # Compute initial phase using inverse sn
        sn_val = k0 / k if k > 1e-15 else 0.0
        sn_val = np.clip(sn_val, -1.0, 1.0)
        # ellipj requires u and m
        # We solve for u0 such that sn(u0, m) = sn_val
        # Use arcsin approximation for small angles
        if m < 0.99:
            u0 = ellipk(m) * np.arcsin(sn_val) / (np.pi / 2)
        else:
            u0 = np.arcsin(sn_val)
    else:
        u0 = 0.0

    # Compute elliptic functions
    u = omega * t + u0
    sn, cn, dn = ellipj(u, m)

    # Exact solution
    theta = 2.0 * np.sign(cn) * np.arcsin(np.clip(np.abs(k * sn), 0.0, 1.0))
    thetadot = np.sign(thetadot0) * np.sqrt(e0) * cn

    return theta, thetadot


def compute_pendulum_period(
    g: float = 9.81,
    l: float = 1.0,
    theta0: float = np.pi / 3.0,
) -> float:
    """
    Compute the exact period of the nonlinear pendulum.

    Formula:
        T = 4·K(k) / ω

    where K(k) is the complete elliptic integral of the first kind
    and k = sin(θ₀/2).

    Parameters
    ----------
    g : float
        Acceleration parameter.
    l : float
        Length parameter.
    theta0 : float
        Initial amplitude.

    Returns
    -------
    T : float
        Period.
    """
    if l <= 0.0:
        raise ValueError("l must be positive.")

    omega = np.sqrt(g / l)
    k = np.sin(theta0 / 2.0)
    k2 = k ** 2

    K = ellipk(k2)
    T = 4.0 * K / omega
    return T


def simulate_network_synchronization(
    n_neurons: int = 10,
    g: float = 1.0,
    l: float = 1.0,
    coupling: float = 0.1,
    t_final: float = 20.0,
    n_steps: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a network of coupled nonlinear pendulums representing
    neural phase oscillators with synaptic coupling.

    The network dynamics are:

        d²θ_i/dt² = -(g/l)·sin(θ_i) + coupling·Σ_j sin(θ_j - θ_i)

    Parameters
    ----------
    n_neurons : int
        Number of oscillators.
    g, l : float
        Pendulum parameters.
    coupling : float
        Synaptic coupling strength.
    t_final : float
        Simulation time.
    n_steps : int
        Number of steps.
    seed : int
        Random seed.

    Returns
    -------
    t : np.ndarray
        Time points.
    theta : np.ndarray
        Phases, shape (n_steps+1, n_neurons).
    thetadot : np.ndarray
        Phase velocities.
    """
    if n_neurons < 1:
        raise ValueError("n_neurons must be >= 1.")
    if l <= 0.0 or g < 0.0:
        raise ValueError("Invalid pendulum parameters.")

    rng = np.random.default_rng(seed)

    # Initial conditions
    theta0 = rng.uniform(-np.pi / 2.0, np.pi / 2.0, n_neurons)
    thetadot0 = rng.uniform(-0.5, 0.5, n_neurons)
    y0 = np.concatenate([theta0, thetadot0])

    def rhs(t, y):
        theta = y[:n_neurons]
        thetadot = y[n_neurons:]

        # Intrinsic pendulum forces
        dtheta = thetadot
        dthetadot = -(g / l) * np.sin(theta)

        # Synaptic coupling (Kuramoto-like)
        for i in range(n_neurons):
            dthetadot[i] += coupling * np.sum(np.sin(theta - theta[i]))

        return np.concatenate([dtheta, dthetadot])

    t, y_history = rk4_integrate(rhs, (0.0, t_final), y0, n_steps)
    theta = y_history[:, :n_neurons]
    thetadot = y_history[:, n_neurons:]

    return t, theta, thetadot


def simulate_homeostatic_plasticity_pipeline(
    n_synapses: int = 5,
    t_final: float = 30.0,
) -> dict:
    """
    Run a comprehensive homeostatic plasticity simulation.

    Parameters
    ----------
    n_synapses : int
        Number of synapses.
    t_final : float
        Simulation time.

    Returns
    -------
    results : dict
        Simulation results.
    """
    rng = np.random.default_rng(130)

    results = []
    for i in range(n_synapses):
        w0 = rng.uniform(0.1, 0.9)
        m = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.2, 1.0)
        k = rng.uniform(0.5, 2.0)
        F = rng.uniform(-0.1, 0.1)

        t, y, params = simulate_homeostatic_response(
            w0=w0, m=m, b=b, k=k, F_ext=F, t_final=t_final, n_steps=3000
        )
        results.append({
            "t": t,
            "w": y[:, 0],
            "v": y[:, 1],
            "params": params,
        })

    # Pendulum synchronization
    t_net, theta_net, thetadot_net = simulate_network_synchronization(
        n_neurons=8, t_final=t_final, n_steps=3000
    )

    return {
        "synapses": results,
        "network_t": t_net,
        "network_theta": theta_net,
        "network_thetadot": thetadot_net,
    }


if __name__ == "__main__":
    t, y, params = simulate_homeostatic_response()
    print(f"Damping regime: {params['regime']}")
    print(f"Natural frequency: {params['omega_n']:.4f}")
    print(f"Final weight: {y[-1, 0]:.6f}")

    theta, thetadot = pendulum_nonlinear_exact(np.linspace(0, 10, 100))
    print(f"Pendulum period: {compute_pendulum_period():.4f}")
