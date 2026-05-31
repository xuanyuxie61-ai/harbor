# -*- coding: utf-8 -*-
"""
Hamiltonian Structure and Dynamical System Analysis
====================================================
Analyzes the Hamiltonian structure of nonlinear PDE discretizations
using the Chirikov standard map and molecular dynamics energy tracking.

Inspired by:
- chirikov_iteration: standard map for area-preserving dynamics
- md_parfor: velocity Verlet integration and energy conservation

Mathematical formulation:
- A Hamiltonian system satisfies:
    dq/dt = dH/dp,   dp/dt = -dH/dq
- The Chirikov standard map is a symplectic discretization:
    y_{n+1} = y_n + K sin(x_n)   (mod 2pi)
    x_{n+1} = x_n + y_{n+1}      (mod 2pi)
  with Hamiltonian H = y^2/2 + K cos(x).
- For spectral PDEs, the semi-discrete Hamiltonian is:
    H = 0.5 ||p||^2 + 0.5 u^T L u + V(u)
  where L is the discrete Laplacian and V(u) is the nonlinear potential.
- Velocity Verlet for MD:
    x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt^2
    v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt
"""

import numpy as np


def chirikov_map_step(xy, K=0.55):
    """
    Apply one step of the Chirikov standard map.

    Parameters
    ----------
    xy : ndarray, shape (2,)
        Current state [x, y].
    K : float
        Perturbation parameter.

    Returns
    -------
    xy_new : ndarray
        Next state.
    """
    x, y = xy[0], xy[1]
    y_new = y + K * np.sin(x)
    x_new = x + y_new
    y_new = np.mod(y_new, 2.0 * np.pi)
    x_new = np.mod(x_new, 2.0 * np.pi)
    return np.array([x_new, y_new])


def chirikov_hamiltonian(xy, K=0.55):
    """
    Compute the Chirikov Hamiltonian H = y^2/2 + K cos(x).

    Parameters
    ----------
    xy : ndarray, shape (2,)
    K : float

    Returns
    -------
    H : float
        Hamiltonian value.
    """
    x, y = xy[0], xy[1]
    return 0.5 * y ** 2 + K * np.cos(x)


def chirikov_orbit(n_steps, xy0=None, K=0.55):
    """
    Generate a Chirikov orbit.

    Parameters
    ----------
    n_steps : int
    xy0 : ndarray, optional
        Initial condition. Default [0.6, 0.7].
    K : float

    Returns
    -------
    orbit : ndarray, shape (n_steps+1, 2)
    energy : ndarray
        Hamiltonian along orbit.
    """
    if xy0 is None:
        xy0 = np.array([0.6, 0.7])
    orbit = np.zeros((n_steps + 1, 2))
    energy = np.zeros(n_steps + 1)
    orbit[0] = xy0
    energy[0] = chirikov_hamiltonian(xy0, K)
    for i in range(n_steps):
        orbit[i + 1] = chirikov_map_step(orbit[i], K)
        energy[i + 1] = chirikov_hamiltonian(orbit[i + 1], K)
    return orbit, energy


def pde_hamiltonian(u, v, D2, nonlinear_potential_func=None, dx=1.0):
    """
    Compute the semi-discrete Hamiltonian of a nonlinear wave equation:
        H = 0.5 * v^T v + 0.5 * u^T (-D2) u + sum_i V(u_i)
    where v = du/dt is the momentum, D2 is the spectral Laplacian.

    Parameters
    ----------
    u : ndarray
        Displacement field.
    v : ndarray
        Velocity field.
    D2 : ndarray
        Spectral second-derivative matrix (negative definite for Laplacian).
    nonlinear_potential_func : callable, optional
        Nonlinear potential V(u_i) per grid point.
    dx : float
        Quadrature weight.

    Returns
    -------
    H : float
        Total Hamiltonian.
    H_parts : dict
        Kinetic, potential, and nonlinear contributions.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    kinetic = 0.5 * dx * np.dot(v, v)
    # Potential energy from Laplacian: -0.5 u^T D2 u = 0.5 u^T (-D2) u
    potential = -0.5 * dx * np.dot(u, D2 @ u)
    nonlinear = 0.0
    if nonlinear_potential_func is not None:
        nonlinear = dx * np.sum(nonlinear_potential_func(u))
    H = kinetic + potential + nonlinear
    return H, {"kinetic": kinetic, "potential": potential, "nonlinear": nonlinear}


def velocity_verlet_step(u, v, force_func, dt, mass=1.0):
    """
    One velocity Verlet step for Hamiltonian dynamics.
    Adapted from md_parfor molecular dynamics.

    Parameters
    ----------
    u : ndarray
        Positions (analogous to spectral coefficients or field values).
    v : ndarray
        Velocities.
    force_func : callable
        Force function F(u) = -dV/du.
    dt : float
        Time step.
    mass : float
        Mass parameter.

    Returns
    -------
    u_new, v_new : ndarray
        Updated positions and velocities.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = force_func(u) / mass
    u_new = u + v * dt + 0.5 * a * dt ** 2
    a_new = force_func(u_new) / mass
    v_new = v + 0.5 * (a + a_new) * dt
    return u_new, v_new


def ensemble_energy_drift(ensemble_u, ensemble_v, D2, dt_steps, force_func,
                          nonlinear_potential_func=None, dx=1.0):
    """
    Track energy conservation for an ensemble of initial conditions
    using velocity Verlet integration.

    Parameters
    ----------
    ensemble_u : ndarray, shape (n_ensemble, n_dof)
    ensemble_v : ndarray, shape (n_ensemble, n_dof)
    D2 : ndarray
    dt_steps : int
    force_func : callable
    nonlinear_potential_func : callable, optional
    dx : float

    Returns
    -------
    energy_drift : ndarray
        Relative energy drift for each ensemble member.
    """
    n_ens, n_dof = ensemble_u.shape
    energy_drift = np.zeros(n_ens)
    for e in range(n_ens):
        u = ensemble_u[e].copy()
        v = ensemble_v[e].copy()
        H0, _ = pde_hamiltonian(u, v, D2, nonlinear_potential_func, dx)
        for _ in range(dt_steps):
            u, v = velocity_verlet_step(u, v, force_func, 0.01)
        H1, _ = pde_hamiltonian(u, v, D2, nonlinear_potential_func, dx)
        if abs(H0) > 1e-15:
            energy_drift[e] = abs(H1 - H0) / abs(H0)
        else:
            energy_drift[e] = abs(H1 - H0)
    return energy_drift
