# -*- coding: utf-8 -*-
"""
Theta-Method Time Integration for Spectral PDEs
================================================
Implements the one-parameter theta method for ODE/PDE systems:
    Y_{n+1} = Y_n + dt * [ theta * F(t_n, Y_n) + (1-theta) * F(t_{n+1}, Y_{n+1}) ]

Special cases:
- theta = 1.0  : Forward Euler (explicit)
- theta = 0.5  : Crank-Nicolson (second-order, A-stable)
- theta = 0.0  : Backward Euler (implicit, L-stable)

For nonlinear systems, a simplified Newton iteration is used at each step.
Energy monitoring tracks the discrete Hamiltonian/energy functional.
"""

import numpy as np


def theta_residual(F, t_old, y_old, t_new, y_new, theta, dt):
    """
    Compute the theta-method residual:
        R = y_new - y_old - dt * [theta * F(t_old, y_old) + (1-theta) * F(t_new, y_new)]

    Parameters
    ----------
    F : callable
        Right-hand side function F(t, y).
    t_old, t_new : float
        Time levels.
    y_old, y_new : ndarray
        Solution vectors.
    theta : float
        Method parameter in [0, 1].
    dt : float
        Time step.

    Returns
    -------
    residual : ndarray
        Residual vector.
    """
    f_old = F(t_old, y_old)
    f_new = F(t_new, y_new)
    return y_new - y_old - dt * (theta * f_old + (1.0 - theta) * f_new)


def theta_step(F, t_old, y_old, dt, theta, newton_tol=1e-10, newton_max_iter=20,
               jacobian_approx=None):
    """
    Advance one time step using the theta method with simplified Newton iteration.

    Parameters
    ----------
    F : callable
        Right-hand side function.
    t_old : float
        Current time.
    y_old : ndarray
        Current solution.
    dt : float
        Time step size.
    theta : float
        Theta parameter.
    newton_tol : float
        Newton convergence tolerance.
    newton_max_iter : int
        Maximum Newton iterations.
    jacobian_approx : callable, optional
        Function to compute an approximate Jacobian matrix J = dF/dy.
        If None, a diagonal approximation is used.

    Returns
    -------
    y_new : ndarray
        Solution at t_new = t_old + dt.
    info : dict
        Newton iteration count and residual.
    """
    # TODO: Implement one theta-method time step with Newton iteration.
    # This requires:
    #   1. Computing explicit Euler predictor as initial guess
    #   2. Iteratively computing the theta-method residual via theta_residual
    #   3. Building JG = I - dt*(1-theta)*JF and solving for dy
    #   4. Checking convergence and returning y_new with iteration info
    raise NotImplementedError("Hole 3: theta_step is missing.")


def theta_method_integrate(F, t_span, y0, n_steps, theta=0.5,
                           newton_tol=1e-10, jacobian_approx=None,
                           energy_func=None):
    """
    Integrate an ODE system using the theta method over a time interval.

    Parameters
    ----------
    F : callable
        Right-hand side F(t, y).
    t_span : tuple (t0, tf)
        Time interval.
    y0 : ndarray
        Initial condition.
    n_steps : int
        Number of time steps.
    theta : float
        Theta parameter (default 0.5 for Crank-Nicolson).
    newton_tol : float
        Newton tolerance.
    jacobian_approx : callable, optional
        Jacobian approximation.
    energy_func : callable, optional
        Energy functional E(t, y) for monitoring.

    Returns
    -------
    t : ndarray
        Time levels.
    y : ndarray
        Solution history, shape (n_steps+1, len(y0)).
    energy : ndarray, optional
        Energy history if energy_func provided.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    m = len(y0)
    t0, tf = t_span
    dt = (tf - t0) / n_steps

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    t[0] = t0
    y[0, :] = y0

    energy = None
    if energy_func is not None:
        energy = np.zeros(n_steps + 1)
        energy[0] = energy_func(t0, y0)

    for i in range(n_steps):
        y_new, info = theta_step(F, t[i], y[i, :], dt, theta,
                                 newton_tol=newton_tol,
                                 jacobian_approx=jacobian_approx)
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y_new
        if energy_func is not None:
            energy[i + 1] = energy_func(t[i + 1], y_new)

    if energy_func is not None:
        return t, y, energy
    return t, y


def discrete_energy_norm(u, D2, dx_weight=None):
    """
    Compute a discrete energy norm for PDE solutions:
        E(u) = 0.5 * (||u||^2 + ||grad u||^2)
    approximated spectrally.

    Parameters
    ----------
    u : ndarray
        Solution vector.
    D2 : ndarray
        Spectral second-derivative matrix (used to approximate gradient energy).
    dx_weight : float, optional
        Spatial quadrature weight.

    Returns
    -------
    energy : float
        Discrete energy.
    """
    u = np.asarray(u, dtype=np.float64)
    kinetic = 0.5 * np.dot(u, u)
    if dx_weight is not None:
        kinetic *= dx_weight
    # Potential energy: -0.5 * u^T * D2 * u approximates 0.5 * ||grad u||^2
    potential = -0.5 * np.dot(u, D2 @ u)
    if dx_weight is not None:
        potential *= dx_weight
    return kinetic + potential
