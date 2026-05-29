"""
time_integrator.py
==================
Time integration schemes for coupled poroelastic ODE system.

Incorporates:
  - Midpoint implicit-explicit rule (from midpoint)
  - IMEX splitting for operator splitting (from fd1d_predator_prey)
  - Exponential integrator exact solution for linear test (from exp_ode)

The coupled Biot system after spatial discretization reads:

    [ M_uu   0  ] [ ddot(u) ]   [ K_uu   C  ] [ u ]   [ F_u ]
    [ C^T   M_p ] [ dot(p)  ] + [ 0      K_p] [ p ] = [ F_p ]

For quasi-static consolidation (neglecting solid inertia):
    K_uu * u + C * p = F_u
    C^T * dot(u) + M_p * dot(p) + K_p * p = F_p

This is a differential-algebraic system. We use a first-order form:
    dot(y) = f(y, t)
where y = [u; p] and the algebraic constraint is handled by solving
K_uu*u = F_u - C*p at each time step.
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def midpoint_implicit_step(K_uu, C, M_p, K_p, F_u, F_p, u_n, p_n, dt,
                           solver_type="dense"):
    """
    Single time step of the midpoint rule for Biot consolidation.

    Step 1: Solve algebraic system for displacement
        K_uu * u_{n+1} = F_u_{n+1} - C * p_n

    Step 2: Update pressure implicitly
        (M_p + 0.5*dt*K_p) * p_{n+1} = (M_p - 0.5*dt*K_p) * p_n
                                      - C^T * (u_{n+1} - u_n)
                                      + 0.5*dt * (F_p_n + F_p_{n+1})

    Parameters
    ----------
    K_uu, C, M_p, K_p : ndarray
        System matrices.
    F_u, F_p : ndarray
        Force vectors at time n+1.
    u_n, p_n : ndarray
        Current state.
    dt : float
        Time step.
    solver_type : str
        "dense" or "sparse".

    Returns
    -------
    u_new, p_new : ndarray
        Updated state.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    n_u = K_uu.shape[0]
    n_p = M_p.shape[0]

    # Step 1: displacement from equilibrium
    rhs_u = F_u - C @ p_n
    try:
        u_new = np.linalg.solve(K_uu, rhs_u)
    except (np.linalg.LinAlgError, ValueError):
        # Regularize if singular
        reg = 1e-8 * np.eye(n_u)
        u_new = np.linalg.lstsq(K_uu + reg, rhs_u, rcond=None)[0]

    # Step 2: pressure update
    A_p = M_p + 0.5 * dt * K_p
    rhs_p = (M_p - 0.5 * dt * K_p) @ p_n - C.T @ (u_new - u_n)
    rhs_p += 0.5 * dt * F_p

    try:
        p_new = np.linalg.solve(A_p, rhs_p)
    except (np.linalg.LinAlgError, ValueError):
        reg = 1e-8 * np.eye(n_p)
        p_new = np.linalg.lstsq(A_p + reg, rhs_p, rcond=None)[0]

    return u_new, p_new


def imex_splitting_step(K_uu, C, M_p, K_p, F_u, F_p,
                        u_n, p_n, dt, explicit_ratio=0.5):
    """
    IMEX splitting for the consolidation equations.

    Explicit part: fluid pressure diffusion treated with forward Euler
    Implicit part: solid equilibrium and coupling treated with backward Euler

    This mimics the operator splitting in fd1d_predator_prey.

    Parameters
    ----------
    explicit_ratio : float
        Fraction of explicit treatment (0 to 1).
    """
    if not (0.0 <= explicit_ratio <= 1.0):
        raise ValueError("explicit_ratio must be in [0, 1].")

    n_u = K_uu.shape[0]
    n_p = M_p.shape[0]

    # Explicit predictor for pressure diffusion
    if explicit_ratio > 1e-14:
        try:
            p_star = p_n - explicit_ratio * dt * (
                np.linalg.solve(M_p, K_p @ p_n - F_p)
            )
        except np.linalg.LinAlgError:
            p_star = p_n.copy()
    else:
        p_star = p_n.copy()

    # Implicit corrector for displacement
    rhs_u = F_u - C @ p_star
    try:
        u_new = np.linalg.solve(K_uu, rhs_u)
    except np.linalg.LinAlgError:
        u_new = np.linalg.lstsq(K_uu, rhs_u, rcond=None)[0]

    # Implicit corrector for pressure
    A_p = M_p + (1.0 - explicit_ratio) * dt * K_p
    rhs_p = M_p @ p_n - C.T @ (u_new - u_n) + (1.0 - explicit_ratio) * dt * F_p
    try:
        p_new = np.linalg.solve(A_p, rhs_p)
    except np.linalg.LinAlgError:
        p_new = np.linalg.lstsq(A_p, rhs_p, rcond=None)[0]

    return u_new, p_new


def exponential_integrator_exact(alpha, t0, y0, tstop, n_steps):
    """
    Exact solution of y' = alpha * y for verification.

    y(t) = y0 * exp(alpha * (t - t0))

    Returns time array and solution array.
    """
    t = np.linspace(t0, tstop, n_steps + 1)
    y = y0 * np.exp(alpha * (t - t0))
    return t, y


def dynamic_time_stepping(M_uu, C, M_p, K_uu, K_p, F_u_func, F_p_func,
                          u0, v0, p0, tspan, n_steps):
    """
    Newmark-beta time integration for dynamic poroelasticity.

    Uses the Newmark scheme with beta=0.25, gamma=0.5 (average acceleration,
    unconditionally stable).

    M_uu * ddot(u) + K_uu * u + C * p = F_u(t)
    C^T * dot(u) + M_p * dot(p) + K_p * p = F_p(t)

    This is simplified to a sequential update.
    """
    dt = (tspan[1] - tspan[0]) / n_steps
    beta = 0.25
    gamma = 0.5

    n_u = M_uu.shape[0]
    n_p = M_p.shape[0]

    u = u0.copy()
    v = v0.copy()
    a = np.zeros_like(u)
    p = p0.copy()

    # Initial acceleration
    rhs = F_u_func(tspan[0]) - K_uu @ u - C @ p
    try:
        a = np.linalg.solve(M_uu, rhs)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(M_uu, rhs, rcond=None)[0]

    u_hist = np.zeros((n_steps + 1, n_u))
    p_hist = np.zeros((n_steps + 1, n_p))
    u_hist[0, :] = u
    p_hist[0, :] = p

    # Effective stiffness
    K_eff = K_uu + (1.0 / (beta * dt ** 2)) * M_uu

    for n in range(n_steps):
        t = tspan[0] + n * dt
        t_next = t + dt

        # Predictor
        u_pred = u + dt * v + 0.5 * dt ** 2 * (1.0 - 2.0 * beta) * a
        v_pred = v + dt * (1.0 - gamma) * a

        # Solve for displacement at n+1
        F_u = F_u_func(t_next)
        F_p = F_p_func(t_next)

        # Simplified: first update p, then u
        # Pressure equation (semi-implicit)
        A_p = M_p + dt * K_p
        rhs_p = M_p @ p - C.T @ (v_pred + dt * a) + dt * F_p
        try:
            p_new = np.linalg.solve(A_p, rhs_p)
        except np.linalg.LinAlgError:
            p_new = np.linalg.lstsq(A_p, rhs_p, rcond=None)[0]

        # Displacement equation
        rhs_u = F_u - C @ p_new + (1.0 / (beta * dt ** 2)) * M_uu @ u_pred
        try:
            u_new = np.linalg.solve(K_eff, rhs_u)
        except np.linalg.LinAlgError:
            u_new = np.linalg.lstsq(K_eff, rhs_u, rcond=None)[0]

        # Corrector
        a_new = (u_new - u_pred) / (beta * dt ** 2)
        v_new = v_pred + gamma * dt * a_new

        u = u_new
        v = v_new
        a = a_new
        p = p_new

        u_hist[n + 1, :] = u
        p_hist[n + 1, :] = p

    return u_hist, p_hist


def compute_cfl_condition(Vmax, hmin, safety_factor=0.5):
    """
    Compute CFL time step limit for explicit wave propagation.

    dt_max = safety_factor * h_min / V_max
    """
    if Vmax <= 0.0 or hmin <= 0.0:
        raise ValueError("Vmax and hmin must be positive.")
    return safety_factor * hmin / Vmax
