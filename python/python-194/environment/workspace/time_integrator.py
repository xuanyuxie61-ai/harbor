"""
time_integrator.py
==================
Adaptive time integration for transient PDEs using stabilized
semi-implicit schemes with blowup detection.

Integrates concepts from:
  * velocity_verlet (symplectic integration for second-order ODEs)
  * blowup_ode (finite-time singularity detection and adaptive stepping)

Mathematical background
-----------------------
For the transient Stokes system written as a first-order system in time:
    du/dt = F(t, u) = -A u + f(t)

We use a semi-implicit Euler scheme (backward Euler for diffusion,
forward Euler for explicit forcing):
    (I + dt * A) u^{n+1} = u^n + dt * f(t^{n+1})

For second-order systems (e.g., elastodynamics or wave propagation),
the Velocity Verlet scheme is used:
    v_{n+1/2} = v_n + 0.5 * dt * a_n
    u_{n+1}   = u_n + dt * v_{n+1/2}
    a_{n+1}   = M^{-1} (f_{n+1} - K u_{n+1})
    v_{n+1}   = v_{n+1/2} + 0.5 * dt * a_{n+1}

Blowup detection:
    Monitor the growth rate gamma_n = ||u^{n+1}|| / ||u^n||.
    If gamma_n > gamma_max for consecutive steps, reduce dt.
    If dt has been reduced below dt_min, declare blowup and stop.

Adaptive time-step control:
    dt_{n+1} = dt_n * min(2.0, max(0.5, sqrt(tol / err_est)))
    where err_est is estimated via embedded method or Richardson extrapolation.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from sparse_matrix import BandedSPDMatrix, banded_cholesky_solve


def semi_implicit_euler_step(
    u: np.ndarray,
    A: BandedSPDMatrix,
    f: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Single semi-implicit Euler step:
        (I + dt A) u_new = u + dt f
    """
    n = A.n
    M = BandedSPDMatrix(n, A.ml)
    for j in range(n):
        for i in range(j, min(n, j + A.ml + 1)):
            v = A.get(i, j)
            M.set(i, j, v * dt)
            if i == j:
                M.set(i, j, v * dt + 1.0)
    rhs = u + dt * f
    return banded_cholesky_solve(M, rhs)


def velocity_verlet_step(
    u: np.ndarray,
    v: np.ndarray,
    accel_func: Callable[[np.ndarray], np.ndarray],
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Velocity Verlet step for second-order ODE: u'' = a(u).

    Parameters
    ----------
    u           : Current position.
    v           : Current velocity.
    accel_func  : Function computing acceleration a(u).
    dt          : Time step.

    Returns
    -------
    u_new, v_new, a_new
    """
    a = accel_func(u)
    v_half = v + 0.5 * dt * a
    u_new = u + dt * v_half
    a_new = accel_func(u_new)
    v_new = v_half + 0.5 * dt * a_new
    return u_new, v_new, a_new


def adaptive_time_stepping(
    u0: np.ndarray,
    t_span: Tuple[float, float],
    dt_init: float,
    rhs_func: Callable[[float, np.ndarray], np.ndarray],
    A_band: BandedSPDMatrix,
    tol: float = 1e-4,
    gamma_max: float = 2.0,
    dt_min: float = 1e-6,
    dt_max: float = 0.1,
    max_steps: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive semi-implicit Euler with blowup detection for:
        du/dt = rhs(t, u) - A u
    Discretized as:
        (I + dt A) u_new = u + dt * rhs(t, u)

    Parameters
    ----------
    u0       : Initial condition.
    t_span   : (t_start, t_end).
    dt_init  : Initial time step.
    rhs_func : Forcing function rhs(t, u).
    A_band   : Banded SPD matrix representing diffusion operator.
    tol      : Local truncation error tolerance.
    gamma_max: Maximum allowed growth factor per step.
    dt_min   : Minimum time step before declaring blowup.
    dt_max   : Maximum time step.
    max_steps: Safety limit on total steps.

    Returns
    -------
    t_hist   : Time history.
    u_hist   : Solution history (each row is a snapshot).
    dt_hist  : Time step history.
    """
    t_start, t_end = t_span
    t = t_start
    dt = dt_init
    u = u0.copy()

    t_hist = [t]
    u_hist = [u.copy()]
    dt_hist = [dt]

    blowup_counter = 0
    max_blowup = 3

    step = 0
    while t < t_end and step < max_steps:
        step += 1
        dt = min(dt, t_end - t)
        if dt <= 0:
            break

        f = rhs_func(t, u)

        # Full step
        M = BandedSPDMatrix(A_band.n, A_band.ml)
        for j in range(A_band.n):
            for i in range(j, min(A_band.n, j + A_band.ml + 1)):
                v = A_band.get(i, j)
                M.set(i, j, v * dt)
                if i == j:
                    M.set(i, j, v * dt + 1.0)
        rhs = u + dt * f
        try:
            u_full = banded_cholesky_solve(M, rhs)
        except Exception:
            # If solve fails, reduce step and retry
            dt *= 0.5
            if dt < dt_min:
                break
            continue

        # Two half steps for error estimation (Richardson)
        dt2 = 0.5 * dt
        M2 = BandedSPDMatrix(A_band.n, A_band.ml)
        for j in range(A_band.n):
            for i in range(j, min(A_band.n, j + A_band.ml + 1)):
                v = A_band.get(i, j)
                M2.set(i, j, v * dt2)
                if i == j:
                    M2.set(i, j, v * dt2 + 1.0)

        rhs1 = u + dt2 * f
        try:
            u_half = banded_cholesky_solve(M2, rhs1)
        except Exception:
            dt *= 0.5
            if dt < dt_min:
                break
            continue

        f2 = rhs_func(t + dt2, u_half)
        rhs2 = u_half + dt2 * f2
        try:
            u_rich = banded_cholesky_solve(M2, rhs2)
        except Exception:
            dt *= 0.5
            if dt < dt_min:
                break
            continue

        # Error estimate: ||u_full - u_rich||
        err_est = float(np.linalg.norm(u_full - u_rich))
        norm_u = float(np.linalg.norm(u_full))
        rel_err = err_est / max(norm_u, 1e-15)

        # Growth factor for blowup detection
        norm_prev = float(np.linalg.norm(u))
        gamma = norm_u / max(norm_prev, 1e-15)

        if gamma > gamma_max:
            blowup_counter += 1
            if blowup_counter >= max_blowup:
                # Possible blowup: drastically reduce dt
                dt *= 0.25
                blowup_counter = 0
                if dt < dt_min:
                    # Accept but flag
                    dt = dt_min
                    t += dt
                    u = u_full
                    t_hist.append(t)
                    u_hist.append(u.copy())
                    dt_hist.append(dt)
                    break
                continue
        else:
            blowup_counter = max(0, blowup_counter - 1)

        # Accept step if error is within tolerance
        if rel_err <= tol:
            t += dt
            u = u_full
            t_hist.append(t)
            u_hist.append(u.copy())
            dt_hist.append(dt)
            # Increase step cautiously
            if rel_err > 0:
                factor = min(2.0, max(0.5, np.sqrt(tol / (rel_err + 1e-15))))
            else:
                factor = 2.0
            dt = min(dt * factor, dt_max)
        else:
            # Reject and reduce step
            dt *= 0.5
            if dt < dt_min:
                dt = dt_min
                t += dt
                u = u_full
                t_hist.append(t)
                u_hist.append(u.copy())
                dt_hist.append(dt)
                break

    return np.array(t_hist), np.array(u_hist), np.array(dt_hist)


def transient_stokes_step(
    u_n: np.ndarray,
    v_n: np.ndarray,
    p_n: np.ndarray,
    A_viscous: BandedSPDMatrix,
    B_div: np.ndarray,
    f_n: np.ndarray,
    dt: float,
    nu: float = 1.0,
    rho: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single time step for transient incompressible Stokes using
    a fractional-step projection method:

    Step 1: Predict intermediate velocity u* by solving
        (rho/dt I + nu A) u* = (rho/dt) u^n + f^n - B^T p^n

    Step 2: Solve pressure Poisson equation for phi:
        -div(B) phi = - B u* / dt   (simplified 1D model)

    Step 3: Correct:
        u^{n+1} = u* - dt/rho * B^T phi
        p^{n+1} = p^n + phi

    Parameters
    ----------
    u_n, v_n, p_n : Current velocity, velocity-Verlet auxiliary, pressure.
    A_viscous     : Viscous operator (banded SPD).
    B_div         : Divergence operator (1D array).
    f_n           : Forcing.
    dt            : Time step.
    nu            : Kinematic viscosity.
    rho           : Density.

    Returns
    -------
    u_new, v_new, p_new
    """
    # TODO: Implement the fractional-step projection method for transient
    # incompressible Stokes equations in 1D simplified form.
    #
    # Step 1: Predict intermediate velocity u* by solving
    #     (rho/dt * I + nu * A) u* = (rho/dt) * u_n + f_n - B_div * p_n
    #
    # Step 2: Solve pressure Poisson equation for phi (simplified scalar model):
    #     phi = (B_div · u*) / ||B_div||^2
    #
    # Step 3: Correct velocity and pressure:
    #     u_new = u* - (dt/rho) * B_div * phi
    #     p_new = p_n + phi
    #     v_new = (u_new - u_n) / dt
    #
    # Note: The discrete divergence operator B_div is constructed in main.py.
    # The implementation here must be consistent with how B_div is defined there.
    raise NotImplementedError("Hole 2: transient_stokes_step 需要补全分步投影法")
