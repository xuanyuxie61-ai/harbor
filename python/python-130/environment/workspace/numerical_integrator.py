# -*- coding: utf-8 -*-
"""
================================================================================
Numerical Integration Utilities for Synaptic Dynamics
================================================================================

This module provides fundamental numerical methods for integrating ODEs
that arise in synaptic plasticity models, including:

1. Explicit Runge-Kutta methods (RK1/Euler, RK4)
2. Stability analysis via eigenvalue decomposition
3. Error estimation and adaptive stepping

Mathematical Background:
------------------------
For a general ODE system:

    dy/dt = f(t, y),    y(t₀) = y₀

The explicit Euler (RK1) method is:

    y_{n+1} = y_n + h · f(t_n, y_n)

with local truncation error O(h²) and global error O(h).

The classical RK4 method is:

    k₁ = f(t_n, y_n)
    k₂ = f(t_n + h/2, y_n + h·k₁/2)
    k₃ = f(t_n + h/2, y_n + h·k₂/2)
    k₄ = f(t_n + h, y_n + h·k₃)
    y_{n+1} = y_n + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)

with local truncation error O(h⁵) and global error O(h⁴).

Absolute Stability:
-------------------
For the test equation dy/dt = λy, the RK1 stability region is |1 + z| ≤ 1
where z = hλ. This requires h ≤ 2/|λ| for stability.

For synaptic weight dynamics with multiple time scales, the system
may be stiff. The stiffness ratio:

    S = |Re(λ_max)| / |Re(λ_min)|

determines whether explicit methods are practical.

================================================================================
"""

import numpy as np
from typing import Callable, Tuple, Optional


def rk1_integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE using the explicit Euler (RK1) method.

    Parameters
    ----------
    f : callable
        Right-hand side function f(t, y) -> dy/dt.
    tspan : tuple
        (t0, t_stop) integration interval.
    y0 : np.ndarray
        Initial condition vector.
    n_steps : int
        Number of time steps. Must be >= 1.

    Returns
    -------
    t : np.ndarray
        Time points, shape (n_steps+1,).
    y : np.ndarray
        Solution values, shape (n_steps+1, m).

    Raises
    ------
    ValueError
        If n_steps < 1 or tspan is invalid.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps={n_steps} must be >= 1.")
    t0, t_stop = tspan
    if t_stop <= t0:
        raise ValueError(f"t_stop={t_stop} must be > t0={t0}.")

    dt = (t_stop - t0) / n_steps
    m = y0.shape[0]

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    t[0] = t0
    y[0, :] = y0.flatten()

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        dydt = f(t[i], y[i, :])
        if dydt is None or not np.isfinite(dydt).all():
            raise RuntimeError(f"ODE rhs returned invalid values at t={t[i]}")
        y[i + 1, :] = y[i, :] + dt * dydt

    return t, y


def rk4_integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE using the classical Runge-Kutta 4 method.

    Parameters
    ----------
    f : callable
        Right-hand side function f(t, y).
    tspan : tuple
        (t0, t_stop) integration interval.
    y0 : np.ndarray
        Initial condition.
    n_steps : int
        Number of steps. Must be >= 1.

    Returns
    -------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Solution values.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    t0, t_stop = tspan
    if t_stop <= t0:
        raise ValueError("t_stop must be > t0.")

    dt = (t_stop - t0) / n_steps
    m = y0.shape[0]

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    t[0] = t0
    y[0, :] = y0.flatten()

    for i in range(n_steps):
        # TODO: Implement one step of the classical 4th-order Runge-Kutta (RK4) method.
        # Given the current state yi at time ti, compute:
        #   k1 = f(ti, yi)
        #   k2 = f(ti + dt/2, yi + dt*k1/2)
        #   k3 = f(ti + dt/2, yi + dt*k2/2)
        #   k4 = f(ti + dt, yi + dt*k3)
        # Then update:
        #   y[i+1, :] = yi + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        #   t[i+1] = ti + dt
        pass

    return t, y


def estimate_stability_jacobian(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Numerically estimate the Jacobian J_ij = ∂f_i/∂y_j via central differences.

    The Jacobian eigenvalues determine local stability:

        dy/dt = J · y  =>  y(t) = exp(Jt) · y(0)

    Parameters
    ----------
    f : callable
        RHS function.
    t : float
        Current time.
    y : np.ndarray
        State vector.
    eps : float
        Perturbation size.

    Returns
    -------
    J : np.ndarray
        Jacobian matrix.
    """
    m = y.shape[0]
    J = np.zeros((m, m))
    f0 = f(t, y)

    for j in range(m):
        y_plus = y.copy()
        y_plus[j] += eps
        f_plus = f(t, y_plus)

        y_minus = y.copy()
        y_minus[j] -= eps
        f_minus = f(t, y_minus)

        J[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return J


def compute_stiffness_ratio(
    J: np.ndarray,
) -> Tuple[float, complex, complex]:
    """
    Compute the stiffness ratio S = |Re(λ_max)| / |Re(λ_min)|
    from the Jacobian eigenvalues.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix.

    Returns
    -------
    S : float
        Stiffness ratio.
    lambda_max : complex
        Eigenvalue with largest real part magnitude.
    lambda_min : complex
        Eigenvalue with smallest real part magnitude.
    """
    eigvals = np.linalg.eigvals(J)
    re_parts = np.real(eigvals)
    abs_re = np.abs(re_parts)

    idx_max = np.argmax(abs_re)
    idx_min = np.argmin(abs_re + 1e-20)  # avoid div by zero

    S = abs_re[idx_max] / (abs_re[idx_min] + 1e-20)
    return S, eigvals[idx_max], eigvals[idx_min]


def adaptive_rk12(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    tol: float = 1e-6,
    h0: float = 0.01,
    h_min: float = 1e-10,
    h_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive integration using RK1 (Euler) and RK2 (Heun) embedded pair.

    Local error estimate:  e = |y_{RK2} - y_{RK1}|

    Step size control:  h_new = h · min(5, max(0.2, 0.9·sqrt(tol/e)))

    Parameters
    ----------
    f : callable
        RHS function.
    tspan : tuple
        Integration interval.
    y0 : np.ndarray
        Initial condition.
    tol : float
        Error tolerance.
    h0 : float
        Initial step size.
    h_min : float
        Minimum step size.
    h_max : float
        Maximum step size.

    Returns
    -------
    t : np.ndarray
        Time points.
    y : np.ndarray
        Solution values.
    h_history : np.ndarray
        Step size history.
    """
    t0, t_stop = tspan
    if t_stop <= t0:
        raise ValueError("t_stop must be > t0.")
    if tol <= 0.0 or h0 <= 0.0:
        raise ValueError("tol and h0 must be positive.")

    t_list = [t0]
    y_list = [y0.copy()]
    h_list = []

    t_curr = t0
    y_curr = y0.copy()
    h = h0

    max_steps = 100000
    step_count = 0

    while t_curr < t_stop and step_count < max_steps:
        h = min(h, h_max)
        h = max(h, h_min)
        if t_curr + h > t_stop:
            h = t_stop - t_curr

        # RK1
        k1 = f(t_curr, y_curr)
        y_rk1 = y_curr + h * k1

        # RK2 (Heun)
        k2 = f(t_curr + h, y_rk1)
        y_rk2 = y_curr + 0.5 * h * (k1 + k2)

        # Error estimate
        e = np.linalg.norm(y_rk2 - y_rk1)

        if e <= tol or h <= h_min:
            # Accept step
            t_curr = t_curr + h
            y_curr = y_rk2.copy()
            t_list.append(t_curr)
            y_list.append(y_curr.copy())
            h_list.append(h)
            step_count += 1

            # Increase step size
            if e > 1e-20:
                h = h * min(5.0, max(0.2, 0.9 * np.sqrt(tol / e)))
            else:
                h = h * 2.0
        else:
            # Reject and reduce
            h = h * max(0.2, 0.9 * np.sqrt(tol / e))
            if h < h_min:
                h = h_min
                # Accept with warning
                t_curr = t_curr + h
                y_curr = y_rk2.copy()
                t_list.append(t_curr)
                y_list.append(y_curr.copy())
                h_list.append(h)
                step_count += 1

    t_arr = np.array(t_list)
    y_arr = np.array(y_list)
    h_arr = np.array(h_list)
    return t_arr, y_arr, h_arr


if __name__ == "__main__":
    # Test with simple harmonic oscillator
    def harmonic(t, y):
        return np.array([y[1], -y[0]])

    t, y = rk1_integrate(harmonic, (0.0, 10.0), np.array([1.0, 0.0]), 1000)
    print(f"RK1 final state: {y[-1]}")
