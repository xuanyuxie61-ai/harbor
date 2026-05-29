"""
Time integration methods for stiff ODE systems arising in membrane mass transfer.

Adapted from:
  - backward_euler_fixed.m (fixed-point backward Euler)
  - runge_power_series.m (series expansion)
  - Kepler and quasiperiodic conserved-quantity monitoring
"""

import numpy as np
from mass_transfer_ode import (
    reaction_deriv,
    kepler_like_trajectory_deriv,
    quasiperiodic_forcing_deriv,
)


def backward_euler_fixed(f, tspan, y0, n_steps, it_max=10):
    """
    Fixed-point backward Euler integration.

    y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
    Solved via simple fixed-point iteration.
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t = np.linspace(tspan[0], tspan[1], n_steps + 1)
    dt = t[1] - t[0]
    y = np.zeros((n_steps + 1, m), dtype=float)
    y[0, :] = y0
    for i in range(n_steps):
        tp = t[i] + dt
        yp = y[i, :].copy()
        for _ in range(it_max):
            yp_new = y[i, :] + dt * f(tp, yp)
            yp = yp_new
        y[i + 1, :] = yp
    return t, y


def forward_euler(f, tspan, y0, n_steps):
    """
    Explicit forward Euler (for comparison / non-stiff regions).
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t = np.linspace(tspan[0], tspan[1], n_steps + 1)
    dt = t[1] - t[0]
    y = np.zeros((n_steps + 1, m), dtype=float)
    y[0, :] = y0
    for i in range(n_steps):
        y[i + 1, :] = y[i, :] + dt * f(t[i], y[i, :])
    return t, y


def runge_kutta4(f, tspan, y0, n_steps):
    """
    Classical fourth-order Runge-Kutta integrator.
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t = np.linspace(tspan[0], tspan[1], n_steps + 1)
    dt = t[1] - t[0]
    y = np.zeros((n_steps + 1, m), dtype=float)
    y[0, :] = y0
    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        k1 = f(ti, yi)
        k2 = f(ti + 0.5 * dt, yi + 0.5 * dt * k1)
        k3 = f(ti + 0.5 * dt, yi + 0.5 * dt * k2)
        k4 = f(ti + dt, yi + dt * k3)
        y[i + 1, :] = yi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return t, y


def adaptive_rk45(f, tspan, y0, atol=1e-8, rtol=1e-6, h_init=None):
    """
    Adaptive RK45 (Dormand-Prince) integration for stiff membrane ODEs.
    Simplified implementation for robustness.
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t0, tf = tspan
    if h_init is None:
        h = (tf - t0) * 1e-3
    else:
        h = h_init
    t_list = [t0]
    y_list = [y0.copy()]
    t = t0
    y = y0.copy()

    # Dormand-Prince coefficients (simplified)
    a2, a3, a4, a5, a6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0
    b21 = 1 / 5
    b31, b32 = 3 / 40, 9 / 40
    b41, b42, b43 = 44 / 45, -56 / 15, 32 / 9
    b51, b52, b53, b54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    b61, b62, b63, b64, b65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
    b71, b72, b73, b74, b75, b76 = 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84

    c1, c3, c4, c5, c6, c7 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0
    d1, d3, d4, d5, d6, d7 = 5179 / 57600, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40

    max_steps = 100000
    step = 0
    while t < tf and step < max_steps:
        step += 1
        if t + h > tf:
            h = tf - t

        k1 = f(t, y)
        k2 = f(t + a2 * h, y + h * (b21 * k1))
        k3 = f(t + a3 * h, y + h * (b31 * k1 + b32 * k2))
        k4 = f(t + a4 * h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
        k5 = f(t + a5 * h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6 = f(t + a6 * h, y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

        y_next = y + h * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)
        k7 = f(t + h, y_next)
        y_alt = y + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7)

        err = np.linalg.norm(y_next - y_alt) / max(np.linalg.norm(y_next), 1e-30)
        if err <= atol or err <= rtol * np.linalg.norm(y_next):
            t += h
            y = y_next
            t_list.append(t)
            y_list.append(y.copy())
            h *= min(2.0, max(0.5, 0.9 * (atol / max(err, 1e-30)) ** 0.2))
        else:
            h *= 0.5
            if abs(h) < 1e-30:
                raise RuntimeError("Adaptive RK45 step size underflow.")

    return np.array(t_list, dtype=float), np.array(y_list, dtype=float)


def conserved_quantity_kepler(y):
    """
    Compute the Hamiltonian (total energy) of the Kepler-like trajectory:
        H = 0.5 (p1^2 + p2^2) - mu / sqrt(q1^2 + q2^2)
    For validation of symplectic/energy-conserving properties.
    """
    q1, q2, p1, p2 = y[:4]
    mu = 1.0e-20
    eps = 1e-12
    r = np.sqrt(q1 ** 2 + q2 ** 2 + eps ** 2)
    H = 0.5 * (p1 ** 2 + p2 ** 2) - mu / r
    return H


def conserved_quantity_quasiperiodic(y, omega1=np.pi):
    """
    First integral of the quasiperiodic ODE:
        I = (d^2u/dt^2)^2 + (omega1^2+1) (du/dt)^2 + omega1^2 u^2 + 2 u d^2u/dt^2
    """
    u, ud, udd, uddd = y[:4]
    I = udd ** 2 + (omega1 ** 2 + 1.0) * ud ** 2 + (omega1 ** 2) * u ** 2 + 2.0 * u * udd
    return I


def compute_conservation_drift(t, y, conserved_fn):
    """
    Compute the relative drift of a conserved quantity over a trajectory.
    """
    if len(y) == 0:
        return 0.0
    vals = np.array([conserved_fn(yi) for yi in y], dtype=float)
    if abs(vals[0]) < 1e-30:
        return np.max(np.abs(vals - vals[0]))
    return np.max(np.abs(vals - vals[0])) / abs(vals[0])


def power_series_solution_ode(t, coeffs):
    """
    Evaluate a power-series solution sum c_k t^k.
    """
    t = np.asarray(t, dtype=float)
    val = np.zeros_like(t, dtype=float)
    for k, c in enumerate(coeffs):
        val += c * (t ** k)
    return val
