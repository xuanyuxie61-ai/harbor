"""
ode_evolution.py  --  Time evolution via explicit ODE methods
=============================================================
Thin wrapper re-exporting the midpoint method used for time-dependent
problems (damped collective oscillations, decay cascades).
"""
import numpy as np


def euler_step(rhs, y, dt):
    """Forward Euler step (827_ode_euler_system seed)."""
    return y + dt * np.asarray(rhs(y))


def midpoint_step(rhs, y, dt):
    """Explicit midpoint step (829_ode_midpoint_system seed)."""
    k1 = np.asarray(rhs(y))
    k2 = np.asarray(rhs(y + 0.5 * dt * k1))
    return y + dt * k2


def integrate_ode(rhs, y0, t_span, n_steps, method="midpoint"):
    """Integrate dy/dt = rhs(y) from t_span[0] to t_span[1]."""
    t = np.linspace(t_span[0], t_span[1], n_steps)
    dt = t[1] - t[0]
    y = np.asarray(y0, dtype=np.float64).copy()
    history = np.zeros((n_steps, len(y)))
    history[0, :] = y
    step_fn = midpoint_step if method == "midpoint" else euler_step
    for i in range(1, n_steps):
        y = step_fn(rhs, y, dt)
        history[i, :] = y
    return t, history
