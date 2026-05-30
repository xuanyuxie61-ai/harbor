# -*- coding: utf-8 -*-

import numpy as np


def theta_residual(F, t_old, y_old, t_new, y_new, theta, dt):
    f_old = F(t_old, y_old)
    f_new = F(t_new, y_new)
    return y_new - y_old - dt * (theta * f_old + (1.0 - theta) * f_new)


def theta_step(F, t_old, y_old, dt, theta, newton_tol=1e-10, newton_max_iter=20,
               jacobian_approx=None):






    raise NotImplementedError("Hole 3: theta_step is missing.")


def theta_method_integrate(F, t_span, y0, n_steps, theta=0.5,
                           newton_tol=1e-10, jacobian_approx=None,
                           energy_func=None):
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
    u = np.asarray(u, dtype=np.float64)
    kinetic = 0.5 * np.dot(u, u)
    if dx_weight is not None:
        kinetic *= dx_weight

    potential = -0.5 * np.dot(u, D2 @ u)
    if dx_weight is not None:
        potential *= dx_weight
    return kinetic + potential
