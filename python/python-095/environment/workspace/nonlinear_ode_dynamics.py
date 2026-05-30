
import numpy as np
import math


def anishchenko_adaptive_deriv(t, state, mu=1.2, eta=0.5, gamma_leak=0.01):
    w1, w2, e = state
    dw1 = mu * w1 + w2 - w1 * e - gamma_leak * w1
    dw2 = -w1 - gamma_leak * w2
    de = -eta * e + eta * (1.0 if w1 >= 0.0 else 0.0) * (w1 ** 2)
    return np.array([dw1, dw2, de], dtype=float)


def rk4_integrate(f, t0, y0, tstop, h=0.01):
    y = np.asarray(y0, dtype=float)
    t = t0
    trajectory = [(t, y.copy())]

    while t < tstop:
        if t + h > tstop:
            h = tstop - t
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2.0, y + k1 / 2.0)
        k3 = h * f(t + h / 2.0, y + k2 / 2.0)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t = t + h
        trajectory.append((t, y.copy()))

    return trajectory


def adaptive_lms_ode(t, W, R, P, mu, gamma_reg=0.0):
    W = np.asarray(W, dtype=float)
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)
    dW = mu * (P - R @ W) - gamma_reg * W
    return dW


def stability_boundary_anishchenko(mu_range, gamma_range, n_grid=50):
    mu_grid = np.linspace(mu_range[0], mu_range[1], n_grid)
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_grid)
    stable_mask = np.zeros((n_grid, n_grid), dtype=bool)

    eta = 0.5
    for i, mu in enumerate(mu_grid):
        for j, gamma in enumerate(gamma_grid):
            J = np.array([[mu - gamma, 1.0, 0.0],
                          [-1.0, -gamma, 0.0],
                          [0.0, 0.0, -eta]])
            eigs = np.linalg.eigvals(J)
            stable_mask[i, j] = np.all(np.real(eigs) < 0)

    return mu_grid, gamma_grid, stable_mask
