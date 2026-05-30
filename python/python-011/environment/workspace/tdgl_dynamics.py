# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp


def solve_wave_equation_mol(nx=64, c=1.0, t_span=(0.0, 2.0 * np.pi), nt_eval=200):
    if nx < 3:
        raise ValueError("nx 必须 >= 3。")
    L = 2.0 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx, endpoint=False)


    u0 = np.sin(x)
    v0 = np.zeros_like(x)
    w0 = np.concatenate([u0, v0])

    def deriv(t, w):
        u = w[:nx]
        v = w[nx:]

        u_xx = np.zeros_like(u)
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            u_xx[i] = (u[im] - 2.0 * u[i] + u[ip]) / (dx ** 2)
        du = v
        dv = c * u_xx
        return np.concatenate([du, dv])

    t_eval = np.linspace(t_span[0], t_span[1], nt_eval)
    sol = solve_ivp(deriv, t_span, w0, t_eval=t_eval, method='RK45')

    t = sol.t
    nt = t.size
    u = sol.y[:nx, :].T
    v = sol.y[nx:, :].T


    energy = np.zeros(nt)
    for it in range(nt):
        ux = np.zeros(nx)
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            ux[i] = (u[it, ip] - u[it, im]) / (2.0 * dx)
        H_u = 0.5 * c ** 2 * np.sum(ux ** 2) * dx
        H_v = 0.5 * np.sum(v[it, :] ** 2) * dx
        energy[it] = H_u + H_v

    return t, u, v, energy


def chen_system_rhs(t, xyz, a=40.0, b=3.0, c=28.0):
    x, y, z = xyz
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return np.array([dx, dy, dz])


def solve_chen_system(t_span=(0.0, 15.0), y0=None, params=None, nt_eval=2000):
    if y0 is None:
        y0 = np.array([-0.1, 0.5, -0.6])
    if params is None:
        params = {'a': 40.0, 'b': 3.0, 'c': 28.0}
    a, b, c = params['a'], params['b'], params['c']
    t_eval = np.linspace(t_span[0], t_span[1], nt_eval)

    sol = solve_ivp(
        lambda t, y: chen_system_rhs(t, y, a, b, c),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12
    )
    t = sol.t
    y = sol.y.T


    eps0 = 1e-10
    y0_pert = y0 + np.array([eps0, 0.0, 0.0])
    sol2 = solve_ivp(
        lambda t, y: chen_system_rhs(t, y, a, b, c),
        t_span, y0_pert, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12
    )
    y2 = sol2.y.T
    delta = np.linalg.norm(y2 - y, axis=1)

    mask = delta > 1e-15
    if np.sum(mask) > 10:
        logd = np.log(delta[mask])
        tt = t[mask]

        lam = np.mean(np.diff(logd) / np.diff(tt))
    else:
        lam = 0.0

    return t, y, lam


def tdgl_evolution_1d(Nx=128, L=10.0, T=1.0, Tc=1.5, tau=1.0, xi=1.0,
                      b_coeff=-1.0, t_span=(0.0, 10.0), nt=500):
    if Nx < 3:
        raise ValueError("Nx 必须 >= 3。")
    dx = L / Nx
    alpha = 1.0
    a_T = alpha * (T - Tc)

    x = np.linspace(0, L, Nx, endpoint=False)

    Delta0 = 0.1 * np.random.randn(Nx)

    def deriv(t, D):
        D_xx = np.zeros_like(D)
        for i in range(Nx):
            im = (i - 1) % Nx
            ip = (i + 1) % Nx
            D_xx[i] = (D[im] - 2.0 * D[i] + D[ip]) / (dx ** 2)



        dD = (-a_T * D - abs(b_coeff) * D ** 3 + xi ** 2 * D_xx) / tau
        return dD

    t_eval = np.linspace(t_span[0], t_span[1], nt)
    sol = solve_ivp(deriv, t_span, Delta0, t_eval=t_eval, method='RK45')
    t = sol.t
    Delta = sol.y.T
    max_amp = np.max(np.abs(Delta), axis=1)
    return t, Delta, max_amp


def compute_superfluid_stiffness(order_parameter_history, dx, dt):
    Delta = np.asarray(order_parameter_history, dtype=float)
    if Delta.size == 0:
        return 0.0
    amp2 = np.mean(Delta ** 2, axis=1)

    n = amp2.size
    start = int(0.8 * n)
    if start >= n:
        start = n - 1
    return np.mean(amp2[start:])
