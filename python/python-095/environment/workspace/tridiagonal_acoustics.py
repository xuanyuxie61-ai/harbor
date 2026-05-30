
import numpy as np
import math


def tridiagonal_solver(a, b, c, d):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    n = b.size

    if d.ndim == 1:
        d = d.reshape((n, 1))


    for i in range(1, n):
        if b[i - 1] == 0.0:
            raise ValueError(f"tridiagonal_solver: zero pivot at b[{i - 1}]")
        s = a[i] / b[i - 1]
        b[i] = b[i] - s * c[i - 1]
        d[i, :] = d[i, :] - s * d[i - 1, :]

    x = np.empty_like(d, dtype=float)

    for i in range(n - 1, -1, -1):
        if abs(b[i]) < 1e-15:
            raise ValueError(f"tridiagonal_solver: near-zero pivot at b[{i}]")
        if i == n - 1:
            x[i, :] = d[i, :] / b[i]
        else:
            x[i, :] = (d[i, :] - c[i] * x[i + 1, :]) / b[i]

    if x.shape[1] == 1:
        return x[:, 0]
    return x


def tridiagonal_mv(a, b, c, x):
    x = np.asarray(x, dtype=float)
    n = x.size
    rhs = np.zeros(n, dtype=float)
    rhs[:] = b * x
    rhs[1:] = rhs[1:] + a[1:] * x[:-1]
    rhs[:-1] = rhs[:-1] + c[:-1] * x[1:]
    return rhs


def pipe_helmholtz_solver(L, N, k, source_profile, rho0=1.225, c0=343.0):
    if N <= 0:
        raise ValueError("N must be positive")
    h = L / (N + 1)
    x = np.linspace(h, L - h, N)

    omega = k * c0
    f = -1j * rho0 * omega * np.asarray(source_profile, dtype=complex)


    a = np.ones(N, dtype=float)
    b = np.full(N, (k * h) ** 2 - 2.0, dtype=float)
    c = np.ones(N, dtype=float)
    d = (h ** 2) * f




    a[0] = 0.0
    c[-1] = 0.0


    p_real = tridiagonal_solver(a.copy(), b.copy(), c.copy(), d.real)
    p_imag = tridiagonal_solver(a.copy(), b.copy(), c.copy(), d.imag)
    p = p_real + 1j * p_imag
    return x, p


def lindberg_exact_solution(t):
    t = np.asarray(t, dtype=float)
    n = t.size
    y = np.zeros((n, 4), dtype=float)
    dydt = np.zeros((n, 4), dtype=float)

    g1 = 1.0e4 * (t + 2.0 * np.exp(-t) - 2.0)
    g2 = 1.0e4 * (1.0 - np.exp(-t) - t * np.exp(-t))

    dg1dt = 1.0e4 * (1.0 - 2.0 * np.exp(-t))
    dg2dt = 1.0e4 * (t * np.exp(-t))

    eg1 = np.exp(g1)
    cg2 = np.cos(g2)
    sg2 = np.sin(g2)

    y[:, 0] = eg1 * (cg2 + sg2)
    y[:, 1] = eg1 * (cg2 - sg2)
    y[:, 2] = 1.0 - 2.0 * np.exp(-t)
    y[:, 3] = t * np.exp(-t)

    dydt[:, 0] = eg1 * dg1dt * (cg2 + sg2) + eg1 * (-sg2 + cg2) * dg2dt
    dydt[:, 1] = eg1 * dg1dt * (cg2 - sg2) + eg1 * (-sg2 - cg2) * dg2dt
    dydt[:, 2] = 2.0 * np.exp(-t)
    dydt[:, 3] = (1.0 - t) * np.exp(-t)

    return y, dydt


def lindberg_residual(t, y, dydt):
    t = np.asarray(t)
    y = np.asarray(y)
    dydt = np.asarray(dydt)
    n = t.size
    r = np.zeros((n, 4), dtype=float)
    r[:, 0] = dydt[:, 0] - (1.0e4 * y[:, 0] * y[:, 2] + 1.0e4 * y[:, 1] * y[:, 3])
    r[:, 1] = dydt[:, 1] - (-1.0e4 * y[:, 0] * y[:, 3] + 1.0e4 * y[:, 1] * y[:, 2])
    r[:, 2] = dydt[:, 2] - (1.0 - y[:, 2])
    r[:, 3] = dydt[:, 3] - (-0.5 * y[:, 2] - y[:, 3] + 0.5)
    return r
