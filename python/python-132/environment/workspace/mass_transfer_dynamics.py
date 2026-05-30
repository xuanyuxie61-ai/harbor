
import numpy as np
from utils import ensure_positive, clip_with_warning






def rk45_integrate(yprime, tspan, y0, n_steps, projection=None):
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    t = np.zeros(n_steps + 1, dtype=float)
    y = np.zeros((n_steps + 1, m), dtype=float)
    e = np.zeros((n_steps + 1, m), dtype=float)

    dt = (tspan[1] - tspan[0]) / n_steps
    t[0] = tspan[0]
    y[0, :] = y0
    if projection is not None:
        y[0, :] = projection(y[0, :])
    e[0, :] = 0.0

    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0]
    ], dtype=float)
    b = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0], dtype=float)
    c = np.array([0.0, 0.25, 3.0/8.0, 12.0/13.0, 1.0, 0.5], dtype=float)
    d = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0], dtype=float)

    for i in range(n_steps):
        k1 = dt * yprime(t[i] + c[0] * dt, y[i, :])
        k2 = dt * yprime(t[i] + c[1] * dt, y[i, :] + a[1, 0] * k1)
        k3 = dt * yprime(t[i] + c[2] * dt, y[i, :] + a[2, 0] * k1 + a[2, 1] * k2)
        k4 = dt * yprime(t[i] + c[3] * dt, y[i, :] + a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3)
        k5 = dt * yprime(t[i] + c[4] * dt, y[i, :] + a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4)
        k6 = dt * yprime(t[i] + c[5] * dt, y[i, :] + a[5, 0] * k1 + a[5, 1] * k2 + a[5, 2] * k3 + a[5, 3] * k4 + a[5, 4] * k5)

        y4 = y[i, :] + d[0] * k1 + d[1] * k2 + d[2] * k3 + d[3] * k4 + d[4] * k5 + d[5] * k6
        y5 = y[i, :] + b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6

        t[i + 1] = t[i] + dt
        y[i + 1, :] = y5
        if projection is not None:
            y[i + 1, :] = projection(y[i + 1, :])
        e[i + 1, :] = np.abs(y5 - y4)

    return t, y, e






def maxwell_stefan_diffusion(y, D_matrix, c_total):
    y = np.asarray(y, dtype=float)
    x = y[0:3]
    N = y[3:6]


    x = np.clip(x, 1e-6, 1.0)
    xsum = np.sum(x)
    if xsum > 1e-12:
        x = x / xsum
    else:
        x = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])


    dxdt = N / max(c_total, 1.0)
    dxdt = np.clip(dxdt, -0.1, 0.1)


    dNdt = np.zeros(3, dtype=float)
    for i in range(3):
        force = 0.0
        for j in range(3):
            if i != j:
                Dij = max(D_matrix[i, j], 1e-15)
                diff = x[j] - x[i]

                dist = np.abs(diff) + 1e-3
                force += diff / (dist * Dij)
        dNdt[i] = np.clip(force * c_total * 1e-4, -1.0, 1.0)

    return np.concatenate([dxdt, dNdt])


def simulate_three_component_diffusion(y0, D_matrix, c_total, tspan, n_steps):
    def deriv(t, y):
        return maxwell_stefan_diffusion(y, D_matrix, c_total)
    return rk45_integrate(deriv, tspan, y0, n_steps)






def langford_deriv(t, xyz, a=3.0, b=1.5, c=1.0, d=0.5, e=0.2, f=0.1):
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[0], xyz[1], xyz[2]

    dxdt = (z - b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - z ** 3 / 3.0 - (x ** 2 + y ** 2) * (1.0 + e * z) + f * z * x ** 3

    return np.array([dxdt, dydt, dzdt], dtype=float)


def simulate_langford_mixing(xyz0, tspan, n_steps, a=3.0, b=1.5, c=1.0, d=0.5, e=0.2, f=0.1):
    def deriv(t, y):
        return langford_deriv(t, y, a, b, c, d, e, f)
    return rk45_integrate(deriv, tspan, xyz0, n_steps)






def lorenz96_deriv(t, y, n=40, force=8.0):
    y = np.asarray(y, dtype=float)
    if len(y) != n:
        n = len(y)

    dydt = np.zeros(n, dtype=float)
    for i in range(n):
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        ip1 = (i + 1) % n
        dydt[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + force

    return dydt


def simulate_lorenz96_convection(y0, tspan, n_steps, force=8.0):
    n = len(y0)

    def deriv(t, y):
        return lorenz96_deriv(t, y, n, force)

    return rk45_integrate(deriv, tspan, y0, n_steps)






def distillation_column_deriv(t, state, n_trays, nc, F, z_feed, q_feed,
                               L, V, holdup, alpha_rel, tray_eff):
    state = np.asarray(state, dtype=float)
    dstate = np.zeros_like(state)

    for j in range(n_trays):
        x_j = state[j * nc:(j + 1) * nc].copy()
        x_j = np.clip(x_j, 1e-12, 1.0)
        x_j = x_j / np.sum(x_j)












        K_j = None
        y_eq = None
        y_j = None
        y_jp1 = None



        L_jm1 = float(L[j - 1]) if j > 0 else 0.0
        x_jm1 = state[(j - 1) * nc:j * nc].copy() if j > 0 else x_j.copy()
        if j > 0:
            x_jm1 = np.clip(x_jm1, 1e-12, 1.0)
            x_jm1 = x_jm1 / np.sum(x_jm1)

        V_jp1 = float(V[j + 1]) if j < n_trays - 1 else 0.0
        y_jp1_in = y_jp1.copy()

        F_j = float(F[j])
        z_j = z_feed[j, :].copy()
        z_j = np.clip(z_j, 1e-12, 1.0)
        z_j = z_j / (np.sum(z_j) + 1e-15)


        qf = float(np.clip(q_feed[j], 0.0, 1.0))
        L_feed = qf * F_j
        V_feed = (1.0 - qf) * F_j

        M_j = max(float(holdup[j]), 1e-6)

        for i in range(nc):
            dxdt = (
                L_jm1 * x_jm1[i] + V_jp1 * y_jp1_in[i]
                + L_feed * z_j[i] + V_feed * z_j[i]
                - float(L[j]) * x_j[i] - float(V[j]) * y_j[i]
            ) / M_j

            if x_j[i] <= 1e-8 and dxdt < 0:
                dxdt = 0.0
            if x_j[i] >= 1.0 - 1e-8 and dxdt > 0:
                dxdt = 0.0
            dstate[j * nc + i] = dxdt

    return dstate


def simulate_distillation_dynamics(n_trays, nc, F, z_feed, q_feed, L, V,
                                    holdup, alpha_rel, tray_eff,
                                    x0, tspan, n_steps):
    def deriv(t, y):
        return distillation_column_deriv(
            t, y, n_trays, nc, F, z_feed, q_feed,
            L, V, holdup, alpha_rel, tray_eff
        )

    def projection(y):
        y = np.asarray(y, dtype=float)
        for j in range(n_trays):
            xj = y[j * nc:(j + 1) * nc]
            xj = np.clip(xj, 1e-12, 1.0)
            s = np.sum(xj)
            if s > 1e-12:
                y[j * nc:(j + 1) * nc] = xj / s
        return y

    t, y, e = rk45_integrate(deriv, tspan, x0, n_steps, projection=projection)

    composition_profiles = y.reshape((n_steps + 1, n_trays, nc))
    return t, y, e, composition_profiles
