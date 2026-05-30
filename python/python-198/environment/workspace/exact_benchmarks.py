
import numpy as np


def laplace_radial_2d_exact(x, y, a, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x ** 2 + y ** 2
    r2 = np.clip(r2, 1e-15, None)
    r = np.sqrt(r2)
    
    u = a * np.log(r) + b
    ux = a * x / r2
    uy = a * y / r2
    uxx = a * (r2 - 2 * x ** 2) / (r2 ** 2)
    uxy = -2 * a * x * y / (r2 ** 2)
    uyy = a * (r2 - 2 * y ** 2) / (r2 ** 2)
    
    return u, ux, uy, uxx, uxy, uyy


def laplace_radial_3d_exact(x, y, z, a, b):
    r2 = x ** 2 + y ** 2 + z ** 2
    r2 = np.clip(r2, 1e-15, None)
    r = np.sqrt(r2)
    
    u = a / r + b
    ux = -a * x / (r2 ** 1.5)
    uy = -a * y / (r2 ** 1.5)
    uz = -a * z / (r2 ** 1.5)
    return u, ux, uy, uz


def sawtooth_wave(t, omega=2.0 * np.pi, amplitude=1.0):
    phase = omega * t / (2.0 * np.pi)
    frac = phase - np.floor(phase)
    return amplitude * (2.0 * frac - 1.0)


def sawtooth_ode_rhs(t, y, omega=2.0 * np.pi):
    u, v = y[0], y[1]
    dudt = v
    dvdt = -u + sawtooth_wave(t, omega)
    return np.array([dudt, dvdt])


def solve_sawtooth_ode(t_span, y0, omega=2.0 * np.pi, n_steps=1000):
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, 2))
    y[0] = y0
    
    for i in range(n_steps):
        k1 = sawtooth_ode_rhs(t[i], y[i], omega)
        k2 = sawtooth_ode_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k1, omega)
        k3 = sawtooth_ode_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k2, omega)
        k4 = sawtooth_ode_rhs(t[i] + dt, y[i] + dt * k3, omega)
        y[i + 1] = y[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return t, y


def compute_l2_error(u_num, u_exact, area):
    diff = u_num - u_exact
    err_sq = np.sum(diff ** 2 * area)
    return np.sqrt(err_sq)
