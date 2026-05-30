
import numpy as np
from utils import NumericalConfig


def allen_cahn_potential(u):
    return 0.25 * (u ** 2 - 1.0) ** 2


def allen_cahn_derivative(u, xi):
    return u * (u ** 2 - 1.0) / (xi ** 2)


def laplacian_1d(u, dx):
    n = len(u)
    uxx = np.zeros(n, dtype=float)

    for i in range(1, n - 1):
        uxx[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx ** 2)


    if n > 1:
        uxx[0] = uxx[1]
        uxx[-1] = uxx[-2]

    return uxx


def allen_cahn_rhs(t, u, dx, nu, xi, forcing_func=None):
    uxx = laplacian_1d(u, dx)
    dudt = nu * uxx - u * (u ** 2 - 1.0) / (2.0 * xi ** 2)

    if forcing_func is not None:
        dudt += forcing_func(t, u)


    if len(dudt) > 1:
        dudt[0] = dudt[1]
        dudt[-1] = dudt[-2]

    return dudt


def rk4_step(y, t, dt, rhs_func):
    k1 = dt * rhs_func(t, y)
    k2 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * rhs_func(t + dt, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_regime_shift(x_min, x_max, nx, u_initial, nu, xi,
                          T_total, dt, forcing_func=None,
                          save_interval=10):
    dx = (x_max - x_min) / (nx - 1)
    x = np.linspace(x_min, x_max, nx)

    if u_initial is None:

        u_initial = np.tanh((x - 0.5 * (x_min + x_max)) / (np.sqrt(2.0) * xi))

    u = u_initial.copy()
    n_steps = int(T_total / dt)

    t_history = [0.0]
    u_history = [u.copy()]

    def rhs(t, y):
        return allen_cahn_rhs(t, y, dx, nu, xi, forcing_func)

    for step in range(n_steps):
        t = step * dt
        u = rk4_step(u, t, dt, rhs)


        u = np.clip(u, -1.5, 1.5)

        if (step + 1) % save_interval == 0:
            t_history.append((step + 1) * dt)
            u_history.append(u.copy())

    return u, t_history, u_history


def fishery_forcing(t, u, E_t, epsilon, q, K):

    E_norm = min(E_t * q / 0.5, 2.0)
    forcing = epsilon * E_norm * (1.0 - u ** 2) * np.sign(u)
    return forcing


def compute_regime_shift_time(u_history, threshold=0.0):
    for i, u in enumerate(u_history):
        mean_u = np.mean(u)
        if i > 0:
            prev_mean = np.mean(u_history[i - 1])
            if prev_mean > threshold and mean_u <= threshold:
                return i
            if prev_mean < threshold and mean_u >= threshold:
                return i
    return None


def energy_functional(u, dx, nu, xi):

    grad_u = np.gradient(u, dx)
    grad_energy = 0.5 * nu * np.sum(grad_u ** 2) * dx


    pot_energy = np.sum(allen_cahn_potential(u) / (xi ** 2)) * dx

    return grad_energy + pot_energy
