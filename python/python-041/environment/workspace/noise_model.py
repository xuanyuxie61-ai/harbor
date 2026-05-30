
import numpy as np


def ornstein_uhlenbeck_euler(theta, mu, sigma, x0, tmax, n, rng=None):
    if theta <= 0:
        raise ValueError("theta must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")
    if rng is None:
        rng = np.random.default_rng()
    dt = tmax / n
    t = np.linspace(0.0, tmax, n + 1)
    x = np.zeros(n + 1)
    x[0] = x0

    dw = np.sqrt(dt) * rng.standard_normal(n)
    for j in range(n):
        x[j + 1] = x[j] + dt * theta * (mu - x[j]) + sigma * dw[j]
    return t, x


def ornstein_uhlenbeck_euler_maruyama(theta, mu, sigma, x0, tmax, n, r, rng=None):
    if theta <= 0 or sigma < 0 or n <= 0 or r <= 0:
        raise ValueError("Invalid parameters")
    if rng is None:
        rng = np.random.default_rng()
    dt_large = tmax / n
    dt_small = dt_large / r
    t = np.linspace(0.0, tmax, n + 1)
    x = np.zeros(n + 1)
    x[0] = x0
    for j in range(n):
        dw = np.sqrt(dt_small) * rng.standard_normal(r)
        x[j + 1] = x[j] + dt_large * theta * (mu - x[j]) + sigma * np.sum(dw)
    return t, x


def generate_seismic_noise(n_traces, n_samples, dt, theta=5.0, sigma=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    tmax = n_samples * dt
    noise = np.zeros((n_traces, n_samples))
    for i in range(n_traces):

        th = theta * (1.0 + 0.1 * rng.standard_normal())
        sg = sigma * (1.0 + 0.1 * rng.standard_normal())
        _, x = ornstein_uhlenbeck_euler(th, 0.0, sg, 0.0, tmax, n_samples, rng=rng)

        if len(x) > n_samples:
            x = x[:n_samples]
        elif len(x) < n_samples:

            x = np.concatenate([x, np.full(n_samples - len(x), x[-1])])
        noise[i, :] = x
    return noise


def generate_random_velocity_perturbation(nx, ny, theta=2.0, sigma=0.1, dx=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    std = sigma / np.sqrt(2.0 * theta)
    perturbation = rng.normal(0.0, std, size=(ny, nx))

    for _ in range(3):
        smoothed = perturbation.copy()
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                smoothed[j, i] = 0.25 * (
                    perturbation[j - 1, i] + perturbation[j + 1, i] +
                    perturbation[j, i - 1] + perturbation[j, i + 1]
                )
        perturbation = 0.7 * perturbation + 0.3 * smoothed
    return perturbation
