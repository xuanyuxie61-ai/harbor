import numpy as np


def hexagon01_area():
    return 3.0 * np.sqrt(3.0) / 2.0


def hexagon01_sample(n):

    samples = []
    while len(samples) < n:
        n_batch = max(n - len(samples), 100)
        x = np.random.uniform(-1.0, 1.0, n_batch)
        y = np.random.uniform(-np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0, n_batch)

        mask = (np.abs(x) <= 1.0) & (np.abs(y) <= np.sqrt(3.0) / 2.0) & \
               (np.abs(x) + np.abs(y) / np.sqrt(3.0) <= 1.0)
        samples.extend(list(zip(x[mask], y[mask])))
    samples = np.array(samples[:n])
    return samples[:, 0], samples[:, 1]


def hexagon01_monte_carlo(n, func):
    area = hexagon01_area()
    x, y = hexagon01_sample(n)
    vals = func(x, y)
    return (area / n) * np.sum(vals)


def estimate_lateral_flux(n_samples, diffusivity, concentration_gradient):
    def integrand(x, y):
        return -diffusivity * concentration_gradient(x, y)
    flux = hexagon01_monte_carlo(n_samples, integrand)
    return flux
