#!/usr/bin/env python3

import numpy as np


def get_discrete_pdf(nx=20, ny=20):
    pdf = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) / nx
            y = (j + 0.5) / ny

            pdf[i, j] = (np.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.05) +
                         0.7 * np.exp(-((x - 0.7) ** 2 + (y - 0.5) ** 2) / 0.08))

    pdf = pdf / np.sum(pdf)
    return pdf


def set_discrete_cdf(pdf):
    cdf = np.cumsum(pdf.flatten())
    return cdf


def discrete_pdf_sample(n_samples, pdf, seed=42):
    rng = np.random.default_rng(seed)
    nx, ny = pdf.shape
    cdf = set_discrete_cdf(pdf)

    samples = np.zeros((n_samples, 2))
    for k in range(n_samples):
        u = rng.random()
        idx = np.searchsorted(cdf, u)
        idx = min(idx, nx * ny - 1)
        i = idx // ny
        j = idx % ny

        dx = rng.random() / nx
        dy = rng.random() / ny
        samples[k, 0] = (i / nx) + dx
        samples[k, 1] = (j / ny) + dy

    return samples


def lloyd_iteration(generators, pdf, n_iter=20):
    nx, ny = pdf.shape
    gens = np.asarray(generators, dtype=float).copy()
    n_gens = gens.shape[0]


    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    for _ in range(n_iter):
        new_gens = np.zeros_like(gens)
        counts = np.zeros(n_gens)


        for i in range(nx):
            for j in range(ny):
                px, py = X[i, j], Y[i, j]
                dists = np.sum((gens - [px, py]) ** 2, axis=1)
                k = np.argmin(dists)
                new_gens[k, 0] += px * pdf[i, j]
                new_gens[k, 1] += py * pdf[i, j]
                counts[k] += pdf[i, j]


        for k in range(n_gens):
            if counts[k] > 1e-12:
                new_gens[k] /= counts[k]
            else:

                new_gens[k] = np.random.rand(2)

        gens = new_gens

    return gens


def optimize_sensor_placement(params, n_sensors=None, n_iter=30):
    if n_sensors is None:
        n_sensors = params.get('N_sensors', 16)


    pdf = get_discrete_pdf(nx=40, ny=40)


    gens = discrete_pdf_sample(n_sensors, pdf, seed=42)


    gens_opt = lloyd_iteration(gens, pdf, n_iter=n_iter)


    L_mem = 0.1
    sensors = gens_opt * L_mem

    return sensors


if __name__ == '__main__':
    p = {'N_sensors': 16}
    sensors = optimize_sensor_placement(p)
    print("Sensors shape:", sensors.shape)
    print("Sensor range:", sensors.min(), sensors.max())
