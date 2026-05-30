
import numpy as np


def cvt_lloyd_2d(generators: np.ndarray, density_func, bounds: tuple,
                 n_samples: int = 5000, n_iterations: int = 30):
    generators = np.asarray(generators, dtype=float)
    N = generators.shape[0]
    xmin, xmax, ymin, ymax = bounds

    energy_history = []

    for it in range(n_iterations):

        xs = np.random.uniform(xmin, xmax, size=n_samples)
        ys = np.random.uniform(ymin, ymax, size=n_samples)
        samples = np.column_stack((xs, ys))


        rhos = np.array([density_func(s[0], s[1]) for s in samples], dtype=float)
        rhos = np.clip(rhos, 0.0, None)


        diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)


        new_generators = np.zeros_like(generators)
        weights = np.zeros(N, dtype=float)
        for i in range(N):
            mask = nearest == i
            if np.any(mask):
                new_generators[i] = np.sum(samples[mask] * rhos[mask][:, np.newaxis], axis=0) / np.sum(rhos[mask])
                weights[i] = np.sum(rhos[mask])
            else:

                new_generators[i] = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]


        min_dists = np.min(dists, axis=1)
        energy = np.sum(rhos * min_dists)
        energy_history.append(float(energy))

        generators = new_generators.copy()

    return generators, energy_history


def cvt_circle_nonuniform_density(n: int, radius: float = 1.0,
                                  n_iterations: int = 20, n_samples: int = None):
    if n_samples is None:
        n_samples = 2000 * n


    gens = []
    while len(gens) < n:
        candidates = np.random.uniform(-radius, radius, size=(2 * n, 2))
        mask = np.sum(candidates ** 2, axis=1) <= radius ** 2
        gens.extend(candidates[mask].tolist())
    generators = np.array(gens[:n], dtype=float)

    def density_func(x, y):
        r2 = (x ** 2 + y ** 2) / (radius ** 2)
        return 1.0 + 5.0 * np.exp(-5.0 * r2)

    bounds = (-radius, radius, -radius, radius)
    generators, energy_history = cvt_lloyd_2d(generators, density_func, bounds,
                                             n_samples=n_samples, n_iterations=n_iterations)

    dists = np.linalg.norm(generators, axis=1)
    mask = dists > radius
    generators[mask] *= (radius / dists[mask])[:, np.newaxis]
    return generators, energy_history


def coverage_metric(positions: np.ndarray, density_func, bounds: tuple, n_samples: int = 10000):
    xmin, xmax, ymin, ymax = bounds
    xs = np.random.uniform(xmin, xmax, size=n_samples)
    ys = np.random.uniform(ymin, ymax, size=n_samples)
    samples = np.column_stack((xs, ys))
    rhos = np.array([density_func(s[0], s[1]) for s in samples], dtype=float)
    rhos = np.clip(rhos, 0.0, None)
    diffs = samples[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    min_dists = np.min(dists, axis=1)
    energy = float(np.sum(rhos * min_dists) / np.sum(rhos))
    return energy
