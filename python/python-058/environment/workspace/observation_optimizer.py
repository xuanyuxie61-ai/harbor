
import numpy as np
from typing import Tuple, List


def find_closest_generators(samples: np.ndarray, generators: np.ndarray) -> np.ndarray:
    n_samples = len(samples)
    n_gen = len(generators)
    closest = np.zeros(n_samples, dtype=int)
    for s in range(n_samples):
        dists = np.sum((generators - samples[s])**2, axis=1)
        closest[s] = int(np.argmin(dists))
    return closest


def cvt_energy(samples: np.ndarray, generators: np.ndarray) -> float:
    n_samples = len(samples)
    total = 0.0
    for s in range(n_samples):
        dists = np.sum((generators - samples[s])**2, axis=1)
        total += np.min(dists)
    return total / n_samples


def ccvt_reflect_2d(n_generators: int, domain: Tuple[float, float, float, float],
                    max_iter: int = 100, sample_num: int = 10000,
                    tol: float = 1e-6) -> np.ndarray:
    xmin, xmax, ymin, ymax = domain


    np.random.seed(42)
    generators = np.column_stack([
        np.random.uniform(xmin, xmax, n_generators),
        np.random.uniform(ymin, ymax, n_generators)
    ])

    energy_history = []

    for it in range(max_iter):

        samples = np.column_stack([
            np.random.uniform(xmin, xmax, sample_num),
            np.random.uniform(ymin, ymax, sample_num)
        ])


        closest = find_closest_generators(samples, generators)


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        for s in range(sample_num):
            g_idx = closest[s]
            new_generators[g_idx] += samples[s]
            counts[g_idx] += 1.0


        for g in range(n_generators):
            if counts[g] > 0:
                new_generators[g] /= counts[g]
            else:

                new_generators[g] = generators[g] + np.random.randn(2) * 0.01 * (xmax - xmin)


            if new_generators[g, 0] < xmin:
                new_generators[g, 0] = 2.0 * xmin - new_generators[g, 0]
            if new_generators[g, 0] > xmax:
                new_generators[g, 0] = 2.0 * xmax - new_generators[g, 0]
            if new_generators[g, 1] < ymin:
                new_generators[g, 1] = 2.0 * ymin - new_generators[g, 1]
            if new_generators[g, 1] > ymax:
                new_generators[g, 1] = 2.0 * ymax - new_generators[g, 1]


            new_generators[g, 0] = max(xmin, min(xmax, new_generators[g, 0]))
            new_generators[g, 1] = max(ymin, min(ymax, new_generators[g, 1]))


        shift = np.max(np.linalg.norm(new_generators - generators, axis=1))
        generators = new_generators
        energy = cvt_energy(samples, generators)
        energy_history.append(energy)

        if shift < tol * (xmax - xmin) and it > 10:
            break

    return generators


class ObservationNetworkOptimizer:

    def __init__(self, domain_km: Tuple[float, float, float, float] = (0.0, 200.0, 0.0, 200.0)):
        self.domain = domain_km

    def optimize_radar_placement(self, n_radars: int = 5) -> np.ndarray:
        return ccvt_reflect_2d(n_radars, self.domain, max_iter=80, sample_num=5000)

    def optimize_station_network(self, n_stations: int = 20) -> np.ndarray:
        return ccvt_reflect_2d(n_stations, self.domain, max_iter=60, sample_num=8000)

    def coverage_score(self, generators: np.ndarray, n_test: int = 5000) -> float:
        xmin, xmax, ymin, ymax = self.domain
        samples = np.column_stack([
            np.random.uniform(xmin, xmax, n_test),
            np.random.uniform(ymin, ymax, n_test)
        ])
        energy = cvt_energy(samples, generators)

        area = (xmax - xmin) * (ymax - ymin)
        n_gen = len(generators)
        optimal_energy = area / (np.pi * n_gen) * 0.5
        score = optimal_energy / (energy + 1e-10)
        return min(1.0, score)
