
import numpy as np
from typing import Callable, Tuple


def density_transform(s: float, density_type: int = 0) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    if density_type == 0:
        return s
    elif density_type == 1:
        return np.sqrt(s)
    elif density_type == 2:
        return s ** (1.0 / 3.0)
    elif density_type == 3:
        return s ** (1.0 / 4.0)
    elif density_type == 4:
        euler = np.e
        return np.log(euler / (euler - s * (euler - 1.0)))
    elif density_type == 5:
        return 0.5 + np.arctan(50.0 * (s - 0.5)) / np.pi
    elif density_type == 6:
        return np.sin(np.pi * (s - 0.5))
    else:
        return s


class CVTOptimizer:

    def __init__(self, dim: int = 2, n_generators: int = 16,
                 domain: Tuple[np.ndarray, np.ndarray] = None,
                 density_func: Callable = None,
                 max_iter: int = 100,
                 tol: float = 1e-5):
        self.dim = dim
        self.n = n_generators
        if domain is None:
            self.low = np.zeros(dim)
            self.high = np.ones(dim)
        else:
            self.low = np.asarray(domain[0])
            self.high = np.asarray(domain[1])
        self.density_func = density_func if density_func is not None else lambda x: 1.0
        self.max_iter = max_iter
        self.tol = tol
        self.generators = None
        self.energy_history = []

    def initialize_generators(self, method: str = "latin_hypercube"):
        if method == "random":
            self.generators = np.random.rand(self.n, self.dim)
            self.generators = self.low + self.generators * (self.high - self.low)
        elif method == "grid":

            n_side = int(np.ceil(self.n ** (1.0 / self.dim)))
            coords = [np.linspace(0, 1, n_side) for _ in range(self.dim)]
            pts = []
            import itertools
            for combo in itertools.product(*coords):
                pts.append(combo)
                if len(pts) >= self.n:
                    break
            self.generators = np.array(pts[:self.n])
            self.generators = self.low + self.generators * (self.high - self.low)
        elif method == "latin_hypercube":

            self.generators = np.zeros((self.n, self.dim))
            for d in range(self.dim):
                perm = np.random.permutation(self.n)
                self.generators[:, d] = (perm + 0.5) / self.n
            self.generators = self.low + self.generators * (self.high - self.low)
        elif method == "zeros":
            self.generators = np.zeros((self.n, self.dim))
        else:
            self.generators = np.random.rand(self.n, self.dim)
            self.generators = self.low + self.generators * (self.high - self.low)

    def _find_nearest(self, sample: np.ndarray) -> int:
        diffs = self.generators - sample
        dists_sq = np.sum(diffs ** 2, axis=1)
        return int(np.argmin(dists_sq))

    def _quantize_error(self) -> float:
        n_samples = max(self.n * 50, 500)
        error = 0.0
        for _ in range(n_samples):
            s = np.random.rand(self.dim)
            sample = self.low + s * (self.high - self.low)
            nearest = self._find_nearest(sample)
            rho = self.density_func(sample)
            dist_sq = np.sum((sample - self.generators[nearest]) ** 2)
            error += rho * dist_sq
        return error / n_samples

    def optimize(self, sample_multiplier: int = 50) -> np.ndarray:
        if self.generators is None:
            self.initialize_generators()

        n_samples = self.n * sample_multiplier
        self.energy_history = []

        for iteration in range(self.max_iter):
            generator_new = np.zeros_like(self.generators)
            tally = np.zeros(self.n)

            for _ in range(n_samples):
                sample = self.low + np.random.rand(self.dim) * (self.high - self.low)
                nearest = self._find_nearest(sample)
                generator_new[nearest] += sample
                tally[nearest] += 1.0


            for j in range(self.n):
                if tally[j] > 0:
                    self.generators[j] = generator_new[j] / tally[j]


            if iteration % 5 == 0:
                energy = self._quantize_error()
                self.energy_history.append(energy)
                if len(self.energy_history) > 1:
                    rel_change = abs(self.energy_history[-1] - self.energy_history[-2])
                    if rel_change < self.tol * abs(self.energy_history[-2] + 1e-12):
                        break

        return self.generators.copy()

    def get_sampling_weights(self) -> np.ndarray:
        n_test = self.n * 100
        counts = np.zeros(self.n)
        for _ in range(n_test):
            sample = self.low + np.random.rand(self.dim) * (self.high - self.low)
            nearest = self._find_nearest(sample)
            counts[nearest] += 1.0
        return counts / n_test


def cvt_1d_nonuniform_python(n_generators: int = 10,
                              density_type: int = 0,
                              n_steps: int = 100,
                              n_samples_per_step: int = 1000) -> np.ndarray:

    generators = np.sort(np.random.rand(n_generators))

    for step in range(n_steps):
        gen_new = np.zeros(n_generators)
        tally = np.zeros(n_generators)

        for _ in range(n_samples_per_step):
            s = np.random.rand()
            s_transformed = density_transform(s, density_type)
            s_transformed = np.clip(s_transformed, 0.0, 1.0)


            dists = np.abs(generators - s_transformed)
            nearest = int(np.argmin(dists))
            gen_new[nearest] += s_transformed
            tally[nearest] += 1.0

        for j in range(n_generators):
            if tally[j] > 0:
                generators[j] = gen_new[j] / tally[j]

    return generators
