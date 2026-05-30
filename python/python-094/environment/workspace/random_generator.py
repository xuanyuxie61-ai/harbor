
import numpy as np


def cliff_next(x):
    if x <= 0.0 or x >= 1.0:
        return np.nan
    return (-100.0 * np.log(x)) % 1.0


class CliffGenerator:

    def __init__(self, seed=0.314159265):
        if seed <= 0.0 or seed >= 1.0:
            raise ValueError("seed must be in (0, 1).")
        self.state = float(seed)
        self._validate_state()

    def _validate_state(self):
        if not (0.0 < self.state < 1.0) or not np.isfinite(self.state):
            raise RuntimeError("Cliff generator state became invalid.")

    def next(self):
        self.state = cliff_next(self.state)
        if np.isnan(self.state):

            self.state = 0.5
        self._validate_state()
        return self.state

    def rand(self, size=None):
        if size is None:
            return self.next()
        size = tuple(np.atleast_1d(size))
        arr = np.zeros(size, dtype=float)
        for idx in np.ndindex(size):
            arr[idx] = self.next()
        return arr

    def randn(self, size=None):
        if size is None:
            u1 = self.next()
            u2 = self.next()
            while u1 <= 1e-10:
                u1 = self.next()
            return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

        size = tuple(np.atleast_1d(size))
        arr = np.zeros(size, dtype=float)
        flat = arr.ravel()
        for i in range(0, len(flat), 2):
            u1 = self.next()
            u2 = self.next()
            while u1 <= 1e-10:
                u1 = self.next()
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
            flat[i] = z0
            if i + 1 < len(flat):
                flat[i + 1] = z1
        return arr


class StratifiedSampler:

    def __init__(self, dim, n_strata):
        self.dim = int(dim)
        self.n_strata = int(n_strata)

    def sample(self, a, b, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        a = np.full(self.dim, a, dtype=float) if np.isscalar(a) else np.asarray(a, dtype=float)
        b = np.full(self.dim, b, dtype=float) if np.isscalar(b) else np.asarray(b, dtype=float)

        n_points = self.n_strata ** self.dim
        samples = np.zeros((n_points, self.dim), dtype=float)

        idx = 0


        grids = [np.linspace(a[d], b[d], self.n_strata + 1) for d in range(self.dim)]

        from itertools import product
        for cell in product(range(self.n_strata), repeat=self.dim):
            point = np.zeros(self.dim, dtype=float)
            for d in range(self.dim):
                low = grids[d][cell[d]]
                high = grids[d][cell[d] + 1]
                point[d] = low + rng.random() * (high - low)
            samples[idx, :] = point
            idx += 1

        return samples


def latin_hypercube_sampling(n_samples, dim, a=0.0, b=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = np.zeros((n_samples, dim), dtype=float)
    for d in range(dim):
        perm = rng.permutation(n_samples)
        u = rng.random(n_samples)
        samples[:, d] = (perm + u) / n_samples
    return a + samples * (b - a)
