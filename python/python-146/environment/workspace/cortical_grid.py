
import numpy as np


class CorticalGrid:

    def __init__(self, nx, ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be positive.")
        self.nx = nx
        self.ny = ny
        self.xlim = xlim
        self.ylim = ylim
        self.N = nx * ny
        self._generate_grid()

    def _generate_grid(self):
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.positions = np.column_stack([
            self.X_grid.ravel(), self.Y_grid.ravel()
        ])

    def euclidean_distance(self, i, j):
        if not (0 <= i < self.N and 0 <= j < self.N):
            raise IndexError("Index out of bounds.")
        dx = self.positions[i, 0] - self.positions[j, 0]
        dy = self.positions[i, 1] - self.positions[j, 1]
        return np.sqrt(dx ** 2 + dy ** 2)

    def connection_probability(self, i, j, p0=0.3, sigma=0.5):
        if i == j:
            return 0.0
        d = self.euclidean_distance(i, j)
        return p0 * np.exp(-d ** 2 / (2.0 * sigma ** 2))

    def build_connectivity_matrix(self, p0=0.3, sigma=0.5):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    W[i, j] = self.connection_probability(i, j, p0, sigma)
        return W

    def distance_statistics(self, n_samples=10000):
        rng = np.random.default_rng(seed=11)
        idx1 = rng.integers(0, self.N, size=n_samples)
        idx2 = rng.integers(0, self.N, size=n_samples)
        distances = np.zeros(n_samples)
        for k in range(n_samples):
            distances[k] = self.euclidean_distance(idx1[k], idx2[k])
        dmu = np.mean(distances)
        dvar = np.var(distances)
        return dmu, dvar, distances

    def spatial_receptive_field(self, i, sigma_rf=0.2):
        center = self.positions[i]
        diff = self.positions - center
        dist_sq = np.sum(diff ** 2, axis=1)
        weights = np.exp(-dist_sq / (2.0 * sigma_rf ** 2))
        weights[i] = 0.0
        return weights


def demo_grid_encoding():
    grid = CorticalGrid(nx=5, ny=5, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    W = grid.build_connectivity_matrix(p0=0.4, sigma=0.3)
    dmu, dvar, _ = grid.distance_statistics(n_samples=2000)
    rf = grid.spatial_receptive_field(i=12, sigma_rf=0.25)
    return W, dmu, dvar, rf


def demo_distance_stats():
    grid = CorticalGrid(nx=10, ny=10, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    dmu, dvar, distances = grid.distance_statistics(n_samples=5000)
    return dmu, dvar
