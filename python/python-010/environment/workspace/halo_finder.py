
import numpy as np
from typing import Tuple, List
from collections import deque


class HaloFinder:

    def __init__(self, L: float, linking_length: float = None):
        self.L = L
        self.linking_length = linking_length

    def _distance_periodic(self, p1: np.ndarray, p2: np.ndarray) -> float:
        diff = np.abs(p1 - p2)
        diff = np.minimum(diff, self.L - diff)
        return np.sqrt(np.sum(diff ** 2))

    def fof_groups(
        self, pos: np.ndarray, mass: np.ndarray = None
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        n_part = pos.shape[0]
        if self.linking_length is None:
            self.linking_length = 0.2 * (self.L ** 3 / n_part) ** (1.0 / 3.0)

        visited = np.zeros(n_part, dtype=bool)
        groups = []
        halo_mass = []


        n_bins = max(1, int(self.L / self.linking_length))
        bin_size = self.L / n_bins
        bins = {}
        for i in range(n_part):
            bx = int(pos[i, 0] / bin_size) % n_bins
            by = int(pos[i, 1] / bin_size) % n_bins
            bz = int(pos[i, 2] / bin_size) % n_bins
            key = (bx, by, bz)
            if key not in bins:
                bins[key] = []
            bins[key].append(i)

        def get_neighbors(idx: int) -> List[int]:
            p = pos[idx]
            bx = int(p[0] / bin_size) % n_bins
            by = int(p[1] / bin_size) % n_bins
            bz = int(p[2] / bin_size) % n_bins
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        key = (
                            (bx + dx) % n_bins,
                            (by + dy) % n_bins,
                            (bz + dz) % n_bins,
                        )
                        if key in bins:
                            for j in bins[key]:
                                if j != idx and not visited[j]:
                                    dist = self._distance_periodic(p, pos[j])
                                    if dist < self.linking_length:
                                        neighbors.append(j)
            return neighbors

        for i in range(n_part):
            if visited[i]:
                continue

            queue = deque([i])
            visited[i] = True
            group = [i]
            while queue:
                cur = queue.popleft()
                for nb in get_neighbors(cur):
                    if not visited[nb]:
                        visited[nb] = True
                        group.append(nb)
                        queue.append(nb)
            groups.append(np.array(group))
            if mass is not None:
                halo_mass.append(mass[group].sum())
            else:
                halo_mass.append(len(group))

        return groups, np.array(halo_mass)

    def spherical_overdensity_mass(
        self,
        pos: np.ndarray,
        mass: np.ndarray,
        center: np.ndarray,
        rho_crit: float,
        Delta: float = 200.0,
    ) -> Tuple[float, float]:

        diff = np.abs(pos - center)
        diff = np.minimum(diff, self.L - diff)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        sort_idx = np.argsort(dist)
        sorted_dist = dist[sort_idx]
        sorted_mass = mass[sort_idx]
        cum_mass = np.cumsum(sorted_mass)

        volumes = (4.0 / 3.0) * np.pi * sorted_dist ** 3
        volumes = np.clip(volumes, 1e-30, None)
        rho_avg = cum_mass / volumes
        mask = rho_avg >= Delta * rho_crit
        if mask.sum() == 0:
            return 0.0, 0.0
        idx = np.where(mask)[0][-1]
        return cum_mass[idx], sorted_dist[idx]


def level_set_volume_analysis(
    delta_grid: np.ndarray,
    L: float,
    n_levels: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    delta_min = delta_grid.min()
    delta_max = delta_grid.max()
    levels = np.linspace(delta_min, delta_max, n_levels)
    total_volume = L ** 3
    volumes = np.zeros(n_levels)
    dx3 = (L / delta_grid.shape[0]) ** 3
    for i, lev in enumerate(levels):
        mask = delta_grid >= lev
        volumes[i] = mask.sum() * dx3 / total_volume
    return levels, volumes


def sample_sphere_positive_distance(n_samples: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    x = rng.standard_normal((n_samples, 3))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return np.abs(x) / norms


def angular_distance_histogram(
    directions: np.ndarray, n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(directions)
    if n < 2:
        return np.array([]), np.array([])

    max_pairs = min(n * (n - 1) // 2, 50000)
    rng = np.random.default_rng(seed=42)
    angles = []
    for _ in range(max_pairs):
        i, j = rng.integers(0, n, 2)
        if i != j:
            cos_theta = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
            angles.append(np.arccos(cos_theta))
    angles = np.array(angles)
    hist, edges = np.histogram(angles, bins=n_bins, range=(0.0, np.pi))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    bin_width = edges[1] - edges[0]
    pdf = hist / (hist.sum() * bin_width)
    return bin_centers, pdf


def halo_mass_function_from_groups(
    halo_mass: np.ndarray, volume: float, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(halo_mass) == 0:
        return np.array([]), np.array([]), np.array([])
    logM = np.log10(halo_mass[halo_mass > 0])
    if len(logM) == 0:
        return np.array([]), np.array([]), np.array([])
    bins = np.linspace(logM.min(), logM.max(), n_bins + 1)
    counts, edges = np.histogram(logM, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    dlnM = (edges[1:] - edges[:-1]) * np.log(10)
    dn_dlnM = counts / (volume * dlnM)

    err = np.sqrt(np.clip(counts, 0, None)) / (volume * dlnM)
    return bin_centers, dn_dlnM, err


if __name__ == "__main__":

    pos = np.random.rand(1000, 3) * 100.0
    mass = np.ones(1000) * 1e10
    finder = HaloFinder(L=100.0)
    groups, hmass = finder.fof_groups(pos, mass)
    print(f"识别到 {len(groups)} 个 FOF 晕")
    print(f"最大晕质量: {hmass.max():.3e}")


    delta = np.random.randn(32, 32, 32) * 0.5
    lev, vol = level_set_volume_analysis(delta, 100.0, n_levels=10)
    print(f"水平集体积分数范围: [{vol.min():.4f}, {vol.max():.4f}]")


    dirs = sample_sphere_positive_distance(1000)
    bc, pdf = angular_distance_histogram(dirs)
    print(f"角距离均值: {bc.mean():.4f} (理论 π/2 ≈ {np.pi/2:.4f})")
