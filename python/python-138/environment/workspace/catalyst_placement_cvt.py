
import numpy as np
from typing import Tuple, List, Optional, Callable


class CatalystCVTPlacer:

    def __init__(
        self,
        dim: int = 2,
        n_generators: int = 64,
        bounds: Optional[np.ndarray] = None,
        density_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sample_num: int = 20000,
        max_iter: int = 50,
        tol: float = 1.0e-5,
    ):
        if dim not in (2, 3):
            raise ValueError("仅支持 2D 或 3D 催化剂分布")
        if n_generators < 1:
            raise ValueError("生成元数量必须至少为 1")
        if sample_num < 100:
            raise ValueError("采样点数量过少")

        self.dim = dim
        self.n = n_generators
        self.sample_num = sample_num
        self.max_iter = max_iter
        self.tol = tol

        if bounds is None:
            self.bounds = np.array([[0.0, 1.0]] * dim, dtype=float)
        else:
            self.bounds = np.array(bounds, dtype=float)
            if self.bounds.shape != (dim, 2):
                raise ValueError("bounds 形状应为 (dim, 2)")

        self.density_func = density_func
        if density_func is None:

            self.density_func = lambda pts: np.ones(pts.shape[0])


        self.generators = self._init_generators()

    def _init_generators(self) -> np.ndarray:
        gens = np.zeros((self.n, self.dim))
        for d in range(self.dim):
            lo, hi = self.bounds[d]
            gens[:, d] = np.random.uniform(lo, hi, self.n)
        return gens

    def _sample_points(self) -> np.ndarray:

        samples = []
        batch = min(self.sample_num * 2, 50000)
        while len(samples) < self.sample_num:
            pts = np.zeros((batch, self.dim))
            for d in range(self.dim):
                lo, hi = self.bounds[d]
                pts[:, d] = np.random.uniform(lo, hi, batch)
            dens = self.density_func(pts)
            if np.max(dens) <= 0.0:
                dens = np.ones_like(dens)
            prob = dens / np.max(dens)
            mask = np.random.rand(batch) < prob
            accepted = pts[mask]
            samples.extend(accepted.tolist())
        samples = np.array(samples[: self.sample_num])
        return samples

    def _find_closest(self, points: np.ndarray) -> np.ndarray:


        diff = points[:, np.newaxis, :] - self.generators[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        closest = np.argmin(dists, axis=1)
        return closest

    def _compute_energy(self, samples: np.ndarray, closest: np.ndarray) -> float:
        dens = self.density_func(samples)
        energy = 0.0
        for i in range(self.n):
            mask = closest == i
            if np.any(mask):
                diff = samples[mask] - self.generators[i]
                dist2 = np.sum(diff ** 2, axis=1)
                energy += np.sum(dens[mask] * dist2)
        return energy / len(samples)

    def iterate(self) -> Tuple[np.ndarray, float, float]:
        energy_history = []
        for it in range(self.max_iter):
            samples = self._sample_points()
            closest = self._find_closest(samples)
            dens = self.density_func(samples)

            new_gens = np.zeros_like(self.generators)
            counts = np.zeros(self.n)
            for i in range(self.n):
                mask = closest == i
                if np.any(mask):
                    weights = dens[mask]
                    total_weight = np.sum(weights)
                    if total_weight > 0.0:
                        new_gens[i] = np.sum(samples[mask] * weights[:, np.newaxis], axis=0) / total_weight
                    else:
                        new_gens[i] = self.generators[i]
                    counts[i] = total_weight
                else:

                    new_gens[i] = self._init_generators()[0]


            for d in range(self.dim):
                lo, hi = self.bounds[d]
                new_gens[:, d] = np.clip(new_gens[:, d], lo, hi)

            shifts = np.sqrt(np.sum((new_gens - self.generators) ** 2, axis=1))
            max_shift = np.max(shifts)
            self.generators = new_gens

            energy = self._compute_energy(samples, closest)
            energy_history.append(energy)

            if max_shift < self.tol:
                break

        return self.generators, energy, max_shift

    def compute_uniformity_index(self) -> float:
        samples = self._sample_points()
        closest = self._find_closest(samples)
        dens = self.density_func(samples)
        volumes = np.zeros(self.n)
        for i in range(self.n):
            mask = closest == i
            if np.any(mask):
                volumes[i] = np.sum(dens[mask])
        mu = np.mean(volumes)
        if mu < 1.0e-12:
            return 0.0
        sigma = np.std(volumes)
        eta = max(0.0, 1.0 - sigma / mu)
        return eta

    def get_catalyst_loading_map(self, grid_res: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.dim != 2:
            raise NotImplementedError("仅 2D 支持网格密度图")
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], grid_res)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], grid_res)
        xv, yv = np.meshgrid(x, y)
        pts = np.column_stack([xv.ravel(), yv.ravel()])
        closest = self._find_closest(pts)

        density = np.zeros(len(pts))
        for i in range(len(pts)):
            gen = self.generators[closest[i]]
            dist2 = np.sum((pts[i] - gen) ** 2)
            density[i] = np.exp(-dist2 / (2.0 * 0.01 ** 2))
        density_map = density.reshape((grid_res, grid_res))
        return density_map, np.stack([xv, yv], axis=-1)
