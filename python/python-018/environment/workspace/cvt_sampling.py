
import numpy as np
from typing import Tuple, Optional, Callable


class CVTSampler:

    def __init__(self, n_generators: int, domain: Tuple[float, float,
                                                          float, float],
                 density_fn: Optional[Callable] = None,
                 rng_seed: Optional[int] = None):
        if n_generators < 1:
            raise ValueError("生成元数量必须为正")
        self.n = n_generators
        self.domain = domain
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.density_fn = density_fn
        self.rng = np.random.RandomState(rng_seed)

    def _sample_uniform(self, n_points: int) -> np.ndarray:
        points = np.zeros((n_points, 2))
        points[:, 0] = self.rng.uniform(self.xmin, self.xmax, n_points)
        points[:, 1] = self.rng.uniform(self.ymin, self.ymax, n_points)
        return points

    def _density_weighted_sample(self, n_points: int) -> np.ndarray:
        if self.density_fn is None:
            return self._sample_uniform(n_points)

        points = np.zeros((n_points, 2))
        count = 0
        max_attempts = n_points * 100
        attempts = 0


        test_points = self._sample_uniform(1000)
        rhos = np.array([self.density_fn(p[0], p[1])
                         for p in test_points])
        rho_max = np.max(rhos) * 1.2 if len(rhos) > 0 else 1.0

        while count < n_points and attempts < max_attempts:
            attempts += 1
            p = self._sample_uniform(1)[0]
            rho = self.density_fn(p[0], p[1])
            if self.rng.rand() < rho / rho_max:
                points[count] = p
                count += 1

        if count < n_points:
            points[count:] = self._sample_uniform(n_points - count)

        return points

    def _find_nearest_generator(self, samples: np.ndarray,
                                 generators: np.ndarray) -> np.ndarray:


        diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def _compute_centroids(self, samples: np.ndarray,
                           nearest: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.n, 2))
        counts = np.zeros(self.n)

        for i in range(len(samples)):
            idx = nearest[i]
            centroids[idx] += samples[i]
            counts[idx] += 1


        for j in range(self.n):
            if counts[j] > 0:
                centroids[j] /= counts[j]
            else:

                centroids[j] = self._sample_uniform(1)[0]

        return centroids

    def lloyd_iterate(self, num_iterations: int = 50,
                       sample_multiplier: int = 1000) -> np.ndarray:

        generators = self._sample_uniform(self.n)
        sample_num = sample_multiplier * self.n

        for it in range(num_iterations):

            samples = self._density_weighted_sample(sample_num)


            nearest = self._find_nearest_generator(samples, generators)


            new_generators = self._compute_centroids(samples, nearest)


            diff = np.max(np.linalg.norm(new_generators - generators, axis=1))
            generators = new_generators

            if diff < 1e-8:
                break

        return generators

    def cvt_energy(self, generators: np.ndarray,
                    sample_num: int = 10000) -> float:
        samples = self._density_weighted_sample(sample_num)
        nearest = self._find_nearest_generator(samples, generators)

        energy = 0.0
        for i in range(len(samples)):
            idx = nearest[i]
            dist2 = np.sum((samples[i] - generators[idx]) ** 2)
            if self.density_fn is not None:
                rho = self.density_fn(samples[i, 0], samples[i, 1])
            else:
                rho = 1.0
            energy += rho * dist2

        return energy / sample_num

    def brillouin_zone_kpoints(self, a: float = 1.0,
                                num_iterations: int = 30) -> np.ndarray:

        self.density_fn = None

        pi = np.pi
        self.domain = (-pi / a, pi / a, -pi / a, pi / a)
        self.xmin, self.xmax, self.ymin, self.ymax = self.domain

        kpoints = self.lloyd_iterate(num_iterations=num_iterations)
        return kpoints

    def impurity_optimized_positions(self, interaction_range: float,
                                      num_iterations: int = 30) -> np.ndarray:

        self.density_fn = None
        positions = self.lloyd_iterate(num_iterations=num_iterations)
        return positions


class BrillouinZoneIntegrator:

    def __init__(self, kpoints: np.ndarray,
                 domain: Tuple[float, float, float, float]):
        self.kpoints = kpoints
        self.domain = domain
        self.n = len(kpoints)

    def integrate(self, integrand_fn: Callable) -> float:
        xmin, xmax, ymin, ymax = self.domain
        area = (xmax - xmin) * (ymax - ymin)
        weight = area / self.n

        total = 0.0
        for k in self.kpoints:
            total += integrand_fn(k[0], k[1])

        return weight * total

    def fermi_surface_sampling(self, dispersion_fn: Callable,
                                e_fermi: float,
                                tolerance: float = 0.1) -> np.ndarray:
        fs_points = []
        for k in self.kpoints:
            e = dispersion_fn(k[0], k[1])
            if abs(e - e_fermi) < tolerance:
                fs_points.append(k)

        return np.array(fs_points) if fs_points else np.array([])



from typing import Callable


def demo():

    cvt = CVTSampler(n_generators=25,
                      domain=(0.0, 1.0, 0.0, 1.0),
                      rng_seed=42)
    generators = cvt.lloyd_iterate(num_iterations=50)
    energy = cvt.cvt_energy(generators)
    print(f"CVT energy with {len(generators)} generators: {energy:.6f}")


    bz_cvt = CVTSampler(n_generators=64,
                         domain=(-np.pi, np.pi, -np.pi, np.pi),
                         rng_seed=42)
    kpts = bz_cvt.brillouin_zone_kpoints(a=1.0, num_iterations=30)
    print(f"BZ k-points range: x=[{kpts[:,0].min():.3f}, {kpts[:,0].max():.3f}], "
          f"y=[{kpts[:,1].min():.3f}, {kpts[:,1].max():.3f}]")


    integrator = BrillouinZoneIntegrator(kpts, bz_cvt.domain)

    def test_func(kx, ky):
        return np.cos(kx) * np.cos(ky)

    result = integrator.integrate(test_func)

    print(f"Integral of cos(kx)cos(ky): {result:.6f}")


if __name__ == "__main__":
    demo()
