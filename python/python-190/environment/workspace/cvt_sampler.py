
import numpy as np
from scipy.spatial import Delaunay, cKDTree


def cvt_energy(generators: np.ndarray, samples: np.ndarray,
               weights: np.ndarray = None) -> float:
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    tree = cKDTree(generators)
    dists, _ = tree.query(samples, k=1)
    energy = float(np.sum(weights * dists ** 2))
    return energy


def lloyd_step(generators: np.ndarray, samples: np.ndarray,
               weights: np.ndarray = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    tree = cKDTree(generators)
    _, idx = tree.query(samples, k=1)
    k = generators.shape[0]
    new_gens = np.zeros_like(generators)
    for i in range(k):
        mask = idx == i
        if np.sum(mask) == 0:

            new_gens[i] = generators[i]
        else:
            w = weights[mask]
            new_gens[i] = np.sum(samples[mask] * w[:, None], axis=0) / np.sum(w)
    return new_gens


def cvt_2d_sampling(k: int = 16, n_samples: int = 5000,
                    itermax: int = 50, tol: float = 1e-5,
                    bounds: tuple = ((0.0, 1.0), (0.0, 1.0)),
                    seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    gens = rng.random((k, 2))
    gens[:, 0] = gens[:, 0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    gens[:, 1] = gens[:, 1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]

    for it in range(itermax):

        samples = rng.random((n_samples, 2))
        samples[:, 0] = samples[:, 0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
        samples[:, 1] = samples[:, 1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
        new_gens = lloyd_step(gens, samples)
        motion = float(np.max(np.sqrt(np.sum((new_gens - gens) ** 2, axis=1))))
        gens = new_gens
        if motion < tol:
            break
    return gens


def cvt_latent_samples(k: int = 32, dim: int = 8, n_samples: int = 10000,
                       itermax: int = 30, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gens = rng.random((k, dim))
    for it in range(itermax):
        samples = rng.random((n_samples, dim))
        new_gens = lloyd_step(gens, samples)
        motion = float(np.max(np.sqrt(np.sum((new_gens - gens) ** 2, axis=1))))
        gens = new_gens
        if motion < 1e-4:
            break
    return gens
