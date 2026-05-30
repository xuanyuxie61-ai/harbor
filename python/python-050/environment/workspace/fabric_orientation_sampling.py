
import numpy as np
from typing import Tuple


def uniform_on_positive_hemisphere(n_samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = rng.standard_normal((n_samples, 3), dtype=np.float64)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = xyz / norms


    points = np.abs(points)


    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return points


def watson_odf_sample(n_samples: int, concentration: float = 5.0,
                      preferred_direction: np.ndarray = None,
                      seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if preferred_direction is None:
        m = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        m = np.asarray(preferred_direction, dtype=np.float64)
        m = m / np.linalg.norm(m)

    samples = []
    max_trials = n_samples * 100
    trial = 0


    while len(samples) < n_samples and trial < max_trials:
        trial += 1

        xyz = rng.standard_normal(3)
        xyz[2] = abs(xyz[2])
        c = xyz / np.linalg.norm(xyz)


        prob = np.exp(concentration * (np.dot(c, m) ** 2))
        if rng.random() < prob / np.exp(concentration):
            samples.append(c)

    if len(samples) < n_samples:

        while len(samples) < n_samples:
            xyz = rng.standard_normal(3)
            xyz[2] = abs(xyz[2])
            samples.append(xyz / np.linalg.norm(xyz))

    return np.array(samples, dtype=np.float64)


def compute_second_order_tensor(orientations: np.ndarray) -> np.ndarray:
    c = np.asarray(orientations, dtype=np.float64)
    N = len(c)
    if N == 0:
        return np.eye(3, dtype=np.float64) / 3.0


    a2 = np.zeros((3, 3), dtype=np.float64)
    for ci in c:
        a2 += np.outer(ci, ci)
    a2 /= N


    a2 = 0.5 * (a2 + a2.T)
    trace = np.trace(a2)
    if trace > 0:
        a2 = a2 / trace
    return a2


def angular_distance_stats(orientations: np.ndarray) -> dict:
    c = np.asarray(orientations, dtype=np.float64)
    N = len(c)
    if N < 2:
        return {
            'mean_angle_rad': np.pi / 2.0,
            'std_angle_rad': 0.0,
            'mean_cos2': 1.0 / 3.0,
            'j_index': 1.0,
        }


    max_pairs = 10000
    if N * (N - 1) // 2 > max_pairs:
        rng = np.random.default_rng(42)
        idx1 = rng.integers(0, N, max_pairs)
        idx2 = rng.integers(0, N, max_pairs)
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]
    else:
        idx1, idx2 = np.triu_indices(N, k=1)

    dots = np.abs(np.sum(c[idx1] * c[idx2], axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    angles = np.arccos(dots)

    mean_angle = float(np.mean(angles))
    std_angle = float(np.std(angles))
    mean_cos2 = float(np.mean(dots ** 2))


    j_index = float(np.mean(1.0 / (dots + 0.1)))

    return {
        'mean_angle_rad': mean_angle,
        'std_angle_rad': std_angle,
        'mean_cos2': mean_cos2,
        'j_index': j_index,
    }


def fabric_anisotropy_indices(a2_tensor: np.ndarray) -> dict:
    a2 = np.asarray(a2_tensor, dtype=np.float64)
    a2 = 0.5 * (a2 + a2.T)

    evals = np.linalg.eigvalsh(a2)
    evals = np.sort(evals)[::-1]


    evals = evals / np.maximum(np.sum(evals), 1e-15)

    S = 2.0 * evals[0] - 1.0
    G = 1.0 - 2.0 * evals[2]
    I_s = np.sqrt(1.5 * np.sum((evals - 1.0 / 3.0) ** 2))

    return {
        'eigenvalues': evals.tolist(),
        'single_maximum': float(S),
        'girdle': float(G),
        'strength_index': float(I_s),
    }


def monte_carlo_fabric_simulation(n_samples: int = 10000,
                                   concentration: float = 5.0,
                                   seed: int = 42) -> dict:
    orientations = watson_odf_sample(n_samples, concentration, seed=seed)
    a2 = compute_second_order_tensor(orientations)
    stats = angular_distance_stats(orientations)
    indices = fabric_anisotropy_indices(a2)

    return {
        'orientations': orientations,
        'second_order_tensor': a2,
        'angular_stats': stats,
        'anisotropy_indices': indices,
    }
