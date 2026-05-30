
import numpy as np
from typing import Tuple, List


def latin_hypercube_sampling(dim: int, n_points: int,
                              domain: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
    if domain is None:
        low = np.zeros(dim)
        high = np.ones(dim)
    else:
        low = np.asarray(domain[0])
        high = np.asarray(domain[1])

    samples = np.zeros((n_points, dim))
    for d in range(dim):
        perm = np.random.permutation(n_points)

        samples[:, d] = (perm + 0.5) / n_points


    return low + samples * (high - low)


def latin_center_sampling(dim: int, n_points: int,
                           domain: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
    if domain is None:
        low = np.zeros(dim)
        high = np.ones(dim)
    else:
        low = np.asarray(domain[0])
        high = np.asarray(domain[1])

    samples = np.zeros((n_points, dim))
    for d in range(dim):
        perm = np.random.permutation(n_points)
        samples[:, d] = (2.0 * perm + 1.0) / (2.0 * n_points)

    return low + samples * (high - low)


def triangle_grid_points(n: int, vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices)
    dim = vertices.shape[1]
    n_points = (n + 1) * (n + 2) // 2
    points = np.zeros((n_points, dim))
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            points[p] = (i * vertices[0] + j * vertices[1] + k * vertices[2]) / n
            p += 1
    return points


def triangle_grid_count(n: int) -> int:
    return (n + 1) * (n + 2) // 2


def set_partition_equivalence(n_elements: int,
                               relation_matrix: np.ndarray = None) -> List[List[int]]:
    if relation_matrix is None:

        return [[i] for i in range(n_elements)]

    R = np.asarray(relation_matrix)
    visited = np.zeros(n_elements, dtype=bool)
    classes = []

    for i in range(n_elements):
        if visited[i]:
            continue

        equiv_class = [i]
        visited[i] = True
        for j in range(i + 1, n_elements):
            if not visited[j] and R[i, j] > 0.5:
                equiv_class.append(j)
                visited[j] = True
        classes.append(equiv_class)

    return classes


def power_set_non_empty(n: int) -> List[List[int]]:
    subsets = []
    for mask in range(1, 1 << n):
        subset = [i for i in range(n) if mask & (1 << i)]
        subsets.append(subset)
    return subsets


def stratified_sampling(n_strata: int, dim: int,
                        samples_per_stratum: int = 1) -> np.ndarray:
    total_samples = (n_strata ** dim) * samples_per_stratum
    samples = np.zeros((total_samples, dim))
    idx = 0

    import itertools
    for cell in itertools.product(range(n_strata), repeat=dim):
        for _ in range(samples_per_stratum):
            s = np.random.rand(dim)
            samples[idx] = (np.array(cell) + s) / n_strata
            idx += 1
    return samples


def sobol_like_sampling(dim: int, n_points: int) -> np.ndarray:
    samples = np.zeros((n_points, dim))
    for d in range(dim):
        base = d + 2
        for i in range(n_points):

            val = 0.0
            f = 1.0 / base
            n = i + 1
            while n > 0:
                val += f * (n % base)
                n //= base
                f /= base
            samples[i, d] = val
    return samples
