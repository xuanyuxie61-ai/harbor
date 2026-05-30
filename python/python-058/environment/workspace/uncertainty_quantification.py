
import numpy as np
from typing import List, Tuple, Callable


def clenshaw_curtis_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    N = n - 1
    theta = np.pi * np.arange(N + 1) / N
    x = np.cos(theta)


    v = np.ones(N + 1)
    v[0] = 0.5
    v[-1] = 0.5
    k = np.arange(N + 1)

    w = np.zeros(N + 1)
    if N % 2 == 0:

        w[0] = 1.0 / (N**2 - 1)
        for j in range(1, N):
            if j % 2 == 0:
                w[j] = 2.0 / (N**2 - 1)
            else:
                w[j] = 0.0
        w[N] = 1.0 / (N**2 - 1)

        g = np.zeros(N // 2)
        g[0] = 1.0
        for m in range(1, N // 2):
            g[m] = -g[m-1] * (N - 2*m + 1) / (N - 2*m)
        for j in range(0, N + 1, 2):
            s = 0
            for m in range(N // 2):
                s += g[m] / (4*m*m - 1) * np.cos(2*m*theta[j])
            w[j] = 4.0 / N * s
    else:

        for j in range(N + 1):
            s = 0.0
            for m in range((N + 1) // 2):
                s += np.sin((2*m + 1) * theta[j]) / (2*m + 1)
            w[j] = 4.0 / N * s


    w = w / np.sum(w) * 2.0
    return x, w


def sparse_grid_index_set(dim: int, level: int) -> List[Tuple[int, ...]]:
    indices = []

    def recurse(current, sum_val, d):
        if d == dim:
            if sum_val <= level + dim - 1:
                indices.append(tuple(current))
            return

        min_i = 1

        max_i = level + dim - 1 - sum_val - (dim - d - 1) * 1
        for i in range(min_i, max_i + 1):
            current.append(i)
            recurse(current, sum_val + i, d + 1)
            current.pop()

    recurse([], 0, 0)
    return indices


def sparse_grid_points_weights(dim: int, level: int) -> Tuple[np.ndarray, np.ndarray]:
    if dim < 1:
        return np.zeros((1, 0)), np.ones(1)
    if level < 0:
        level = 0

    indices = sparse_grid_index_set(dim, level)


    max_level = level + dim - 1
    rules = {}
    for lvl in range(1, max_level + 1):
        n = 2**(lvl - 1) + 1 if lvl > 1 else 1
        x, w = clenshaw_curtis_rule(n)
        rules[lvl] = (x, w)


    points_list = []
    weights_list = []

    for idx in indices:



        prod_points = [rules[i][0] for i in idx]
        prod_weights = [rules[i][1] for i in idx]


        import itertools
        for comb in itertools.product(*[range(len(p)) for p in prod_points]):
            pt = np.array([prod_points[d][comb[d]] for d in range(dim)])
            wt = np.prod([prod_weights[d][comb[d]] for d in range(dim)])
            points_list.append(pt)
            weights_list.append(wt)

    if not points_list:
        return np.zeros((1, dim)), np.ones(1)

    points = np.array(points_list)
    weights = np.array(weights_list)


    unique_pts = []
    unique_wts = []
    tol = 1e-12
    for pt, wt in zip(points, weights):
        found = False
        for j, upt in enumerate(unique_pts):
            if np.linalg.norm(pt - upt) < tol:
                unique_wts[j] += wt
                found = True
                break
        if not found:
            unique_pts.append(pt)
            unique_wts.append(wt)

    points = np.array(unique_pts)
    weights = np.array(unique_wts)

    vol = 2.0**dim
    weights = weights / np.sum(weights) * vol
    return points, weights


def scale_to_physical(points: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    dim = points.shape[1]
    scaled = np.zeros_like(points)
    for d in range(dim):
        a, b = bounds[d]
        scaled[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (b + a)
    return scaled


class EnsembleSparseGridUQ:

    def __init__(self, dim: int, level: int = 2):
        self.dim = dim
        self.level = level
        self.points_std, self.weights = sparse_grid_points_weights(dim, level)
        self.n_points = len(self.points_std)

    def compute_expectation(self, f: Callable[[np.ndarray], float],
                           bounds: List[Tuple[float, float]]) -> float:
        phys_points = scale_to_physical(self.points_std, bounds)
        total = 0.0
        for i in range(self.n_points):
            val = f(phys_points[i])
            if np.isfinite(val):
                total += self.weights[i] * val
        return total / (2.0**self.dim)

    def compute_statistics(self, f: Callable[[np.ndarray], float],
                           bounds: List[Tuple[float, float]]) -> Tuple[float, float]:
        phys_points = scale_to_physical(self.points_std, bounds)
        mean = 0.0
        var = 0.0
        vol = 2.0**self.dim
        for i in range(self.n_points):
            val = f(phys_points[i])
            if np.isfinite(val):
                mean += self.weights[i] * val / vol
                var += self.weights[i] * val**2 / vol
        var = max(0.0, var - mean**2)
        return mean, np.sqrt(var)
