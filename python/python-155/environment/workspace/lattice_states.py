import numpy as np
from typing import List, Tuple, Optional
from utils import gcd_vec





def diophantine_nd_check(a: np.ndarray, b: int) -> bool:
    if b < 0:
        return False
    if np.any(a <= 0):
        return False
    g = gcd_vec(a)
    if b % g != 0:
        return False
    return True


def diophantine_nd_nonnegative(a: np.ndarray, b: int) -> List[np.ndarray]:
    n = len(a)
    solutions = []
    if n == 0:
        return solutions
    if not diophantine_nd_check(a, b):
        return solutions

    def backtrack(idx: int, residual: int, current: List[int]):
        if idx == n - 1:
            if residual % a[idx] == 0:
                val = residual // a[idx]
                current.append(val)
                solutions.append(np.array(current, dtype=int))
                current.pop()
            return
        max_val = residual // a[idx]
        for val in range(max_val, -1, -1):
            current.append(val)
            backtrack(idx + 1, residual - a[idx] * val, current)
            current.pop()

    backtrack(0, b, [])
    return solutions


def diophantine_nd_nonnegative_bounded(a: np.ndarray, b: int,
                                       bounds: np.ndarray) -> List[np.ndarray]:
    n = len(a)
    solutions = []
    if n == 0:
        return solutions

    def backtrack(idx: int, residual: int, current: List[int]):
        if idx == n - 1:
            if residual % a[idx] == 0:
                val = residual // a[idx]
                if val <= bounds[idx]:
                    current.append(val)
                    solutions.append(np.array(current, dtype=int))
                    current.pop()
            return
        max_val = min(residual // a[idx], bounds[idx])
        for val in range(max_val, -1, -1):
            current.append(val)
            backtrack(idx + 1, residual - a[idx] * val, current)
            current.pop()

    backtrack(0, b, [])
    return solutions





def cc_level_to_order(level: int) -> int:
    if level < 0:
        return 1
    if level == 0:
        return 1
    return 2 ** level + 1


def cc_abscissa(order: int) -> np.ndarray:
    if order < 1:
        return np.array([0.0])
    if order == 1:
        return np.array([0.0])
    i = np.arange(order)
    return np.cos((order - 1 - i) * np.pi / (order - 1))


def _generate_compositions(n: int, k: int):
    if k == 1:
        yield [n]
        return
    for i in range(n + 1):
        for tail in _generate_compositions(n - i, k - 1):
            yield [i] + tail


def generate_cc_sparse_grid(dim: int, max_level: int) -> np.ndarray:
    from itertools import product
    points = []
    for total_level in range(max_level + 1):
        for levels in _generate_compositions(total_level, dim):
            orders = [cc_level_to_order(lv) for lv in levels]
            grids = [cc_abscissa(o) for o in orders]
            for coord in product(*grids):
                points.append(np.array(coord, dtype=float))
    if not points:
        return np.zeros((0, dim))

    pts = np.array(points)
    unique = []
    seen = set()
    for p in pts:
        key = tuple(np.round(p, 12))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return np.array(unique)


def constrained_parameter_grid(dim: int, max_level: int,
                               weights: Optional[np.ndarray] = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(dim)
    from itertools import product
    points = []
    for total_level in range(max_level + 1):
        for levels in _generate_compositions(total_level, dim):
            if np.dot(weights, levels) <= max_level:
                orders = [cc_level_to_order(lv) for lv in levels]
                grids = [cc_abscissa(o) for o in orders]
                for coord in product(*grids):
                    points.append(np.array(coord, dtype=float))
    if not points:
        return np.zeros((0, dim))
    pts = np.array(points)
    unique = []
    seen = set()
    for p in pts:
        key = tuple(np.round(p, 12))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return np.array(unique)





def random_level_sample(values: np.ndarray, num_levels: int,
                        seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if num_levels <= 0 or values.size == 0:
        return np.array([])
    indices = np.random.choice(values.size, size=min(num_levels, values.size), replace=False)
    return np.sort(values.flat[indices])


def analyze_probability_landscape(prob_grid: np.ndarray, num_levels: int = 20,
                                  seed: Optional[int] = None) -> dict:
    levels = random_level_sample(prob_grid, num_levels, seed)
    return {
        "levels": levels,
        "min_prob": float(np.min(prob_grid)),
        "max_prob": float(np.max(prob_grid)),
        "mean_prob": float(np.mean(prob_grid)),
        "std_prob": float(np.std(prob_grid)),
        "level_spacing_mean": float(np.mean(np.diff(levels))) if len(levels) > 1 else 0.0
    }





def build_hypercube_states(n: int, dim: int) -> np.ndarray:
    if dim < 1:
        return np.zeros((1, 1))
    from itertools import product
    verts = np.array(list(product([-1, 1], repeat=dim)), dtype=float)
    return verts


def diophantine_constrained_states(a: np.ndarray, b: int,
                                   bounds: Optional[np.ndarray] = None) -> np.ndarray:
    if bounds is not None:
        sols = diophantine_nd_nonnegative_bounded(a, b, bounds)
    else:
        sols = diophantine_nd_nonnegative(a, b)
    if not sols:
        return np.zeros((0, len(a)), dtype=int)
    return np.array(sols)
