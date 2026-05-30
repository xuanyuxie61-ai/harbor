
import numpy as np
from typing import Callable, Tuple, List






def clenshaw_curtis_points(level: int) -> np.ndarray:
    if level == 0:
        return np.array([0.5], dtype=np.float64)
    n = 2**(level - 1) + 1
    j = np.arange(n)

    x = np.cos(j * np.pi / (n - 1))

    return 0.5 * (x + 1.0)


def clenshaw_curtis_weights(level: int) -> np.ndarray:
    if level == 0:
        return np.array([1.0], dtype=np.float64)
    n = 2**(level - 1) + 1
    x = clenshaw_curtis_points(level)

    w = np.zeros(n, dtype=np.float64)
    if n == 2:
        w[:] = 0.5
    else:

        w[0] = 1.0 / (n * (n - 2))
        w[-1] = w[0]
        for j in range(1, n - 1):
            if j % 2 == 0:
                w[j] = 2.0 / (1.0 - j * j)
            else:
                w[j] = 0.0

        w = w / np.sum(w) * 2.0

    return w * 0.5






def lagrange_basis_1d(xi: float, nodes: np.ndarray, k: int) -> float:
    n = len(nodes)
    if n == 1:
        return 1.0
    result = 1.0
    xk = nodes[k]
    for j in range(n):
        if j != k:
            denom = xk - nodes[j]
            if abs(denom) < 1e-14:
                continue
            result *= (xi - nodes[j]) / denom
    return result


def interpolate_1d(xi: float, nodes: np.ndarray, values: np.ndarray) -> float:
    result = 0.0
    for k in range(len(nodes)):
        result += values[k] * lagrange_basis_1d(xi, nodes, k)
    return result






def generate_multi_index_combinations(d: int, q: int) -> List[Tuple[int, ...]]:
    results = []
    min_sum = max(0, q - d + 1)
    max_sum = q

    def backtrack(pos: int, current: List[int], current_sum: int):
        if pos == d:
            if min_sum <= current_sum <= max_sum:
                results.append(tuple(current))
            return

        remaining = d - pos - 1
        for val in range(0, max_sum - current_sum + 1):

            if current_sum + val + 0 > max_sum:
                break
            if current_sum + val + remaining < min_sum:
                continue
            current.append(val)
            backtrack(pos + 1, current, current_sum + val)
            current.pop()

    backtrack(0, [], 0)
    return results


def smolyak_coefficient(d: int, q: int, i_sum: int) -> int:
    from math import comb
    s = q - i_sum
    if s < 0 or s > d - 1:
        return 0
    sign = (-1)**s
    return sign * comb(d - 1, s)


class SmolyakSparseGrid:
    def __init__(self, d: int, q: int):
        self.d = d
        self.q = q
        self.multi_indices = generate_multi_index_combinations(d, q)
        self.nodes_dict = {}
        self.values_dict = {}
        self._build_grid()

    def _build_grid(self):
        for mi in self.multi_indices:
            key = tuple(mi)
            nodes_list = []
            for dim, level in enumerate(mi):
                nodes_list.append(clenshaw_curtis_points(level))

            grids = np.meshgrid(*nodes_list, indexing='ij')
            flat_nodes = np.column_stack([g.ravel() for g in grids])
            self.nodes_dict[key] = flat_nodes

    def sample_function(self, func: Callable[[np.ndarray], float]):
        for key, nodes in self.nodes_dict.items():
            vals = np.array([func(pt) for pt in nodes], dtype=np.float64)
            self.values_dict[key] = vals

    def interpolate(self, x: np.ndarray) -> float:
        result = 0.0
        for mi in self.multi_indices:
            key = tuple(mi)
            coeff = smolyak_coefficient(self.d, self.q, sum(mi))
            if coeff == 0:
                continue
            nodes_list = []
            for dim, level in enumerate(mi):
                nodes_list.append(clenshaw_curtis_points(level))


            flat_nodes = self.nodes_dict[key]
            flat_vals = self.values_dict.get(key)
            if flat_vals is None:
                continue



            n_per_dim = [len(nl) for nl in nodes_list]

            val_sum = 0.0
            for idx_flat in range(len(flat_vals)):

                idx_multi = []
                temp = idx_flat
                for dim in range(self.d - 1, -1, -1):
                    idx_multi.append(temp % n_per_dim[dim])
                    temp //= n_per_dim[dim]
                idx_multi = idx_multi[::-1]


                basis_val = 1.0
                for dim in range(self.d):
                    basis_val *= lagrange_basis_1d(
                        x[dim], nodes_list[dim], idx_multi[dim])
                val_sum += flat_vals[idx_flat] * basis_val

            result += coeff * val_sum
        return result

    def get_total_points(self) -> int:
        return sum(len(pts) for pts in self.nodes_dict.values())






def full_tensor_product_points(d: int, n_per_dim: int) -> np.ndarray:
    x1d = np.linspace(0.0, 1.0, n_per_dim)
    grids = np.meshgrid(*([x1d] * d), indexing='ij')
    return np.column_stack([g.ravel() for g in grids])


def compare_grid_cardinality(d: int, q: int) -> Tuple[int, int]:
    sg = SmolyakSparseGrid(d, q)
    n_sparse = sg.get_total_points()

    max_level = max(sg.nodes_dict.keys(), key=lambda k: max(k))
    n_max = max(len(clenshaw_curtis_points(level)) for level in max_level)
    n_full = n_max ** d
    return n_sparse, n_full






def build_compliance_surrogate(design_sampler: Callable[[np.ndarray], float],
                                d: int, q: int = 4) -> SmolyakSparseGrid:
    sg = SmolyakSparseGrid(d, q)
    sg.sample_function(design_sampler)
    return sg


def test_function_oscillatory(x: np.ndarray) -> float:
    r = np.linalg.norm(x)
    return np.cos(2.0 * np.pi * r) * np.exp(-0.5 * r * r)
