
import numpy as np
from typing import Tuple, List, Callable, Optional
from itertools import product


class ClenshawCurtisRule:

    def __init__(self):
        self.rules_cache = {}

    def get_rule(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        if level in self.rules_cache:
            return self.rules_cache[level]

        if level == 0:
            x = np.array([0.5])
            w = np.array([1.0])
        else:
            n = 2 ** level + 1

            x_cheb = np.cos(np.pi * np.arange(n) / (n - 1))
            x = 0.5 * (x_cheb + 1.0)


            w = self._cc_weights(n)
            w = 0.5 * w

        self.rules_cache[level] = (x, w)
        return x, w

    def _cc_weights(self, n: int) -> np.ndarray:
        if n == 1:
            return np.array([2.0])

        w = np.zeros(n)
        theta = np.pi * np.arange(n) / (n - 1)


        for j in range(n):
            if j == 0 or j == n - 1:
                c = 1.0
            else:
                c = 2.0

            val = 0.0
            for k in range(0, n // 2):
                if 2 * k == n - 1:
                    continue
                b = 1.0
                if k == 0:
                    b = 1.0
                else:
                    b = 2.0
                val += b / (4.0 * k * k - 1.0) * np.cos(2.0 * k * theta[j])
            w[j] = c * val / (n - 1)


        w = w / np.sum(w) * 2.0
        return w

    def get_nested_points(self, level: int) -> np.ndarray:
        if level == 0:
            return np.array([0.5])

        x_all, _ = self.get_rule(level)
        x_prev, _ = self.get_rule(level - 1)


        new_points = []
        for x in x_all:
            if not any(abs(x - xp) < 1e-14 for xp in x_prev):
                new_points.append(x)
        return np.array(new_points)


class SparseGridGenerator:

    def __init__(self, dim: int, level_max: int):
        if dim < 1 or level_max < 0:
            raise ValueError("dim >= 1 且 level_max >= 0")

        self.dim = dim
        self.level_max = level_max
        self.cc = ClenshawCurtisRule()

    def compute_size(self) -> int:
        q = self.level_max + self.dim
        count = 0

        for l_vec in product(range(self.level_max + 1), repeat=self.dim):
            l_sum = sum(l_vec)
            if q - self.dim <= l_sum <= q - self.dim:
                count += 1



        if self.dim == 1:
            return 2 ** self.level_max + 1
        else:

            return int((self.level_max + 1) ** self.dim * 0.5)

    def generate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.level_max + self.dim
        points_dict = {}


        for l_vec in self._multi_indices(q):
            l_sum = sum(l_vec)
            coeff = (-1) ** (q - l_sum) * self._n_choose_k(self.dim - 1, q - l_sum)

            if coeff == 0:
                continue


            rules_1d = []
            for d in range(self.dim):
                x_d, w_d = self.cc.get_rule(l_vec[d])
                rules_1d.append((x_d, w_d))


            indices = [range(len(r[0])) for r in rules_1d]
            for idx in product(*indices):
                point = np.array([rules_1d[d][0][idx[d]] for d in range(self.dim)])
                weight = coeff * np.prod([rules_1d[d][1][idx[d]] for d in range(self.dim)])

                key = tuple(np.round(point, 14))
                if key in points_dict:
                    points_dict[key] += weight
                else:
                    points_dict[key] = weight

        points = np.array([np.array(k) for k in points_dict.keys()])
        weights = np.array(list(points_dict.values()))


        if len(weights) > 0:
            weights = weights / np.sum(weights)

        return points, weights

    def _multi_indices(self, q: int):
        from itertools import combinations_with_replacement


        def generate_indices(dim, max_sum, current=[]):
            if dim == 0:
                if sum(current) <= max_sum:
                    yield tuple(current)
                return
            for i in range(max_sum + 1):
                if sum(current) + i <= max_sum:
                    yield from generate_indices(dim - 1, max_sum, current + [i])

        yield from generate_indices(self.dim, q)

    def _n_choose_k(self, n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result


class OzoneModelUQ:

    def __init__(self, dim: int = 5, level_max: int = 3):
        self.dim = dim
        self.level_max = level_max
        self.grid_gen = SparseGridGenerator(dim, level_max)







        self.param_names = ['O3_column', 'A_factor', 'Ea_factor',
                            'Kzz_factor', 'J_factor']

    def map_parameters(self, xi: np.ndarray) -> dict:
        params = {}

        params['A_factor'] = 0.5 + xi[0]
        params['Ea_factor'] = 0.8 + 0.4 * xi[1]
        params['Kzz_factor'] = 0.2 + 2.0 * xi[2]
        params['J_factor'] = 0.7 + 0.6 * xi[3]
        params['temp_offset'] = -5.0 + 10.0 * xi[4]
        return params

    def model_response(self, xi: np.ndarray,
                       base_o3_column: float = 300.0) -> float:
        params = self.map_parameters(xi)



        o3 = base_o3_column
        o3 *= params['A_factor'] ** 0.3
        o3 *= np.exp(-0.5 * (params['Ea_factor'] - 1.0) ** 2 / 0.04)
        o3 *= params['Kzz_factor'] ** (-0.2)
        o3 *= params['J_factor'] ** 0.4
        o3 += 2.0 * params['temp_offset']


        interaction = 10.0 * (xi[0] - 0.5) * (xi[2] - 0.5)
        o3 += interaction

        return np.clip(o3, 100.0, 600.0)

    def compute_statistics(self) -> dict:
        points, weights = self.grid_gen.generate_grid()

        if len(points) == 0:
            return {}

        values = np.array([self.model_response(p) for p in points])


        mean = np.sum(weights * values)


        variance = np.sum(weights * (values - mean) ** 2)
        std = np.sqrt(max(variance, 0.0))


        sorted_indices = np.argsort(values)
        cum_weights = np.cumsum(weights[sorted_indices])

        def quantile(q: float) -> float:
            idx = np.searchsorted(cum_weights, q)
            idx = min(idx, len(values) - 1)
            return values[sorted_indices[idx]]

        q05 = quantile(0.05)
        q25 = quantile(0.25)
        q50 = quantile(0.50)
        q75 = quantile(0.75)
        q95 = quantile(0.95)

        return {
            'n_points': len(points),
            'mean': mean,
            'variance': variance,
            'std': std,
            'q05': q05,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'q95': q95,
            'points': points,
            'weights': weights,
            'values': values
        }

    def sobol_first_order(self, n_monte_carlo: int = 5000) -> dict:
        np.random.seed(42)


        A = np.random.rand(n_monte_carlo, self.dim)
        B = np.random.rand(n_monte_carlo, self.dim)

        f_A = np.array([self.model_response(a) for a in A])
        f_B = np.array([self.model_response(b) for b in B])

        total_var = np.var(np.concatenate([f_A, f_B]))
        if total_var < 1e-20:
            return {f'X{i}': 0.0 for i in range(self.dim)}

        sobol = {}
        for i in range(self.dim):
            A_B = A.copy()
            A_B[:, i] = B[:, i]
            f_AB = np.array([self.model_response(ab) for ab in A_B])


            S_i = np.mean(f_B * (f_AB - f_A)) / (total_var + 1e-30)
            sobol[f'X{i}_{self.param_names[i]}'] = np.clip(S_i, 0.0, 1.0)

        return sobol
