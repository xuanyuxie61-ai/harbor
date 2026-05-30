
import numpy as np
from numpy.polynomial.legendre import leggauss
from itertools import combinations


class SparseGridGL:

    def __init__(self, dim, level_max):
        self.dim = int(dim)
        self.level_max = int(level_max)
        self.level_min = max(0, level_max + 1 - dim)

    def _level_to_order(self, level):
        if level == 0:
            return 1
        return 2 ** level - 1

    def _generate_compositions(self, n, k):
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in self._generate_compositions(n - i, k - 1):
                yield (i,) + rest

    def build_grid(self):
        point_dict = {}

        for level in range(self.level_min, self.level_max + 1):

            for level_1d in self._generate_compositions(level, self.dim):

                coeff = (-1) ** (self.level_max - level)
                from math import comb
                coeff *= comb(self.dim - 1, self.level_max - level)


                orders = [self._level_to_order(l) for l in level_1d]

                grids_1d = []
                weights_1d = []
                for o in orders:
                    if o == 1:
                        x = np.array([0.0])
                        w = np.array([2.0])
                    else:
                        x, w = leggauss(o)
                    grids_1d.append(x)
                    weights_1d.append(w)



                indices = [np.arange(len(g)) for g in grids_1d]
                import itertools
                for idx in itertools.product(*indices):
                    point = np.array([grids_1d[d][idx[d]] for d in range(self.dim)])
                    weight = np.prod([weights_1d[d][idx[d]] for d in range(self.dim)])
                    total_weight = coeff * weight

                    key = tuple(np.round(point, decimals=12))
                    if key in point_dict:
                        point_dict[key] += total_weight
                    else:
                        point_dict[key] = total_weight

        points = np.array([np.array(k) for k in point_dict.keys()])
        weights = np.array([point_dict[k] for k in point_dict.keys()])
        return points, weights

    def integrate(self, func, domain_bounds):
        points, weights = self.build_grid()

        scaled_points = np.zeros_like(points)
        scale_factors = []
        for d in range(self.dim):
            a, b = domain_bounds[d]
            scaled_points[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (b + a)
            scale_factors.append(0.5 * (b - a))
        jacobian = np.prod(scale_factors)

        f_vals = func(scaled_points)
        return jacobian * np.sum(weights * f_vals)


class AlloyPhaseSpaceSampler:

    def __init__(self, param_names, param_bounds, level_max=3):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.dim = len(param_names)
        self.grid = SparseGridGL(self.dim, level_max)

    def sample(self, callback=None):
        points, weights = self.grid.build_grid()
        scaled = np.zeros_like(points)
        for d in range(self.dim):
            a, b = self.param_bounds[d]
            scaled[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (b + a)

        if callback is not None:
            for i in range(scaled.shape[0]):
                params = {self.param_names[d]: scaled[i, d] for d in range(self.dim)}
                callback(params)
        return scaled, weights

    def compute_expectation(self, func_values, domain_bounds):
        _, weights = self.grid.build_grid()
        jacobian = 1.0
        for d in range(self.dim):
            a, b = domain_bounds[d]
            jacobian *= 0.5 * (b - a)
        total_weight = jacobian * np.sum(weights)
        return jacobian * np.sum(weights * func_values) / (total_weight + 1e-15)
