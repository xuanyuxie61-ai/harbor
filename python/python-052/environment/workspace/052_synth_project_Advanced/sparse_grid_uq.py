
import numpy as np
from itertools import product

class SparseGridUQ:

    def __init__(self, dim, level):
        self.dim = dim
        self.level = level

    @staticmethod
    def clenshaw_curtis_nested(level):
        if level == 0:
            return np.array([0.0]), np.array([2.0])
        n = 2**(level - 1) + 1
        j = np.arange(n)
        x = np.cos(j * np.pi / (n - 1))

        w = np.ones(n) * 2.0 / (n - 1)
        w[0] *= 0.5
        w[-1] *= 0.5
        return x, w

    def _multi_index_set(self):
        q = self.level
        d = self.dim
        indices = []

        def recurse(current, remaining):
            if len(current) == d:
                if sum(current) <= q + d - 1:
                    indices.append(tuple(current))
                return

            min_val = 1
            max_val = remaining - (d - len(current) - 1)
            for val in range(min_val, max_val + 1):
                recurse(current + [val], remaining - val)
        for total in range(d, q + d):
            recurse([], total)
        return indices

    def build_sparse_grid(self):
        indices = self._multi_index_set()
        node_list = []
        weight_list = []

        for idx in indices:

            rules = [self.clenshaw_curtis_nested(i) for i in idx]

            x_grids = [r[0] for r in rules]
            w_grids = [r[1] for r in rules]
            for coords in product(*[range(len(xg)) for xg in x_grids]):
                node = np.array([x_grids[d][coords[d]] for d in range(self.dim)])
                weight = np.prod([w_grids[d][coords[d]] for d in range(self.dim)])
                node_list.append(node)
                weight_list.append(weight)

        if not node_list:
            return np.zeros((1, self.dim)), np.ones(1)

        nodes = np.array(node_list)
        weights = np.array(weight_list)



        rounded = np.round(nodes, decimals=12)
        unique, inverse = np.unique(rounded.view(rounded.dtype.descr * self.dim),
                                     return_inverse=True)
        unique = unique.view(rounded.dtype).reshape(-1, self.dim)
        new_weights = np.zeros(len(unique))
        for i in range(len(weights)):
            new_weights[inverse[i]] += weights[i]

        return unique, new_weights

    def propagate_expectation(self, model_evaluator):
        nodes, weights = self.build_sparse_grid()
        values = np.array([model_evaluator(node) for node in nodes])



        total_weight = np.sum(weights)
        if abs(total_weight - 2**self.dim) > 1e-6:
            weights = weights / total_weight * (2**self.dim)

        mean = np.sum(weights * values) / np.sum(weights)
        variance = np.sum(weights * (values - mean)**2) / np.sum(weights)
        return mean, variance, nodes, values


def total_degree_size(dim, degree):
    from math import comb
    return comb(degree + dim, dim)


def comp_next(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for tail in comp_next(n - i, k - 1):
            yield (i,) + tail
