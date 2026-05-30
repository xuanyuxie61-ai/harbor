import numpy as np
from combustion_utils import check_positive, check_nonnegative, check_interval


def clenshaw_curtis_nodes_1d(level):
    check_nonnegative(level, "level")
    if level == 0:
        return np.array([0.0])
    n = 2 ** level
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    return x


def piecewise_linear_basis(nodes, x_eval):
    nodes = np.asarray(nodes)
    x = float(x_eval)
    n = len(nodes)
    w = np.zeros(n)
    if x <= nodes[0]:
        w[0] = 1.0
        return w
    if x >= nodes[-1]:
        w[-1] = 1.0
        return w

    for i in range(n - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            h = nodes[i + 1] - nodes[i]
            if abs(h) < 1.0e-14:
                w[i] = 1.0
            else:
                w[i] = (nodes[i + 1] - x) / h
                w[i + 1] = (x - nodes[i]) / h
            return w
    w[-1] = 1.0
    return w


def sparse_grid_index_set(dim, max_level):
    indices = []

    def recurse(current, dim_idx, sum_level):
        if dim_idx == dim:
            if sum_level <= max_level + dim - 1:
                indices.append(current[:])
            return
        for l in range(1, max_level + 1):
            if sum_level + l > max_level + dim - 1:
                break
            current.append(l)
            recurse(current, dim_idx + 1, sum_level + l)
            current.pop()

    recurse([], 0, 0)
    return indices


class SparseGridChemistry:

    def __init__(self, max_level=3, dim=4):
        check_positive(max_level, "max_level")
        check_positive(dim, "dim")
        self.max_level = max_level
        self.dim = dim
        self.grids = [{} for _ in range(dim)]
        self.values = {}

    def build(self, rate_func):

        for di in range(self.dim):
            for li in range(0, self.max_level + 1):
                nodes = clenshaw_curtis_nodes_1d(li)
                self.grids[di][li] = nodes


        self.values = {}
        for level_vec in sparse_grid_index_set(self.dim, self.max_level):
            nodes_list = [self.grids[d][level_vec[d]] for d in range(self.dim)]

            mesh = np.meshgrid(*nodes_list, indexing='ij')
            flat_nodes = np.stack([m.ravel() for m in mesh], axis=1)
            for pt in flat_nodes:
                key = tuple(np.round(pt, decimals=12))
                if key not in self.values:
                    self.values[key] = rate_func(pt)

    def evaluate(self, y_point):
        y_point = np.asarray(y_point, dtype=float)
        if y_point.shape[0] != self.dim:
            raise ValueError(f"Point dimension {y_point.shape[0]} != grid dimension {self.dim}")
        y_point = np.clip(y_point, -1.0, 1.0)


        result = 0.0
        count = 0
        for level_vec in sparse_grid_index_set(self.dim, self.max_level):
            nodes_list = [self.grids[d][level_vec[d]] for d in range(self.dim)]
            mesh = np.meshgrid(*nodes_list, indexing='ij')
            flat_nodes = np.stack([m.ravel() for m in mesh], axis=1)
            flat_vals = np.array([self.values.get(tuple(np.round(pt, decimals=12)), 0.0) for pt in flat_nodes])


            interp_val = flat_vals.reshape([len(nl) for nl in nodes_list])
            for d in range(self.dim):
                nodes_d = nodes_list[d]
                w = piecewise_linear_basis(nodes_d, y_point[d])

                shape = list(interp_val.shape)
                shape[d] = 1
                new_val = np.zeros(shape)
                for i in range(len(nodes_d)):
                    slc = [slice(None)] * self.dim
                    slc[d] = i
                    new_val += w[i] * interp_val[tuple(slc)]
                interp_val = new_val
            result += interp_val.flat[0]
            count += 1

        if count > 0:
            result /= count
        return result

    def evaluate_batch(self, points):
        points = np.asarray(points, dtype=float)
        return np.array([self.evaluate(p) for p in points])
