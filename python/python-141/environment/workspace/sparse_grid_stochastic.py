
import numpy as np
from math import comb


def sandia_sgmgg_coef_naive(dim_num, point_num, sparse_index):
    sparse_index = np.asarray(sparse_index, dtype=np.int64)
    coef = np.zeros(point_num, dtype=np.int64)
    for j1 in range(point_num):
        for j2 in range(point_num):
            neighbor = True
            term = 1
            for i in range(dim_num):
                dif = sparse_index[i, j2] - sparse_index[i, j1]
                if dif == 0:
                    pass
                elif dif == 1:
                    term = -term
                else:
                    neighbor = False
                    break
            if neighbor:
                coef[j1] += term
    return coef


def sandia_sgmgg_coef_inc2(m, n1, s1, c1, s2):
    s1 = np.asarray(s1, dtype=np.int64)
    c1 = np.asarray(c1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)
    c3 = np.zeros(n1 + 1, dtype=np.int64)
    c3[:n1] = c1.copy()
    c3[n1] = 1


    n4 = 0
    c4 = np.zeros(n1, dtype=np.int64)
    s4 = np.zeros((m, n1), dtype=np.int64)

    for j in range(n1):
        s_min = np.minimum(s1[:, j], s2)
        k = -1

        for j2 in range(n1):
            if np.array_equal(s1[:, j2], s_min):
                k = j2
                break
        if k >= 0:
            c3[k] -= c1[j]
        else:

            found = False
            for j2 in range(n4):
                if np.array_equal(s4[:, j2], s_min):
                    c4[j2] -= c1[j]
                    found = True
                    break
            if not found:
                s4[:, n4] = s_min
                c4[n4] = -c1[j]
                n4 += 1


    if np.any(c4[:n4] != 0):
        raise RuntimeError("增量系数计算出错：非活跃索引残留非零系数")

    return c3


def generate_sparse_grid_indices(dim_num, level):
    if dim_num <= 0 or level < 0:
        raise ValueError("dim_num>0, level>=0")

    max_sum = dim_num + level
    indices = []

    def recurse(dim, current_sum, current_idx):
        if dim == dim_num:
            if current_sum <= max_sum:
                indices.append(current_idx.copy())
            return
        remaining_dims = dim_num - dim - 1
        min_val = 1

        max_val = max_sum - current_sum - remaining_dims
        max_val = max(max_val, min_val)
        for v in range(min_val, max_val + 1):
            current_idx[dim] = v
            recurse(dim + 1, current_sum + v, current_idx)

    recurse(0, 0, np.zeros(dim_num, dtype=np.int64))
    if not indices:
        return np.zeros((dim_num, 0), dtype=np.int64)
    return np.array(indices, dtype=np.int64).T


def clenshaw_curtis_nodes_weights(level):
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** level + 1

    j = np.arange(n)
    x = -np.cos(np.pi * j / (n - 1))

    w = np.zeros(n)
    if n == 1:
        w[0] = 2.0
        return x, w

    theta = np.pi * j / (n - 1)
    for i in range(n):
        wi = 1.0
        for k in range(1, (n - 1) // 2 + 1):
            b = 2.0 if (2 * k == n - 1) else 1.0
            wi -= b * np.cos(2.0 * k * theta[i]) / (4.0 * k * k - 1.0)
        w[i] = wi * 2.0 / (n - 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return x, w


class SparseGridIntegrator:

    def __init__(self, dim_num, level):
        self.dim_num = dim_num
        self.level = level
        self.indices = generate_sparse_grid_indices(dim_num, level)
        self.coef = sandia_sgmgg_coef_naive(dim_num, self.indices.shape[1], self.indices)

        max_level_per_dim = np.max(self.indices) if self.indices.size > 0 else 1
        self._1d_nodes = {}
        self._1d_weights = {}
        for lvl in range(max_level_per_dim + 1):
            x, w = clenshaw_curtis_nodes_weights(lvl)
            self._1d_nodes[lvl] = x
            self._1d_weights[lvl] = w

    def integrate(self, func, domain=None):
        if domain is None:
            domain = [(-1.0, 1.0)] * self.dim_num
        if len(domain) != self.dim_num:
            raise ValueError("domain维度与dim_num不匹配")

        total = 0.0
        n_points = self.indices.shape[1]
        for p in range(n_points):
            idx = self.indices[:, p]
            c = self.coef[p]
            if c == 0:
                continue

            nodes_list = [self._1d_nodes[int(idx[d])] for d in range(self.dim_num)]
            weights_list = [self._1d_weights[int(idx[d])] for d in range(self.dim_num)]

            def tensor_product_recurse(dim, current_x, current_w):
                nonlocal total
                if dim == self.dim_num:

                    x_transformed = np.zeros(self.dim_num, dtype=np.float64)
                    jac = 1.0
                    for d in range(self.dim_num):
                        a, b = domain[d]
                        x_transformed[d] = (current_x[d] + 1.0) * 0.5 * (b - a) + a
                        jac *= 0.5 * (b - a)
                    total += c * current_w * func(x_transformed) * jac
                    return
                for xi, wi in zip(nodes_list[dim], weights_list[dim]):
                    current_x[dim] = xi
                    tensor_product_recurse(dim + 1, current_x, current_w * wi)

            tensor_product_recurse(0, np.zeros(self.dim_num), 1.0)

        return total

    def get_total_points(self):
        n_points = self.indices.shape[1]
        count = 0
        for p in range(n_points):
            if self.coef[p] == 0:
                continue
            idx = self.indices[:, p]
            prod = 1
            for d in range(self.dim_num):
                prod *= len(self._1d_nodes[int(idx[d])])
            count += prod
        return count


def sparse_grid_expectation_heston(dim_num, level, payoff_func, params):
    S0 = params['S0']
    v0 = params['v0']
    r = params['r']
    T = params['T']



    logS_std = np.sqrt(v0 * T)
    domain = [
        (np.log(S0) - 3*logS_std, np.log(S0) + 3*logS_std),
        (max(v0 * 0.1, 1e-4), v0 * 3.0)
    ]
    if dim_num > 2:
        for _ in range(dim_num - 2):
            domain.append((-3.0, 3.0))

    sg = SparseGridIntegrator(dim_num, level)

    def integrand(x):
        S = np.exp(x[0])
        v = max(x[1], 1e-8)
        return payoff_func(S, v)

    expectation = sg.integrate(integrand, domain)
    return np.exp(-r * T) * expectation
