
import numpy as np
import math





def sphere01_sample(n):
    x = np.random.randn(3, n)
    norm = np.sqrt(np.sum(x ** 2, axis=0))
    x = x / norm
    return x


def sphere01_monomial_integral(e):
    e = np.array(e)
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    if np.all(e == 0):
        return 2.0 * np.sqrt(np.pi ** 3) / np.math.gamma(1.5)
    if np.any(e % 2 == 1):
        return 0.0
    integral = 2.0
    for ei in e:
        integral *= math.gamma(0.5 * (ei + 1))
    integral /= math.gamma(0.5 * np.sum(e + 1))
    return integral


def spherical_mean_integrand(func, n_samples=10000):
    pts = sphere01_sample(n_samples)
    vals = np.array([func(pts[0, i], pts[1, i], pts[2, i]) for i in range(n_samples)])
    return np.mean(vals), np.std(vals) / np.sqrt(n_samples)





def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1 or n == 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def fibonacci_lattice_2d(k, func):
    if k < 3:
        raise ValueError("k 必须 >= 3")
    m = fibonacci(k)
    n = fibonacci(k - 1)

    quad = 0.0
    for j in range(m):
        x = (j * n % m) / m
        y = j / m
        quad += func(np.array([x, y]))
    quad /= m
    return quad


def lattice_rule_nd(dim_num, m, z, func):
    quad = 0.0
    for j in range(m):
        x = (j * z) % m / m
        quad += func(x)
    quad /= m
    return quad





def cc_abscissa(order, idx):
    if order == 1:
        return 0.0
    if idx < 0 or idx >= order:
        raise IndexError("CC 节点索引越界")
    return np.cos(np.pi * (order - 1 - idx) / (order - 1))


def cc_weights(order):
    if order == 1:
        return np.array([2.0])
    n = order - 1
    theta = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = 1.0 / (n ** 2 - 1)
        w[n] = w[0]
        for k in range(1, n // 2):
            v = v - 2.0 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
        v = v - np.cos(n * theta[1:-1]) / (n ** 2 - 1)
    else:
        w[0] = 1.0 / n ** 2
        w[n] = w[0]
        for k in range(1, (n + 1) // 2):
            v = v - 2.0 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
    w[1:-1] = 2.0 * v / n
    return w


def sparse_grid_cc_1d(level):
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    order = 2 ** level + 1
    nodes = np.array([cc_abscissa(order, i) for i in range(order)])
    weights = cc_weights(order)
    return nodes, weights


def tensor_product_grid(nodes_list, weights_list):
    dims = len(nodes_list)
    if dims == 1:
        return nodes_list[0].reshape(-1, 1), weights_list[0]


    nodes_prev, weights_prev = tensor_product_grid(nodes_list[:-1], weights_list[:-1])
    nodes_last = nodes_list[-1]
    weights_last = weights_list[-1]

    n_prev = nodes_prev.shape[0]
    n_last = len(nodes_last)
    nodes = np.zeros((n_prev * n_last, dims))
    weights = np.zeros(n_prev * n_last)
    idx = 0
    for i in range(n_prev):
        for j in range(n_last):
            nodes[idx, :-1] = nodes_prev[i]
            nodes[idx, -1] = nodes_last[j]
            weights[idx] = weights_prev[i] * weights_last[j]
            idx += 1
    return nodes, weights


def sparse_grid_cc_smolyak(dim_num, level_max):
    from combinatorial_stats import combination_lex_index


    grids = []
    for total in range(level_max, level_max + dim_num):

        def generate(dim, remain, current):
            if dim == 1:
                yield current + [remain]
            else:
                for v in range(0, remain + 1):
                    yield from generate(dim - 1, remain - v, current + [v])

        for levels in generate(dim_num, total, []):

            if all(l >= 0 for l in levels):

                coeff = ((-1) ** (level_max + dim_num - 1 - total)) * math.comb(dim_num - 1, total - level_max)
                nodes_list = []
                weights_list = []
                for l in levels:
                    n, w = sparse_grid_cc_1d(l)
                    nodes_list.append(n)
                    weights_list.append(w)
                nodes, weights = tensor_product_grid(nodes_list, weights_list)
                grids.append((nodes, coeff * weights))


    all_nodes = []
    all_weights = []
    for nodes, weights in grids:
        all_nodes.append(nodes)
        all_weights.append(weights)

    if len(all_nodes) == 0:
        return np.zeros((0, dim_num)), np.zeros(0)

    all_nodes = np.vstack(all_nodes)
    all_weights = np.concatenate(all_weights)


    rounded = np.round(all_nodes, 12)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    merged_weights = np.zeros(unique.shape[0])
    for i, w in enumerate(all_weights):
        merged_weights[inverse[i]] += w


    return unique, merged_weights


def integrate_sparse_grid(func, dim_num, level_max):
    nodes, weights = sparse_grid_cc_smolyak(dim_num, level_max)
    total = 0.0
    for i in range(nodes.shape[0]):
        total += weights[i] * func(nodes[i])
    return total
