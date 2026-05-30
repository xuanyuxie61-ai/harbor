
import numpy as np


def matrix_chain_optimal_order(dims):
    dims = np.asarray(dims, dtype=int)
    n = len(dims) - 1
    if n < 2:
        return 0, np.zeros((max(n, 1), max(n, 1)), dtype=int)
    if np.any(dims <= 0):
        raise ValueError("All dimensions must be positive.")

    m = np.full((n, n), np.inf, dtype=float)
    s = np.zeros((n, n), dtype=int)

    for i in range(n):
        m[i, i] = 0.0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i, j]:
                    m[i, j] = cost
                    s[i, j] = k

    return int(m[0, n - 1]), s


def print_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        left = print_optimal_parens(s, i, s[i, j])
        right = print_optimal_parens(s, s[i, j] + 1, j)
        return f"({left} x {right})"


class TensorChainOptimizer:

    def __init__(self, tensor_dims):
        self.tensor_dims = tensor_dims

    def optimize_einsum_chain(self, contractions):
        n = len(self.tensor_dims)


        dims = [1] * (n + 1)
        for i, td in enumerate(self.tensor_dims):
            if len(td) >= 2:
                dims[i] = td[0]
                dims[i + 1] = td[-1]
            else:
                dims[i] = td[0]
                dims[i + 1] = 1

        cost, s = matrix_chain_optimal_order(dims)
        order_str = print_optimal_parens(s, 0, n - 1) if n > 0 else ""
        return order_str, cost


def apply_optimal_matrix_chain(matrices, s, i=None, j=None):
    n = len(matrices)
    if n == 0:
        raise ValueError("Empty matrix chain.")
    if n == 1:
        return matrices[0]
    if i is None:
        i = 0
    if j is None:
        j = n - 1

    if i == j:
        return matrices[i]

    k = s[i, j]
    left = apply_optimal_matrix_chain(matrices, s, i, k)
    right = apply_optimal_matrix_chain(matrices, s, k + 1, j)
    return left @ right


class AcousticOperatorChain:

    def __init__(self, operators):
        self.operators = operators
        self.dims = [op.shape[0] for op in operators]
        self.dims.append(operators[-1].shape[1])
        self._optimal_s = None
        self._optimal_cost = None

    def optimize(self):
        cost, s = matrix_chain_optimal_order(self.dims)
        self._optimal_cost = cost
        self._optimal_s = s
        return cost

    def apply(self, vector):
        if self._optimal_s is None:
            self.optimize()



        result = vector.copy()
        for op in reversed(self.operators):
            if result.ndim == 1:
                result = op @ result
            else:
                result = op @ result
        return result

    def flops_estimate(self):
        if self._optimal_cost is None:
            self.optimize()
        return self._optimal_cost
