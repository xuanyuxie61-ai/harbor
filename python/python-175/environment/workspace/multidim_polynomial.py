
import numpy as np
from math import comb
from itertools import combinations_with_replacement


def multi_index_total_degree(alpha):
    return int(np.sum(alpha))


def multi_index_grlex_compare(alpha, beta):
    alpha = np.asarray(alpha, dtype=int)
    beta = np.asarray(beta, dtype=int)
    da = multi_index_total_degree(alpha)
    db = multi_index_total_degree(beta)
    if da < db:
        return -1
    if da > db:
        return 1

    for a, b in zip(alpha, beta):
        if a > b:
            return -1
        if a < b:
            return 1
    return 0


def multi_index_rank_grlex(alpha, d):
    alpha = np.asarray(alpha, dtype=int)
    if len(alpha) != d:
        raise ValueError("alpha length must equal dimension d")
    n = multi_index_total_degree(alpha)

    rank = comb(n + d, d) - 1

    s = n
    for k in range(d - 1):
        s -= alpha[k]
        rank -= comb(s + d - k - 1, d - k)
    return rank


def multi_index_unrank_grlex(rank, d):
    if rank < 0:
        raise ValueError("rank must be non-negative")

    n = 0
    while comb(n + d, d) - 1 <= rank:
        n += 1
    n -= 1
    rank -= comb(n + d, d) - 1
    alpha = np.zeros(d, dtype=int)
    s = n
    for k in range(d - 1):

        alpha_k = 0
        while alpha_k <= s:
            dec = comb(s - alpha_k + d - k - 1, d - k - 1)
            if dec <= rank:
                alpha_k += 1
            else:
                break
        alpha_k -= 1
        alpha[k] = alpha_k
        rank -= comb(s - alpha_k + d - k - 1, d - k - 1)
        s -= alpha_k
    alpha[-1] = s
    return alpha


def enumerate_multi_indices_grlex(d, max_degree):
    if d < 1:
        raise ValueError("dimension d must be positive")
    if max_degree < 0:
        return np.zeros((0, d), dtype=int)
    indices = []

    for deg in range(max_degree + 1):

        for dividers in combinations_with_replacement(range(deg + d - 1), d - 1):
            alpha = np.zeros(d, dtype=int)
            prev = -1
            for i, pos in enumerate(dividers):
                alpha[i] = pos - prev - 1
                prev = pos
            alpha[-1] = deg + d - 1 - 1 - prev
            indices.append(alpha.copy())

    indices.sort(key=lambda a: (multi_index_total_degree(a), tuple(-a)))
    return np.array(indices, dtype=int)


def enumerate_multi_indices_total_degree(d, max_degree):
    return enumerate_multi_indices_grlex(d, max_degree)


def sparse_grid_index_set(d, level, rule="tensor"):
    if rule == "tensor":
        grids = [np.arange(level + 1) for _ in range(d)]
        mesh = np.array(np.meshgrid(*grids, indexing='ij'))
        return mesh.reshape(d, -1).T
    elif rule == "total":
        return enumerate_multi_indices_grlex(d, level)
    elif rule == "hyperbolic":
        all_idx = enumerate_multi_indices_grlex(d, level)
        mask = np.ones(all_idx.shape[0], dtype=bool)
        for i in range(all_idx.shape[0]):
            prod = 1.0
            for k in range(d):
                prod *= (all_idx[i, k] + 1)
            if prod > level + 1:
                mask[i] = False
        return all_idx[mask]
    else:
        raise ValueError(f"Unknown rule: {rule}")


def multivariate_orthogonal_basis(alpha, xi, poly_eval_1d):
    alpha = np.asarray(alpha, dtype=int)
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
    d = len(alpha)
    if xi.shape[1] != d:
        raise ValueError("xi must have d columns")
    vals = np.ones(xi.shape[0])
    for k in range(d):
        vals *= poly_eval_1d(alpha[k], xi[:, k])
    return vals


def test_multidim_polynomial():
    d = 3
    max_deg = 3
    idx = enumerate_multi_indices_grlex(d, max_deg)

    for i, alpha in enumerate(idx):
        r = multi_index_rank_grlex(alpha, d)
        assert r == i, f"Rank mismatch at i={i}, alpha={alpha}"

    for i in range(len(idx)):
        alpha = multi_index_unrank_grlex(i, d)
        assert np.array_equal(alpha, idx[i])

    n_choose = comb(max_deg + d, d)
    assert len(idx) == n_choose
    print("multidim_polynomial: all self-tests passed")


if __name__ == "__main__":
    test_multidim_polynomial()
