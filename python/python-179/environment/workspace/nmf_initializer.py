
import numpy as np
from typing import Tuple
from system_utils import EPS, clip_to_range






def random_contingency_table(nrow: int, ncol: int,
                              nrowt: np.ndarray, ncolt: np.ndarray,
                              seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    nrowt = np.asarray(nrowt, dtype=int)
    ncolt = np.asarray(ncolt, dtype=int)
    if nrowt.sum() != ncolt.sum():
        raise ValueError("Row sums and column sums must be equal.")
    table = np.zeros((nrow, ncol), dtype=int)
    row_rem = nrowt.copy()
    col_rem = ncolt.copy()
    total = int(row_rem.sum())
    for i in range(nrow):
        for j in range(ncol):
            if row_rem[i] == 0 or col_rem[j] == 0:
                continue

            rem_total = int(row_rem[i:].sum())
            if rem_total == 0:
                break

            p = col_rem[j] / max(col_rem[j:].sum(), 1)
            max_val = min(row_rem[i], col_rem[j])

            val = np.random.binomial(row_rem[i], p)
            val = min(val, max_val)
            val = max(val, 0)
            table[i, j] = val
            row_rem[i] -= val
            col_rem[j] -= val
    return table






def nmf_init_random(m: int, n: int, rank: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(m, rank)
    H = np.random.rand(rank, n)
    return W, H


def nmf_init_coltable(m: int, n: int, rank: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    row_sum_W = np.ones(rank, dtype=int) * max(m // rank + 1, 2)
    col_sum_W = np.random.randint(2, 6, size=m)
    total_W = max(row_sum_W.sum(), col_sum_W.sum())

    row_sum_W = np.round(row_sum_W * total_W / row_sum_W.sum()).astype(int)
    col_sum_W = np.round(col_sum_W * total_W / col_sum_W.sum()).astype(int)

    diff = row_sum_W.sum() - col_sum_W.sum()
    idx = 0
    while diff != 0:
        if diff > 0:
            if col_sum_W[idx % m] + diff > 0:
                col_sum_W[idx % m] += diff
                diff = 0
            else:
                col_sum_W[idx % m] += 1
                diff -= 1
        else:
            if row_sum_W[idx % rank] - diff > 0:
                row_sum_W[idx % rank] -= diff
                diff = 0
            else:
                row_sum_W[idx % rank] += 1
                diff += 1
        idx += 1
    W_int = random_contingency_table(rank, m, row_sum_W, col_sum_W, seed=seed)
    W = W_int.astype(float) + EPS
    W = W / (W.sum(axis=1, keepdims=True) + EPS)

    row_sum_H = np.ones(rank, dtype=int) * max(n // rank + 1, 2)
    col_sum_H = np.random.randint(2, 6, size=n)
    total_H = max(row_sum_H.sum(), col_sum_H.sum())
    row_sum_H = np.round(row_sum_H * total_H / row_sum_H.sum()).astype(int)
    col_sum_H = np.round(col_sum_H * total_H / col_sum_H.sum()).astype(int)
    diff = row_sum_H.sum() - col_sum_H.sum()
    idx = 0
    while diff != 0:
        if diff > 0:
            if col_sum_H[idx % n] + diff > 0:
                col_sum_H[idx % n] += diff
                diff = 0
            else:
                col_sum_H[idx % n] += 1
                diff -= 1
        else:
            if row_sum_H[idx % rank] - diff > 0:
                row_sum_H[idx % rank] -= diff
                diff = 0
            else:
                row_sum_H[idx % rank] += 1
                diff += 1
        idx += 1
    H_int = random_contingency_table(rank, n, row_sum_H, col_sum_H, seed=seed + 1 if seed else None)
    H = H_int.astype(float).T + EPS
    H = H / (H.sum(axis=0, keepdims=True) + EPS)
    return W, H


def ntf_init_random(shape: Tuple[int, ...], ranks: Tuple[int, ...],
                    seed: int = None) -> list:
    if seed is not None:
        np.random.seed(seed)
    d = len(shape)
    if len(ranks) != d:
        raise ValueError("ranks length must match tensor order.")
    factors = []
    for k in range(d):
        F = np.random.rand(shape[k], ranks[k]) + EPS

        col_norms = np.linalg.norm(F, axis=0)
        F = F / (col_norms + EPS)
        factors.append(F)
    return factors






def nonnegative_projection(X: np.ndarray) -> np.ndarray:
    return np.maximum(X, 0.0)


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    tau = max(tau, 0.0)
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)
