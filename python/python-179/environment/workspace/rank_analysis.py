
import numpy as np
from typing import Tuple, List
from system_utils import EPS, TOL_RANK, check_finite






def rref_compute(A: np.ndarray, tol: float = None) -> Tuple[np.ndarray, List[int]]:
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    if tol is None:
        tol = TOL_RANK * max(m, n)
    pivot_cols = []
    r = 0
    for j in range(n):

        pivot_val = 0.0
        pivot_row = -1
        for i in range(r, m):
            if abs(A[i, j]) > pivot_val:
                pivot_val = abs(A[i, j])
                pivot_row = i
        if pivot_val <= tol:
            continue

        if pivot_row != r:
            A[[r, pivot_row], :] = A[[pivot_row, r], :]

        A[r, :] /= A[r, j]

        for i in range(m):
            if i != r and abs(A[i, j]) > tol:
                A[i, :] -= A[i, j] * A[r, :]
        pivot_cols.append(j)
        r += 1
        if r >= m:
            break
    return A, pivot_cols


def rref_rank(A: np.ndarray, tol: float = None) -> int:
    _, pivots = rref_compute(A, tol=tol)
    return len(pivots)


def rref_columns(A: np.ndarray, tol: float = None) -> np.ndarray:
    _, pivots = rref_compute(A, tol=tol)
    return A[:, pivots]






def unfold_tensor(tensor: np.ndarray, mode: int) -> np.ndarray:
    tensor = np.asarray(tensor)
    d = tensor.ndim
    if not (0 <= mode < d):
        raise ValueError(f"mode must be in [0, {d-1}]")

    perm = [mode] + [i for i in range(d) if i != mode]
    tensor_perm = np.transpose(tensor, perm)
    n_mode = tensor.shape[mode]
    rest = int(np.prod([tensor.shape[i] for i in range(d) if i != mode]))
    return tensor_perm.reshape(n_mode, rest)


def tensor_multilinear_ranks(tensor: np.ndarray, tol: float = None) -> List[int]:
    tensor = np.asarray(tensor)
    d = tensor.ndim
    ranks = []
    for k in range(d):
        A_k = unfold_tensor(tensor, k)
        ranks.append(rref_rank(A_k, tol=tol))
    return ranks


def estimate_tensor_train_ranks(tensor: np.ndarray, tol: float = None) -> List[int]:
    tensor = np.asarray(tensor)
    d = tensor.ndim
    shape = tensor.shape
    tt_ranks = [1]
    for k in range(1, d):
        left_size = int(np.prod(shape[:k]))
        right_size = int(np.prod(shape[k:]))
        unfolding = tensor.reshape(left_size, right_size)
        r = rref_rank(unfolding, tol=tol)
        tt_ranks.append(r)
    tt_ranks.append(1)
    return tt_ranks






def collatz_polynomial_next(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=int)
    p = p % 2
    if p.size == 0:
        return np.array([1], dtype=int)
    if p[0] == 0:

        if len(p) == 1:
            return np.array([0], dtype=int)
        return p[1:]
    else:

        q = np.zeros(len(p) + 1, dtype=int)
        q[1:] = p
        q[:len(p)] = (q[:len(p)] + p) % 2
        q[0] = (q[0] + 1) % 2
        return q


def collatz_polynomial_sequence(p0: np.ndarray, max_steps: int = 100) -> List[np.ndarray]:
    seq = [np.asarray(p0, dtype=int).copy()]
    p = seq[0].copy()
    for _ in range(max_steps):
        if len(p) == 1:
            break
        p = collatz_polynomial_next(p)
        seq.append(p.copy())
    return seq


def build_hankel_tensor_from_sequence(seq: List[np.ndarray],
                                       dimensions: Tuple[int, ...]) -> np.ndarray:
    d = len(dimensions)
    tensor = np.zeros(dimensions, dtype=float)
    it = np.nditer(tensor, flags=['multi_index'], op_flags=[['writeonly']])
    while not it.finished:
        idx = it.multi_index
        flat_idx = sum(idx)
        if flat_idx < len(seq):

            val = float(seq[flat_idx][0]) if seq[flat_idx].size > 0 else 0.0
        else:
            val = 0.0
        it[0] = val
        it.iternext()
    return tensor


def hankel_matrix_from_sequence(s: np.ndarray, m: int, n: int) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    H = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            idx = i + j
            H[i, j] = s[idx] if idx < len(s) else 0.0
    return H
