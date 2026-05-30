
import numpy as np


def r8pp_fa(n: int, a: np.ndarray) -> tuple:
    if a.size != n * (n + 1) // 2:
        raise ValueError("Packed array length mismatch.")

    r = a.copy()
    info = 0
    jj = 0

    for j in range(1, n + 1):
        s = 0.0
        kj = jj
        kk = 0
        for k in range(1, j):
            kj += 1
            t = r[kj - 1]
            for i in range(1, k):
                t -= r[kk + i - 1] * r[jj + i - 1]
            kk += k
            if abs(r[kk - 1]) < 1e-15:
                info = j
                return r, info
            t = t / r[kk - 1]
            r[kj - 1] = t
            s += t * t

        jj += j
        s = r[jj - 1] - s
        if s <= 0.0:
            info = j
            return r, info
        r[jj - 1] = np.sqrt(s)

    return r, info


def r8pp_sl(n: int, r: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = b.copy().astype(float)


    kk = 0
    for k in range(1, n + 1):
        x[k - 1] = (x[k - 1] - np.dot(r[kk:kk + k - 1], x[:k - 1])) / r[kk + k - 1]
        kk += k


    for k in range(n, 0, -1):
        kk = k * (k - 1) // 2
        x[k - 1] = x[k - 1] / r[kk + k - 1]
        for i in range(1, k):
            x[i - 1] -= r[kk + i - 1] * x[k - 1]

    return x


def dense_to_packed(a_dense: np.ndarray) -> np.ndarray:
    n = a_dense.shape[0]
    a = np.zeros(n * (n + 1) // 2)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            a[idx] = a_dense[i, j]
            idx += 1
    return a


def packed_to_dense(n: int, a: np.ndarray) -> np.ndarray:
    a_dense = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            a_dense[i, j] = a[idx]
            if i != j:
                a_dense[j, i] = a[idx]
            idx += 1
    return a_dense


def spd_sample(n: int, cov_packed: np.ndarray) -> np.ndarray:
    r, info = r8pp_fa(n, cov_packed)
    if info != 0:
        raise RuntimeError(f"Cholesky factorization failed at step {info}.")
    z = np.random.randn(n)
    x = np.zeros(n)

    kk = 0
    for k in range(1, n + 1):
        x[k - 1] = np.dot(r[kk:kk + k], z[:k])
        kk += k
    return x
