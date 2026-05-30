# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Tuple, Optional




def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("LU分解仅对方阵有效")
    p = np.arange(n, dtype=int)

    for k in range(n - 1):

        piv = np.argmax(np.abs(A[k:n, k])) + k
        if A[piv, k] == 0.0:
            raise ValueError("矩阵奇异，无法LU分解")
        if piv != k:
            A[[k, piv], :] = A[[piv, k], :]
            p[[k, piv]] = p[[piv, k]]


        rows = k + 1
        A[rows:n, k] /= A[k, k]

        A[rows:n, rows:n] -= np.outer(A[rows:n, k], A[k, rows:n])

    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)

    P = np.zeros((n, n), dtype=np.float64)
    P[p, np.arange(n)] = 1.0
    return P, L, U


def solve_lu(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    Pb = P @ b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError("系数矩阵必须是方阵")
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if A.shape[0] != b.shape[0]:
        raise ValueError("A与b维度不匹配")


    if np.allclose(A, A.T, atol=1e-12) and np.all(np.linalg.eigvalsh(A) > 0):
        L = np.linalg.cholesky(A)
        n = A.shape[0]
        y = np.zeros((n, b.shape[1]), dtype=np.float64)
        for i in range(n):
            y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
        x = np.zeros_like(y)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - L[i + 1:, i].reshape(1, -1) @ x[i + 1:]) / L[i, i]
        return x.squeeze()

    P, L, U = lu_decomposition(A)
    return solve_lu(P, L, U, b).squeeze()





def tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(b)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if n < 2:
        raise ValueError("三对角系统维度至少为2")
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-300:
            denom = np.sign(denom) * 1e-300 if denom != 0 else 1e-300
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.zeros(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x





def brent_root(f: Callable[[float], float], a: float, b: float,
               tol: float = 1e-12, max_iter: int = 100) -> float:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Brent求根要求区间端点函数值异号")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = a, fa
    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * tol * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or fb == 0.0:
            return b
        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:

                p = 2.0 * m * s
                q = 1.0 - s
            else:

                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < (min1 if min1 < min2 else min2):
                d, e = p / q, d
            else:
                d = e = m
        a, fa = b, fb
        if abs(d) > tol_act:
            b += d
        else:
            b += tol_act if m > 0 else -tol_act
        fb = f(b)
    return b





def adaptive_simpson(f: Callable[[float], float], a: float, b: float,
                     tol: float = 1e-10, max_level: int = 20) -> float:
    c = 0.5 * (a + b)
    fa, fb = f(a), f(b)
    fc = f(c)
    S = simpson_step(f, a, b, fa, fb, fc)
    return _adaptive_simpson_recursive(f, a, b, tol, max_level, fa, fb, fc, S)


def simpson_step(f, a, b, fa, fb, fc):
    return (b - a) / 6.0 * (fa + 4.0 * fc + fb)


def _adaptive_simpson_recursive(f, a, b, tol, max_level, fa, fb, fc, S):
    c = 0.5 * (a + b)
    d = 0.5 * (a + c)
    e = 0.5 * (c + b)
    fd = f(d)
    fe = f(e)
    Sleft = simpson_step(f, a, c, fa, fc, fd)
    Sright = simpson_step(f, c, b, fc, fb, fe)
    S2 = Sleft + Sright
    if max_level <= 0 or abs(S2 - S) <= 15.0 * tol:
        return S2 + (S2 - S) / 15.0
    left = _adaptive_simpson_recursive(f, a, c, 0.5 * tol, max_level - 1, fa, fc, fd, Sleft)
    right = _adaptive_simpson_recursive(f, c, b, 0.5 * tol, max_level - 1, fc, fb, fe, Sright)
    return left + right





def cubic_spline_coeffs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x) - 1
    if n < 2:
        raise ValueError("至少需要3个点做样条")
    h = np.diff(x)
    delta = np.diff(y) / h


    a_tri = np.zeros(n + 1, dtype=np.float64)
    b_tri = np.zeros(n + 1, dtype=np.float64)
    c_tri = np.zeros(n + 1, dtype=np.float64)
    d_tri = np.zeros(n + 1, dtype=np.float64)

    b_tri[0] = h[1]
    c_tri[0] = h[0] + h[1]
    d_tri[0] = ((h[0] + 2.0 * c_tri[0]) * h[1] * delta[0] + h[0] ** 2 * delta[1]) / c_tri[0]
    for i in range(1, n):
        a_tri[i] = h[i]
        b_tri[i] = 2.0 * (h[i - 1] + h[i])
        c_tri[i] = h[i - 1]
        d_tri[i] = 3.0 * (h[i] * delta[i - 1] + h[i - 1] * delta[i])
    a_tri[n] = h[n - 2] + h[n - 1]
    b_tri[n] = h[n - 2]
    d_tri[n] = ((h[n - 1] + 2.0 * a_tri[n]) * h[n - 2] * delta[n - 1] + h[n - 1] ** 2 * delta[n - 2]) / a_tri[n]

    c = tridiag_solve(a_tri, b_tri, c_tri, d_tri)
    a = y[:-1]
    b_coeff = np.zeros(n, dtype=np.float64)
    d_coeff = np.zeros(n, dtype=np.float64)
    for i in range(n):
        b_coeff[i] = delta[i] - h[i] / 3.0 * (2.0 * c[i] + c[i + 1])
        d_coeff[i] = (c[i + 1] - c[i]) / (3.0 * h[i]) if h[i] != 0 else 0.0
    return a, b_coeff, c[:-1], d_coeff


def cubic_spline_eval(xi: np.ndarray, x: np.ndarray, a: np.ndarray,
                      b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    result = np.zeros_like(xi)
    for j, xv in enumerate(xi):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1
            i = max(0, min(i, len(x) - 2))
        dx = xv - x[i]
        result[j] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return result





def cooley_tukey_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    if n == 0:
        return x

    N = 1
    while N < n:
        N <<= 1
    if N > n:
        x = np.pad(x, (0, N - n), mode='constant')

    x = _bit_reverse_copy(x)

    for s in range(1, int(np.log2(N)) + 1):
        m = 1 << s
        wm = np.exp(-2j * np.pi / m)
        for k in range(0, N, m):
            w = 1.0 + 0.0j
            half = m >> 1
            for j in range(half):
                t = w * x[k + j + half]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + half] = u - t
                w *= wm
    return x[:n] if n == N else x


def inverse_fft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return np.conj(cooley_tukey_fft(np.conj(x))) / n


def _bit_reverse_copy(x: np.ndarray) -> np.ndarray:
    N = len(x)
    n_bits = int(np.log2(N))
    A = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        rev = 0
        for j in range(n_bits):
            rev = (rev << 1) | ((i >> j) & 1)
        A[rev] = x[i]
    return A





def matrix_condition_estimate(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] == 0:
        return np.inf
    return s[0] / s[-1]


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(a, fill_value, dtype=np.float64)
    mask = np.abs(b) > 1e-300
    result[mask] = a[mask] / b[mask]
    return result
