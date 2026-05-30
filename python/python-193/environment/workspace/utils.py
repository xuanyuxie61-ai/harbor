
import numpy as np
import math


def safe_divide(a, b, default=0.0):
    eps = np.finfo(float).eps
    if np.isscalar(b):
        if abs(b) < eps:
            return default
        return a / b

    b_safe = np.where(np.abs(b) < eps, np.nan, b)
    result = np.full_like(a, default, dtype=float)
    valid = ~np.isnan(b_safe)
    result[valid] = a[valid] / b_safe[valid]
    return result


def i4_div_rounded(a, b):
    if b == 0:
        return 0


    sign = 1
    if a * b < 0:
        sign = -1
    a = abs(a)
    b = abs(b)
    value = (2 * a + b) // (2 * b)
    return sign * value


def double_factorial2(n):
    if n < 0:
        return 1.0
    if n == 0 or n == 1:
        return 1.0
    result = 1.0
    while n > 0:
        result *= n
        n -= 2
    return result


def legendre_monomial_integral(p):
    if p < 0:
        return 0.0
    if p % 2 == 1:
        return 0.0
    return 2.0 / (p + 1.0)


def chebyshev1_monomial_integral(p):
    if p < 0:
        return 0.0
    if p % 2 == 1:
        return 0.0
    return math.pi * double_factorial2(p - 1) / double_factorial2(p)


def hermite_monomial_integral(p):
    if p < 0:
        return 0.0
    if p % 2 == 1:
        return 0.0
    return double_factorial2(p - 1) * math.sqrt(math.pi) / (2.0 ** (p / 2.0))


def laguerre_monomial_integral(p):
    if p < 0:
        return 0.0
    return float(math.factorial(p))


def parameterize_arc_length(p_data):
    p_data = np.asarray(p_data, dtype=float)
    if p_data.ndim == 1:
        p_data = p_data.reshape(1, -1)
    m, data_num = p_data.shape
    t = np.zeros(data_num)
    for j in range(1, data_num):
        dist = np.linalg.norm(p_data[:, j] - p_data[:, j - 1])
        t[j] = t[j - 1] + dist
    tmax = t[-1]
    if tmax > 0:
        t /= tmax
    return t


def r8vec_bracket(x, xval):
    n = len(x)
    if n < 2:
        return -1
    if xval < x[0] - 1e-12:
        return -1
    if xval > x[-1] + 1e-12:
        return n - 1

    left = 0
    right = n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if xval < x[mid]:
            right = mid
        else:
            left = mid
    return left


def is_symmetric(A, tol=1e-10):
    if isinstance(A, np.ndarray):
        return np.allclose(A, A.T, atol=tol)

    try:
        from scipy.sparse import csr_matrix
        if not hasattr(A, 'T'):
            return False
        diff = A - A.T
        return np.max(np.abs(diff.data)) < tol if hasattr(diff, 'data') and len(diff.data) > 0 else True
    except Exception:
        return False


def condition_number_estimate(A):
    try:
        from scipy.sparse.linalg import eigsh
        n = A.shape[0]
        k = min(3, n - 1)
        if k < 1:
            return 1.0
        lambda_max = eigsh(A, k=k, which='LM', return_eigenvectors=False)
        lambda_min = eigsh(A, k=k, which='SM', return_eigenvectors=False)
        return abs(lambda_max[0]) / max(abs(lambda_min[0]), 1e-15)
    except Exception:
        return np.nan
