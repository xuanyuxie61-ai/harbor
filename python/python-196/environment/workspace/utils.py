
import numpy as np


def rref_matrix(a, tol=None):
    a = np.array(a, dtype=float)
    m, n = a.shape
    if tol is None:
        tol = np.finfo(float).eps * np.sum(np.abs(a))
    det = 1.0
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        i = r
        while abs(a[i, lead]) <= tol:
            i += 1
            if i >= m:
                i = r
                lead += 1
                if lead >= n:
                    lead = -1
                    break
        if lead < 0:
            break

        a[[i, r], :] = a[[r, i], :].copy()
        det *= a[r, lead]
        a[r, :] = a[r, :] / a[r, lead]
        for i2 in range(m):
            if i2 != r:
                a[i2, :] = a[i2, :] - a[i2, lead] * a[r, :]
        lead += 1
    return a, det


def hypersphere_surface_area(dim):
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if dim == 1:
        return 2.0
    if dim % 2 == 0:
        m_half = dim // 2
        area = 2.0 * (np.pi ** m_half)
        for i in range(1, m_half):
            area /= i
    else:
        m_half = (dim - 1) // 2
        area = (np.pi ** m_half) * (2.0 ** dim)
        for i in range(m_half + 1, 2 * m_half + 1):
            area /= i
    return float(area)


def hypersphere_volume(dim):
    return hypersphere_surface_area(dim) / dim


def check_positive_definite_symmetric(mat, tol=1e-12):
    mat = np.array(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    if not np.allclose(mat, mat.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(mat)
    return np.all(eigvals > tol)


def cholesky_factor(a):
    a = np.array(a, dtype=float)
    if not check_positive_definite_symmetric(a):
        raise ValueError("Matrix is not positive definite symmetric")
    return np.linalg.cholesky(a).T


def safe_log(x, eps=1e-300):
    x = np.array(x, dtype=float)
    x = np.where(x <= 0, eps, x)
    return np.log(x)


def safe_sqrt(x):
    x = np.array(x, dtype=float)
    x = np.where(x < 0, 0.0, x)
    return np.sqrt(x)


def binomial_coeff(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res
