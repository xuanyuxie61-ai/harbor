
import numpy as np


def gaussian_solve(A, b):
    n = A.shape[0]
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    
    for jcol in range(n):

        pivot = jcol + np.argmax(np.abs(A[jcol:, jcol]))
        if abs(A[pivot, jcol]) < 1e-15:
            raise ValueError("Zero pivot encountered -- matrix is singular or near-singular")
        if pivot != jcol:
            A[[jcol, pivot], :] = A[[pivot, jcol], :]
            b[[jcol, pivot]] = b[[pivot, jcol]]
        

        pivval = A[jcol, jcol]
        A[jcol, jcol+1:] /= pivval
        b[jcol] /= pivval
        A[jcol, jcol] = 1.0
        

        for i in range(jcol + 1, n):
            factor = A[i, jcol]
            if abs(factor) > 0.0:
                A[i, jcol+1:] -= factor * A[jcol, jcol+1:]
                b[i] -= factor * b[jcol]
                A[i, jcol] = 0.0
    

    x = np.zeros(n, dtype=float)
    for jcol in range(n - 1, -1, -1):
        x[jcol] = b[jcol] - np.dot(A[jcol, jcol+1:], x[jcol+1:])
    return x


def sparse_triplet_to_dense(rows, cols, vals, m, n):
    A = np.zeros((m, n), dtype=float)
    for r, c, v in zip(rows, cols, vals):
        if 0 <= r < m and 0 <= c < n:
            A[r, c] += v
    return A


def frobenius_norm(A):
    return np.sqrt(np.sum(A * A))


def is_symmetric(A, tol=1e-10):
    if A.shape[0] != A.shape[1]:
        return False
    return frobenius_norm(A - A.T) < tol


def is_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_sum = np.sum(np.abs(A[i, :])) - diag
        if diag <= off_sum:
            return False
    return True


def cholesky_factor(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    for j in range(n):
        t = A[j, j] - np.sum(L[j, :j] ** 2)
        if t <= 1e-15:
            raise ValueError("Matrix is not positive definite (negative pivot)")
        L[j, j] = np.sqrt(t)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L


def is_positive_definite(A):
    try:
        cholesky_factor(A)
        return True
    except ValueError:
        return False


def is_normal_matrix(A, tol=1e-8):
    if A.shape[0] != A.shape[1]:
        return False
    return frobenius_norm(A @ A.T - A.T @ A) < tol


def is_orthogonal_matrix(A, tol=1e-8):
    if A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    I = np.eye(n)
    return frobenius_norm(A.T @ A - I) < tol


def analyze_matrix_properties(A):
    results = {
        'shape': A.shape,
        'frobenius_norm': float(frobenius_norm(A)),
        'is_square': (A.shape[0] == A.shape[1]),
    }
    if results['is_square']:
        results['is_symmetric'] = is_symmetric(A)
        results['is_diagonally_dominant'] = is_diagonally_dominant(A)
        results['is_normal'] = is_normal_matrix(A)
        if A.shape[0] <= 200:
            results['is_orthogonal'] = is_orthogonal_matrix(A)
        if results['is_symmetric']:
            results['is_spd'] = is_positive_definite(A)
    return results
