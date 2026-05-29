"""
linear_algebra.py
=================
Linear algebra utilities synthesized from seed projects:
  - 1098_solve (Gaussian elimination with partial pivoting)
  - 1159_st_to_msm (sparse triplet to dense matrix conversion)
  - 737_matrix_analyze (comprehensive matrix property analysis)

Core algorithms:
  - Gaussian elimination with partial pivoting for A x = b
  - Sparse triplet (ST) format to dense matrix conversion
  - Matrix property tests: symmetry, diagonal dominance, SPD, normality, orthogonality
  - Cholesky factorization
"""

import numpy as np


def gaussian_solve(A, b):
    """
    Solve A x = b using Gaussian elimination with partial pivoting.
    
    Based on seed 1098_solve (r8mat_fs).
    Algorithm:
      For each column jcol = 0 .. n-1:
        1. Find pivot row: argmax_{row >= jcol} |A[row, jcol]|
        2. Swap rows if needed
        3. Normalize pivot row: A[jcol, jcol+1:] /= A[jcol, jcol]
        4. Eliminate below pivot
        5. Back-substitute from bottom to top
    
    Mathematical formulation:
      The system is transformed to upper triangular form U x = c,
      then solved by x_i = (c_i - sum_{j=i+1}^{n-1} U_{ij} x_j) / U_{ii}.
    """
    n = A.shape[0]
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    
    for jcol in range(n):
        # Partial pivoting: find row with maximum absolute value in column jcol
        pivot = jcol + np.argmax(np.abs(A[jcol:, jcol]))
        if abs(A[pivot, jcol]) < 1e-15:
            raise ValueError("Zero pivot encountered -- matrix is singular or near-singular")
        if pivot != jcol:
            A[[jcol, pivot], :] = A[[pivot, jcol], :]
            b[[jcol, pivot]] = b[[pivot, jcol]]
        
        # Normalize pivot row
        pivval = A[jcol, jcol]
        A[jcol, jcol+1:] /= pivval
        b[jcol] /= pivval
        A[jcol, jcol] = 1.0
        
        # Eliminate entries below pivot
        for i in range(jcol + 1, n):
            factor = A[i, jcol]
            if abs(factor) > 0.0:
                A[i, jcol+1:] -= factor * A[jcol, jcol+1:]
                b[i] -= factor * b[jcol]
                A[i, jcol] = 0.0
    
    # Back substitution
    x = np.zeros(n, dtype=float)
    for jcol in range(n - 1, -1, -1):
        x[jcol] = b[jcol] - np.dot(A[jcol, jcol+1:], x[jcol+1:])
    return x


def sparse_triplet_to_dense(rows, cols, vals, m, n):
    """
    Convert sparse triplet (ST) format to dense matrix.
    
    Based on seed 1159_st_to_msm.
    ST files store (row, col, value) triplets with 0-based indexing.
    This function rebases from 0-based to 0-based (Python convention).
    
    Mathematical operation:
      A = sum_k vals[k] * e_{rows[k]} e_{cols[k]}^T
    where e_i is the i-th standard basis vector.
    """
    A = np.zeros((m, n), dtype=float)
    for r, c, v in zip(rows, cols, vals):
        if 0 <= r < m and 0 <= c < n:
            A[r, c] += v
    return A


def frobenius_norm(A):
    """
    Frobenius matrix norm:
      ||A||_F = sqrt( sum_{i,j} A_{ij}^2 )
    """
    return np.sqrt(np.sum(A * A))


def is_symmetric(A, tol=1e-10):
    """Test symmetry: ||A - A^T||_F < tol."""
    if A.shape[0] != A.shape[1]:
        return False
    return frobenius_norm(A - A.T) < tol


def is_diagonally_dominant(A):
    """
    Test strict diagonal dominance:
      |A_{ii}| > sum_{j != i} |A_{ij}|  for all i
    """
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_sum = np.sum(np.abs(A[i, :])) - diag
        if diag <= off_sum:
            return False
    return True


def cholesky_factor(A):
    """
    Cholesky factorization: A = L * L^T for symmetric positive definite A.
    
    Based on seed 737_matrix_analyze (r8mat_cholesky_factor).
    Algorithm:
      For j = 0 .. n-1:
        L[j,j] = sqrt( A[j,j] - sum_{k=0}^{j-1} L[j,k]^2 )
        For i = j+1 .. n-1:
          L[i,j] = ( A[i,j] - sum_{k=0}^{j-1} L[i,k] L[j,k] ) / L[j,j]
    """
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
    """Test SPD via Cholesky factorization."""
    try:
        cholesky_factor(A)
        return True
    except ValueError:
        return False


def is_normal_matrix(A, tol=1e-8):
    """
    Test normality: A A^T = A^T A.
    A normal matrix is unitarily diagonalizable.
    """
    if A.shape[0] != A.shape[1]:
        return False
    return frobenius_norm(A @ A.T - A.T @ A) < tol


def is_orthogonal_matrix(A, tol=1e-8):
    """
    Test orthogonality: A^T A = I.
    For orthogonal matrices, A^{-1} = A^T.
    """
    if A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    I = np.eye(n)
    return frobenius_norm(A.T @ A - I) < tol


def analyze_matrix_properties(A):
    """
    Comprehensive matrix analysis based on seed 737_matrix_analyze.
    Tests ~10 different structural and algebraic properties.
    
    Returns a dictionary with analysis results.
    """
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
