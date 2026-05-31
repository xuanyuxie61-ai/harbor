"""
periodic_solver.py

Specialized direct solvers for periodic tridiagonal (R83P) linear systems.
Reimplemented from the R83P seed project for efficient Gaussian Markov random
field prior operations in circular spatial domains.
"""
import numpy as np


def r83_np_fa(n: int, a: np.ndarray):
    """
    Factor a non-periodic tridiagonal matrix stored in R83 format.

    R83 storage (3 x n):
        a[0, 1:] = superdiagonal
        a[1, :]  = diagonal
        a[2, :-1] = subdiagonal

    Returns:
        a_lu: factorization data (same shape)
        info: 0 if successful, else index of zero pivot
    """
    if n < 2:
        raise ValueError("r83_np_fa requires n >= 2")
    if a.shape != (3, n):
        raise ValueError(f"r83_np_fa: a must be shape (3, {n}), got {a.shape}")

    a_lu = a.copy()
    info = 0

    for i in range(n - 1):
        if a_lu[1, i] == 0.0:
            info = i + 1
            return a_lu, info
        a_lu[2, i] = a_lu[2, i] / a_lu[1, i]
        a_lu[1, i + 1] = a_lu[1, i + 1] - a_lu[2, i] * a_lu[0, i + 1]

    if a_lu[1, n - 1] == 0.0:
        info = n

    return a_lu, info


def r83_np_sl(n: int, a_lu: np.ndarray, b: np.ndarray, job: int):
    """
    Solve a non-periodic tridiagonal system factored by r83_np_fa.

    Parameters:
        n: order of matrix
        a_lu: factorization from r83_np_fa
        b: right-hand side (length n)
        job: 0 -> solve A x = b, nonzero -> solve A^T x = b

    Returns:
        x: solution vector (length n)
    """
    if n < 2:
        raise ValueError("r83_np_sl requires n >= 2")
    if a_lu.shape != (3, n):
        raise ValueError("r83_np_sl: a_lu shape mismatch")
    x = b.copy()

    if job == 0:
        # Forward solve L y = b
        for i in range(1, n):
            x[i] = x[i] - a_lu[2, i - 1] * x[i - 1]
        # Back solve U x = y
        for i in range(n - 1, -1, -1):
            x[i] = x[i] / a_lu[1, i]
            if i > 0:
                x[i - 1] = x[i - 1] - a_lu[0, i] * x[i]
    else:
        # Forward solve U^T y = b
        for i in range(n):
            x[i] = x[i] / a_lu[1, i]
            if i < n - 1:
                x[i + 1] = x[i + 1] - a_lu[0, i + 1] * x[i]
        # Back solve L^T x = y
        for i in range(n - 2, -1, -1):
            x[i] = x[i] - a_lu[2, i] * x[i + 1]

    return x


def r83p_fa(n: int, a: np.ndarray):
    """
    Factor a periodic tridiagonal matrix in R83P format.

    R83P storage (3 x n):
        a[0, 0]  = A(N,1)   (lower-left wrap)
        a[0, 1:] = superdiagonal A(j, j+1)
        a[1, :]  = diagonal
        a[2, :-1] = subdiagonal A(j+1, j)
        a[2, -1] = A(1,N)   (upper-right wrap)

    Uses block Schur-complement decomposition treating the matrix as
        [ A1  A2 ]
        [ A3  A4 ]
    where A1 is the (n-1)x(n-1) tridiagonal leading principal submatrix.

    Returns:
        a_lu: factorization data (3 x n)
        work2, work3, work4: auxiliary data for r83p_sl
        info: 0 if successful
    """
    if n < 3:
        raise ValueError("r83p_fa requires n >= 3")
    if a.shape != (3, n):
        raise ValueError(f"r83p_fa: a must be shape (3, {n}), got {a.shape}")

    a_lu = np.zeros((3, n), dtype=float)

    # Factor A1 (leading (n-1)x(n-1) block)
    a_lu[:, : n - 1], info = r83_np_fa(n - 1, a[:, : n - 1])
    if info != 0:
        return a_lu, None, None, None, info

    # Restore corner entries for the periodic part
    a_lu[0, 0] = a[0, 0]
    a_lu[2, n - 2] = a[2, n - 2]
    a_lu[:, n - 1] = a[:, n - 1]

    # work2 = inverse(A1) * A2
    work2 = np.zeros(n - 1, dtype=float)
    work2[0] = a[2, n - 1]   # upper-right wrap
    if n - 1 > 2:
        work2[1 : n - 2] = 0.0
    work2[n - 2] = a[0, n - 1]  # last superdiagonal
    work2 = r83_np_sl(n - 1, a_lu[:, : n - 1], work2, 0)

    # work3 = inverse(A1^T) * A3^T
    work3 = np.zeros(n - 1, dtype=float)
    work3[0] = a[0, 0]        # lower-left wrap
    if n - 1 > 2:
        work3[1 : n - 2] = 0.0
    work3[n - 2] = a[2, n - 2]  # last subdiagonal
    work3 = r83_np_sl(n - 1, a_lu[:, : n - 1], work3, 1)

    # Schur complement: A4 := A4 - A3 * inv(A1) * A2
    work4 = a[1, n - 1] - a[0, 0] * work2[0] - a[2, n - 2] * work2[n - 2]

    if abs(work4) < 1e-15:
        info = n
        return a_lu, work2, work3, work4, info

    return a_lu, work2, work3, work4, 0


def r83p_sl(n: int, a_lu: np.ndarray, b: np.ndarray, job: int,
            work2: np.ndarray, work3: np.ndarray, work4: float):
    """
    Solve a periodic tridiagonal system factored by r83p_fa.

    Parameters:
        n, a_lu, job: as in r83_np_sl
        work2, work3, work4: auxiliary data from r83p_fa

    Returns:
        x: solution vector
    """
    if n < 3:
        raise ValueError("r83p_sl requires n >= 3")
    x = b.copy()

    if job == 0:
        # Solve A1 * X1 = B1
        x[: n - 1] = r83_np_sl(n - 1, a_lu[:, : n - 1], x[: n - 1], 0)
        # X2 = B2 - A3 * X1
        x[n - 1] = x[n - 1] - a_lu[0, 0] * x[0] - a_lu[2, n - 2] * x[n - 2]
        # Solve A4 * X2 = X2
        x[n - 1] = x[n - 1] / work4
        # X1 := X1 - inv(A1) * A2 * X2
        x[: n - 1] = x[: n - 1] - work2 * x[n - 1]
    else:
        # Solve A1^T * X1 = B1
        x[: n - 1] = r83_np_sl(n - 1, a_lu[:, : n - 1], x[: n - 1], 1)
        # X2 := X2 - A2^T * B1
        x[n - 1] = x[n - 1] - a_lu[2, n - 1] * x[0] - a_lu[0, n - 1] * x[n - 2]
        # Solve A4 * X2 = X2
        x[n - 1] = x[n - 1] / work4
        # X1 := X1 - (inv(A1) * A3)^T * X2
        x[: n - 1] = x[: n - 1] - work3 * x[n - 1]

    return x
