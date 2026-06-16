"""
linear_algebra.py
=================
Core linear-algebra primitives for the KKT solver:

  * PLU factorisation with partial pivoting  (from r8ge/r8ge_fa)
  * Forward / backward substitution           (from r8ge/r8ge_sl)
  * Conjugate-gradient iteration              (from r8ge/r8ge_cg)
  * Condition-number estimator                (from r8ge/r8ge_dif2 / r8ge_res)
  * Iterative refinement

These replace any scipy.linalg dependency; the entire project needs only
numpy.  The PLU routines are direct translations of the LINPACK-style
r8ge_fa / r8ge_sl kernels by Burkardt & Dongarra.

KKT role
--------
The KKT saddle-point system is factored once per active-set iteration
via PLU; iterative refinement compensates for the ill-conditioning of
the Hilbert-type blocks that arise in the regularisation matrix.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# PLU factorisation  (LINPACK r8ge_fa)
# ---------------------------------------------------------------------------

def plu_factor(A: np.ndarray):
    """PLU factorisation of an (n, n) matrix with partial pivoting.

    Returns
    -------
    LU    : (n, n) array storing L (unit lower, below diagonal) and
            U (upper, on and above diagonal).
    pivot : (n,) int array of pivot indices (1-based in LINPACK, 0-based here).
    info  : 0 on success; k if U(k,k) is exactly zero.
    """
    A = np.array(A, dtype=np.float64, order="F")
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("plu_factor: matrix must be square")
    pivot = np.zeros(n, dtype=np.int64)
    info = 0

    for k in range(n):
        # Find pivot --------------------------------------------------
        kp = k + int(np.argmax(np.abs(A[k:n, k])))
        pivot[k] = kp
        if abs(A[kp, k]) < 1.0e-300:
            info = k + 1
            return A, pivot, info
        # Swap rows ---------------------------------------------------
        if kp != k:
            A[[k, kp]] = A[[kp, k]]
        # Eliminate ---------------------------------------------------
        for i in range(k + 1, n):
            A[i, k] /= A[k, k]
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]
    return A, pivot, info


def plu_solve(LU: np.ndarray, pivot: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve  A x = b  given the PLU factorisation from plu_factor."""
    LU = np.asfortranarray(LU)
    b = np.array(b, dtype=np.float64).ravel().copy()
    n = LU.shape[0]
    if b.size != n:
        raise ValueError(f"plu_solve: b has size {b.size}, expected {n}")

    # Apply row permutations
    for k in range(n):
        kp = int(pivot[k])
        if kp != k:
            b[k], b[kp] = b[kp], b[k]
    # Forward substitution  L z = P b
    for k in range(n):
        for i in range(k + 1, n):
            b[i] -= LU[i, k] * b[k]
    # Backward substitution  U x = z
    for k in range(n - 1, -1, -1):
        b[k] /= LU[k, k]
        for i in range(k):
            b[i] -= LU[i, k] * b[k]
    return b


def plu_solve_safe(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience wrapper: factor and solve in one call."""
    LU, pivot, info = plu_factor(A)
    if info != 0:
        raise ValueError(f"plu_solve_safe: singular at pivot {info}")
    return plu_solve(LU, pivot, b)


# ---------------------------------------------------------------------------
# Conjugate-gradient solver  (from r8ge/r8ge_cg)
# ---------------------------------------------------------------------------

def cg_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1.0e-10,
) -> tuple[np.ndarray, dict]:
    """Conjugate-gradient method for SPD systems.

    Returns the solution x and a dict with keys 'iters', 'residual', 'converged'.
    """
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64).ravel()
    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)
    b_norm = max(np.linalg.norm(b), 1.0e-30)

    info = {"iters": 0, "residual": np.sqrt(rs_old) / b_norm, "converged": False}
    for it in range(max_iter):
        Ap = A @ p
        pAp = float(p @ Ap)
        if abs(pAp) < 1.0e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        info["iters"] = it + 1
        info["residual"] = math_sqrt(rs_new) / b_norm
        if info["residual"] < tol:
            info["converged"] = True
            break
        beta = rs_new / max(rs_old, 1.0e-30)
        p = r + beta * p
        rs_old = rs_new
    return x, info


def math_sqrt(x: float) -> float:
    """Safe sqrt for non-negative arguments."""
    return float(np.sqrt(max(x, 0.0)))


# ---------------------------------------------------------------------------
# Condition-number estimator (1-norm, Hager-style)
# ---------------------------------------------------------------------------

def condition_estimate(A: np.ndarray) -> float:
    """Estimate  cond_1(A) = ||A||_1 * ||A^{-1}||_1  via 5 power iterations
    of the Hager algorithm.  Falls back to inf if A is singular."""
    n = A.shape[0]
    LU, pivot, info = plu_factor(A.copy())
    if info != 0:
        return float("inf")

    def solve(v):
        return plu_solve(LU, pivot, v)

    x = np.ones(n) / n
    for _ in range(5):
        xi = solve(x)
        xi_norm = np.linalg.norm(xi, 1)
        if xi_norm < 1.0e-30:
            return float("inf")
        xi /= xi_norm
        z = solve(xi) if False else xi  # simplified
        z = np.sign(xi)
        z = solve(z)
        z_norm = np.linalg.norm(z, np.inf)
        if z_norm <= float(xi @ z):
            break
        x = np.zeros(n)
        j = int(np.argmax(np.abs(z)))
        x[j] = 1.0
    A1 = max(np.sum(np.abs(A), axis=0))
    return A1 * xi_norm


# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------

def iterative_refinement(
    A: np.ndarray,
    b: np.ndarray,
    max_refine: int = 3,
) -> np.ndarray:
    """Improve an initial PLU solve by iterative refinement.

    At each step:
        r_k = b - A x_k           (residual in double precision)
        d_k = (PLU)^{-1} r_k
        x_{k+1} = x_k + d_k

    This recovers accuracy lost to the ill-conditioning of Hilbert-type
    blocks in the KKT system.
    """
    LU, pivot, info = plu_factor(A.copy())
    if info != 0:
        raise ValueError("iterative_refinement: singular matrix")
    x = plu_solve(LU, pivot, b)
    for _ in range(max_refine):
        r = b - A @ x
        if np.linalg.norm(r) < 1.0e-14 * np.linalg.norm(b):
            break
        d = plu_solve(LU, pivot, r)
        x += d
    return x


# ---------------------------------------------------------------------------
# Matrix inversion via PLU
# ---------------------------------------------------------------------------

def invert_matrix(A: np.ndarray) -> np.ndarray:
    """Compute A^{-1} by solving A X = I column by column.

    Uses PLU factorisation with iterative refinement for each column.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("invert_matrix: matrix must be square")
    LU, pivot, info = plu_factor(A.copy())
    if info != 0:
        raise ValueError(f"invert_matrix: singular at pivot {info}")
    Ainv = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        ej = np.zeros(n, dtype=np.float64)
        ej[j] = 1.0
        x = plu_solve(LU, pivot, ej)
        # Iterative refinement
        for _ in range(2):
            r = ej - A @ x
            if np.linalg.norm(r) < 1.0e-14:
                break
            x += plu_solve(LU, pivot, r)
        Ainv[:, j] = x
    return Ainv
