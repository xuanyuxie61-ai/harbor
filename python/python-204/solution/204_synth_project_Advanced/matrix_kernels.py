"""
matrix_kernels.py
=================

Dense matrix-matrix kernels inspired by the ``mxm`` benchmark
(Burkardt).  In the Sobol / polynomial-chaos pipeline these kernels
appear in two places:

1. **Fast PCE regression** — the design matrix ``Phi`` of size
   ``(N_samples, P_bases)`` is multiplied against the response vector
   ``y`` in block form; we therefore need an optimised GEMM-style
   triple loop with *loop tiling* for cache locality.
2. **Sobol second-order index aggregation** — the estimator
   ``S_{ij} = Cov(f(A_i^B), f(B_j^A)) / Var(Y)`` is assembled as a
   Gram matrix ``G = U @ U.T`` where ``U`` is the centered response
   block.  The same tiled GEMM is reused.

All kernels are written in pure Python/NumPy but are structured so
that a future drop-in BLAS swap is trivial.  A condition-number
guard detects rank deficiency before solving any least-squares
sub-problem (the QR path lives in ``quadrature_pce``).

References
----------
* J. Burkardt, ``mxm`` — matrix-matrix product benchmark.
* Golub & Van Loan, *Matrix Computations*, 4th ed., §1.4, §3.2.
"""

from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------
# Tiled GEMM
# ---------------------------------------------------------------------
def mxm_tiled(A: np.ndarray, B: np.ndarray, tile: int = 64) -> np.ndarray:
    """Compute ``C = A @ B`` using blocked (tiled) multiplication.

    Parameters
    ----------
    A, B : 2-D arrays
        ``A.shape == (n1, n2)``, ``B.shape == (n2, n3)``.
    tile : int
        Block size for all three loops.

    Returns
    -------
    C : ndarray
        ``(n1, n3)`` matrix product.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("mxm_tiled: A and B must be 2-D")
    n1, n2 = A.shape
    n2b, n3 = B.shape
    if n2 != n2b:
        raise ValueError(f"mxm_tiled: shape mismatch ({n1},{n2}) x ({n2b},{n3})")

    C = np.zeros((n1, n3), dtype=np.result_type(A, B))
    for i0 in range(0, n1, tile):
        i1 = min(i0 + tile, n1)
        for k0 in range(0, n2, tile):
            k1 = min(k0 + tile, n2)
            A_blk = A[i0:i1, k0:k1]
            for j0 in range(0, n3, tile):
                j1 = min(j0 + tile, n3)
                C[i0:i1, j0:j1] += A_blk @ B[k0:k1, j0:j1]
    return C


# ---------------------------------------------------------------------
# Timed wrapper (benchmark + reproducibility)
# ---------------------------------------------------------------------
def mxm_timed(A: np.ndarray, B: np.ndarray) -> dict:
    """Run ``A @ B`` and report wall time + basic stats.

    Returns a dict ``{'C': ..., 'time_s': ..., 'flops': ..., 'gflops': ...}``.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("mxm_timed: A and B must be 2-D")
    n1, n2 = A.shape
    _, n3 = B.shape
    flops = float(2 * n1 * n2 * n3)
    t0 = time.perf_counter()
    C = mxm_tiled(A, B)
    dt = time.perf_counter() - t0
    gflops = flops / max(dt, 1.0e-12) / 1.0e9
    return dict(C=C, time_s=dt, flops=flops, gflops=gflops)


# ---------------------------------------------------------------------
# Gram matrix G = U @ U.T  (used by Sobol 2nd-order indices)
# ---------------------------------------------------------------------
def gram_centered(U: np.ndarray) -> np.ndarray:
    """Return the centered Gram matrix ``(U - mu) @ (U - mu).T``.

    Centering removes the sample mean so that ``gram_centered / (N - 1)``
    equals the covariance matrix of the rows of ``U``.
    """
    if U.ndim != 2:
        raise ValueError("gram_centered: U must be 2-D")
    mu = U.mean(axis=0, keepdims=True)
    Uc = U - mu
    return mxm_tiled(Uc, Uc.T)


# ---------------------------------------------------------------------
# Sobol 2nd-order cross-Gram  G_{ij} = U_i @ U_j.T  (centered)
# ---------------------------------------------------------------------
def cross_gram_centered(Ui: np.ndarray, Uj: np.ndarray) -> np.ndarray:
    """Centered cross-Gram between two response blocks.

    Used to assemble the 2nd-order Sobol index estimator via
    ``S_{ij} = (Ui_centered @ Uj_centered.T).mean() / Var(Y)``.
    """
    if Ui.ndim != 2 or Uj.ndim != 2:
        raise ValueError("cross_gram_centered: Ui, Uj must be 2-D")
    if Ui.shape[0] != Uj.shape[0]:
        raise ValueError("cross_gram_centered: row count mismatch")
    mi = Ui.mean(axis=0, keepdims=True)
    mj = Uj.mean(axis=0, keepdims=True)
    return mxm_tiled(Ui - mi, (Uj - mj).T)


# ---------------------------------------------------------------------
# Condition-number guard (used before PCE regression)
# ---------------------------------------------------------------------
def safe_rcond(M: np.ndarray, tol: float = 1.0e-12) -> float:
    """Return reciprocal condition number of ``M``.

    Values below ``tol`` signal near-singularity; callers should switch
    to the rank-revealing QR path in ``quadrature_pce``.
    """
    if M.ndim != 2:
        raise ValueError("safe_rcond: M must be 2-D")
    if M.shape[0] == 0 or M.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0.0
    return float(s[-1] / s[0])


# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 80))
    B = rng.standard_normal((80, 90))
    res = mxm_timed(A, B)
    print(f"shape   : {res['C'].shape}")
    print(f"time [s]: {res['time_s']:.4e}")
    print(f"GFLOP/s : {res['gflops']:.3f}")
    U = rng.standard_normal((50, 7))
    G = gram_centered(U)
    print(f"Gram shape : {G.shape}, symmetric error: "
          f"{np.max(np.abs(G - G.T)):.3e}")
