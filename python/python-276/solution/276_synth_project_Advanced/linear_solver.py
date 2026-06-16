"""
linear_solver.py — Dense LU solver with partial pivoting
========================================================

The elastic equilibrium equations around a defect, the Poisson equation for
the Hartree potential, and the Dyson equation for the cluster Green's
function all reduce at some point to solving a dense or sparse linear
system  A x = b. This module provides a straightforward LU factorisation
with partial pivoting — a Python port of the classical LINPACK routines
`dgefa` and `dgesl` from seed project `687_linpack_bench/linpack_bench.m`.

The implementation is intentionally simple (no BLAS) because the systems
in this project are small (N ≤ a few thousand) and the goal is clarity
and reproducibility rather than raw speed.

The LINPACK benchmark itself (solving A x = b and reporting Mflop/s) is
reproduced in `run_linpack_benchmark` for engineering-complexity testing
of the host machine — useful to characterise the runtime of the full
defect pipeline.

Seed project integration:
  * 687_linpack_bench/linpack_bench.m: dgefa + dgesl + daxpy + timing.
"""

from __future__ import annotations
import math
import time
import numpy as np
from typing import Tuple


# -------------------------------------------------------------------------
# (1) BLAS-1 helper: daxpy (y ← y + α x)
# -------------------------------------------------------------------------
def daxpy(n: int, alpha: float, x: np.ndarray, xi: int,
          y: np.ndarray, yi: int) -> None:
    """Port of the LINPACK `daxpy`:  y[i:] += α x[j:]."""
    for k in range(n):
        y[yi + k] += alpha * x[xi + k]


# -------------------------------------------------------------------------
# (2) LU factorisation with partial pivoting (LINPACK dgefa)
# -------------------------------------------------------------------------
def dgefa(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """LU factorisation of a dense matrix A with partial pivoting.

    Returns (LU, ipvt, info):
      * LU : (n, n) ndarray — L and U packed together (L has unit diagonal)
      * ipvt : (n,) int array — pivot indices
      * info : 0 if successful, k+1 if U[k, k] is exactly zero

    Port of LINPACK `dgefa.f`.
    """
    n = A.shape[0]
    LU = A.astype(np.float64).copy()
    ipvt = np.zeros(n, dtype=np.int64)
    info = 0

    for k in range(n):
        # find pivot
        kp1 = k
        for i in range(k + 1, n):
            if abs(LU[i, k]) > abs(LU[kp1, k]):
                kp1 = i
        ipvt[k] = kp1
        if abs(LU[kp1, k]) == 0.0:
            info = k + 1
            continue
        # swap rows k and kp1
        if kp1 != k:
            LU[[k, kp1]] = LU[[kp1, k]]
        # elimination
        for i in range(k + 1, n):
            LU[i, k] = LU[i, k] / LU[k, k]
            # daxpy for the rest of the row
            daxpy(n - k - 1, -LU[i, k],
                  LU[k, :], k + 1, LU[i, :], k + 1)
    return LU, ipvt, info


# -------------------------------------------------------------------------
# (3) LU solve (LINPACK dgesl)
# -------------------------------------------------------------------------
def dgesl(LU: np.ndarray, ipvt: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b given the LU factorisation from `dgefa`.

    Port of LINPACK `dgesl.f`.
    """
    n = LU.shape[0]
    x = b.astype(np.float64).copy()

    # forward substitution with pivoting
    for k in range(n):
        p = ipvt[k]
        if p != k:
            x[k], x[p] = x[p], x[k]
        for i in range(k + 1, n):
            x[i] += -LU[i, k] * x[k]   # actually: x[i] -= LU[i, k] * x[k]
        # equivalently (and correctly):
        # daxpy(n - k - 1, x[k], LU[k+1:n, k], x[k+1:n])
    # we redo the forward substitution cleanly
    x = b.astype(np.float64).copy()
    for k in range(n - 1):
        p = ipvt[k]
        x[k], x[p] = x[p], x[k]
        for i in range(k + 1, n):
            x[i] -= LU[i, k] * x[k]
    # backward substitution
    for k in range(n - 1, -1, -1):
        x[k] = x[k] / LU[k, k]
        for i in range(k):
            x[i] -= LU[i, k] * x[k]
    return x


# -------------------------------------------------------------------------
# (4) High-level wrapper
# -------------------------------------------------------------------------
def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b via LU with partial pivoting."""
    LU, ipvt, info = dgefa(A)
    if info != 0:
        raise ValueError(f"LU factorisation failed at pivot {info}")
    return dgesl(LU, ipvt, b)


def residual_norm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Return ||A x − b||_∞."""
    r = A @ x - b
    return float(np.max(np.abs(r)))


# -------------------------------------------------------------------------
# (5) Test-system generator
# -------------------------------------------------------------------------
def make_test_system(n: int, seed: int = 276
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random well-conditioned system A x = b of size n.

    A is a diagonally-dominant random matrix, x_true is random, and
    b = A x_true. Returns (A, x_true, b).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A + n * np.eye(n)       # make diagonally dominant
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return A, x_true, b


# -------------------------------------------------------------------------
# (6) Linpack-style benchmark
# -------------------------------------------------------------------------
def run_linpack_benchmark(n: int = 100, nits: int = 5
                          ) -> Tuple[float, float]:
    """Run the Linpack-style benchmark: nits solves of A x = b.

    Returns (Mflop/s, average residual norm). The Linpack Mflop count for
    one LU solve of size n is  (2/3) n³  flops.
    """
    A, _, b = make_test_system(n, seed=42)
    residuals = []
    t0 = time.perf_counter()
    for _ in range(nits):
        x = solve_lu(A.copy(), b.copy())
        residuals.append(residual_norm(A, b, x))
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / nits
    mflops = (2.0 / 3.0) * n ** 3 / elapsed / 1e6
    return mflops, float(np.mean(residuals))
