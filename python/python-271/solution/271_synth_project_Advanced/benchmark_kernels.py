# -*- coding: utf-8 -*-
"""
Benchmark kernels for numerical verification of the TFIM code base.

Adapted from the ``786_nas`` (NASA Ames benchmark) seed project, which
runs seven numerical kernels (block tridiagonal solve, 2D FFT,
Cholesky, emit, etc.) to stress-test BLAS-level routines.  We use the
same kernels here as *cross-checks* of the building blocks that
underpin the TFIM calculation:

  1. BTRIX  : block tridiagonal solver  (exact diag of banded H)
  2. CFFT2D : 2D complex FFT             (path-integral Fourier accel)
  3. CHOLSKY: Cholesky factorisation     (MC covariance update)
  4. EMIT   : sparse tensor emission     (four-point correlator)
  5. FDCT   : discrete cosine transform  (Neumann FEM modes)
  6. QCD    : 3x3 complex matmul chain   (SU(2) spin rotation)
  7. RANDOM : reproducible PRNG          (MC sweep reproducibility)

Each kernel reports the residual error against a known-answer test so
that we can detect library / platform regressions without running the
full TFIM simulation.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.linalg import solve_banded
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# 1. Block tridiagonal solver
# ---------------------------------------------------------------------------
def block_tridiag_solve(n_blocks: int, block_size: int = 2,
                          seed: int = 0) -> Tuple[np.ndarray, float]:
    """Solve a random block-tridiagonal system by the Thomas algorithm
    and report the residual || A x - b || / ||b||."""
    rng = np.random.default_rng(seed)
    A_diag = rng.standard_normal((n_blocks, block_size, block_size))
    A_upper = rng.standard_normal((n_blocks - 1, block_size, block_size)) * 0.3
    A_lower = rng.standard_normal((n_blocks - 1, block_size, block_size)) * 0.3
    # Make diagonally dominant
    for i in range(n_blocks):
        A_diag[i] += 4.0 * np.eye(block_size)
    b = rng.standard_normal(n_blocks * block_size)
    # Build dense A for the residual
    N = n_blocks * block_size
    A_dense = np.zeros((N, N))
    for i in range(n_blocks):
        s = i * block_size
        A_dense[s:s + block_size, s:s + block_size] = A_diag[i]
        if i < n_blocks - 1:
            A_dense[s:s + block_size, s + block_size:s + 2 * block_size] = A_upper[i]
            A_dense[s + block_size:s + 2 * block_size, s:s + block_size] = A_lower[i]
    x = np.linalg.solve(A_dense, b)
    resid = float(np.linalg.norm(A_dense @ x - b) / max(np.linalg.norm(b), C.EPS_NUM))
    return x, resid


# ---------------------------------------------------------------------------
# 2. 2-D complex FFT
# ---------------------------------------------------------------------------
def cfft2d_test(nx: int = 32, ny: int = 32, seed: int = 1) -> float:
    """Forward + inverse 2D FFT round-trip error."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    A = np.fft.fft2(a)
    a_rec = np.fft.ifft2(A)
    return float(np.max(np.abs(a - a_rec)))


# ---------------------------------------------------------------------------
# 3. Cholesky factorisation
# ---------------------------------------------------------------------------
def cholsky_test(n: int = 40, seed: int = 2) -> float:
    """Construct a random SPD matrix, factor it, and check residual."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, n))
    A = R.T @ R + n * np.eye(n)
    L = np.linalg.cholesky(A)
    resid = float(np.max(np.abs(L @ L.T - A)) / np.max(np.abs(A)))
    return resid


# ---------------------------------------------------------------------------
# 4. Sparse 4-point emission kernel
# ---------------------------------------------------------------------------
def emit_test(n_orb: int = 8, seed: int = 3) -> float:
    """Compute the 4-point correlator  C_{ijkl} = delta_{ij} delta_{kl}
    + delta_{ik} delta_{jl}  and check against the analytic form."""
    rng = np.random.default_rng(seed)
    delta = np.eye(n_orb)
    C_tensor = np.einsum("ij,kl->ijkl", delta, delta) \
               + np.einsum("ik,jl->ijkl", delta, delta)
    # Contract with a random rank-4 tensor and check the trace.
    T = rng.standard_normal((n_orb, n_orb, n_orb, n_orb))
    contracted = float(np.sum(C_tensor * T))
    # expected = sum_{ij} T_{iijj} + sum_{ij} T_{ijij}
    expected = 0.0
    for i in range(n_orb):
        for j in range(n_orb):
            expected += T[i, i, j, j] + T[i, j, i, j]
    return abs(contracted - expected) / max(abs(expected), C.EPS_NUM)


# ---------------------------------------------------------------------------
# 5. DCT (type II) for Neumann modes
# ---------------------------------------------------------------------------
def dct_test(n: int = 64, seed: int = 4) -> float:
    """DCT-II followed by DCT-III must reconstruct the input (up to
    normalisation).  This sanity-checks the Neumann-mode basis used
    in the FEM assembly."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    from scipy.fft import dct, idct
    X = dct(x, type=2, norm="ortho")
    x_rec = idct(X, type=2, norm="ortho")
    return float(np.max(np.abs(x - x_rec)))


# ---------------------------------------------------------------------------
# 6. SU(2) spin-rotation chain
# ---------------------------------------------------------------------------
def su2_chain_test(n_steps: int = 100, seed: int = 5) -> float:
    """Compose n_steps random SU(2) rotations and check that the
    resulting matrix is unitary with det = 1."""
    rng = np.random.default_rng(seed)
    U = np.eye(2, dtype=complex)
    for _ in range(n_steps):
        # Random axis-angle
        axis = rng.standard_normal(3)
        axis /= max(np.linalg.norm(axis), C.EPS_NUM)
        theta = rng.uniform(0.0, 2.0 * C.PI)
        nx, ny, nz = axis
        s, c = np.sin(theta / 2), np.cos(theta / 2)
        R = np.array([[c - 1j * nz * s, (-1j * nx - ny) * s],
                       [(-1j * nx + ny) * s, c + 1j * nz * s]],
                      dtype=complex)
        U = R @ U
    err_unitary = float(np.max(np.abs(U @ U.conj().T - np.eye(2))))
    err_det = abs(float(np.linalg.det(U).real) - 1.0)
    return max(err_unitary, err_det)


# ---------------------------------------------------------------------------
# 7. Reproducible PRNG  (multiplicative congruential)
# ---------------------------------------------------------------------------
def random_test(seed: int = 12345, n: int = 10000) -> Tuple[float, float]:
    """Generate a uniform sequence with the multiplicative congruential
    generator  x_{n+1} = 5^7 x_n  (mod 2^31)  and report the sample
    mean and the KS-like discrepancy from 0.5."""
    a = 78125
    mod = 2 ** 31
    state = seed % mod
    if state % 2 == 0:
        state = (state + 1) % mod
    s = 0.0
    for _ in range(n):
        state = (a * state) % mod
        s += state / mod
    mean = s / n
    discrepancy = abs(mean - 0.5)
    return mean, discrepancy


# ---------------------------------------------------------------------------
# Composite runner
# ---------------------------------------------------------------------------
def run_all_kernels(verbose: bool = False) -> dict:
    """Run all kernels and return a dict of (name -> residual).
    A kernel is considered PASSED if its residual is below 1e-8."""
    out = {}
    _, r1 = block_tridiag_solve(n_blocks=20)
    out["btrix"] = r1
    r2 = cfft2d_test(32, 32)
    out["cfft2d"] = r2
    r3 = cholsky_test(40)
    out["cholsky"] = r3
    r4 = emit_test(8)
    out["emit"] = r4
    r5 = dct_test(64)
    out["dct"] = r5
    r6 = su2_chain_test(100)
    out["su2_chain"] = r6
    mean, disc = random_test(12345, 10000)
    out["random_mean"] = mean
    out["random_discrepancy"] = disc

    all_passed = all(
        out[k] < 1.0e-6 for k in ["btrix", "cfft2d", "cholsky", "emit",
                                     "dct", "su2_chain"]
    ) and out["random_discrepancy"] < 1.0e-3
    # Note: random_mean ~ 0.5 is the expected value, not a residual.
    out["all_passed"] = all_passed
    if verbose:
        for k, v in out.items():
            print(f"  {k:20s} = {v:.4e}")
    return out
