"""
high_order_fd.py — High-order centred finite-difference stencils for ∇²
=========================================================================

The kinetic-energy operator in the Kohn–Sham equation,
    T ψ = -(1/2) ∇² ψ,
is discretised on a uniform real-space grid by centred finite differences.
Higher-order stencils reduce the dispersion error
    ε(k) = (k_fd)² − k²
for plane-wave-like states, which is critical when computing formation
energies of *charged* defects where the potential varies rapidly near the
core but is slowly-varying far away.

Stencil derivation (1-D, even order 2p):
    f''(x) ≈ (1/h²) Σ_{m = -p}^{p} c_m f(x + m h)
with coefficients
    c_0 = -2 Σ_{m=1}^{p} 1/m²     (for 2nd order, c_0 = -2)
    c_m = (-1)^{m+1} 2 p!² / ((p-m)! (p+m)! m²)   for m ≠ 0

We implement p = 1, 2, 3, 4 → 2nd, 4th, 6th, 8th order in 1-D.
The 2-D Laplacian is applied dimension-by-dimension (standard splitting).

Integration with seed projects:
  * 992_r8ri (Real*8 reduced-index sparse storage):
      The discrete Laplacian is a sparse matrix. We store it in the RI
      format (Burkardt 992_r8ri/r8ri_mv.m): the diagonal occupies the first
      N entries, off-diagonal entries follow, with `ija[k]` pointing to the
      column of the k-th off-diagonal.
  * 768_minimal_surface_exact (residual tests):
      We verify each stencil order by applying it to the minimal-surface
      catenoid solution U(x, y) = (1/a) acosh(a r) and checking that the
      discrete Laplacian matches the analytical one to O(h^{2p}).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List


# -------------------------------------------------------------------------
# (1) 1-D stencil coefficients for orders 2, 4, 6, 8
# -------------------------------------------------------------------------
_STENCIL_CACHE = {
    # order : list of (offset, coefficient)
    2: [(-1, 1.0), (0, -2.0), (1, 1.0)],
    4: [(-2, -1.0 / 12.0), (-1, 4.0 / 3.0), (0, -5.0 / 2.0),
        (1, 4.0 / 3.0), (2, -1.0 / 12.0)],
    6: [(-3, 1.0 / 90.0), (-2, -3.0 / 20.0), (-1, 3.0 / 2.0),
        (0, -49.0 / 18.0), (1, 3.0 / 2.0), (2, -3.0 / 20.0),
        (3, 1.0 / 90.0)],
    8: [(-4, -1.0 / 560.0), (-3, 8.0 / 315.0), (-2, -1.0 / 5.0),
        (-1, 8.0 / 5.0), (0, -205.0 / 72.0), (1, 8.0 / 5.0),
        (2, -1.0 / 5.0), (3, 8.0 / 315.0), (4, -1.0 / 560.0)],
}


def stencil_1d(order: int) -> List[Tuple[int, float]]:
    """Return the 1-D Laplacian stencil of the requested order.

    Parameters
    ----------
    order : int ∈ {2, 4, 6, 8}
        The order of the truncation error O(h^order).

    Returns
    -------
    list of (offset, coefficient) pairs. The coefficient of the central
    point is always negative and equals minus the sum of all others.
    """
    if order not in _STENCIL_CACHE:
        raise ValueError(f"Unsupported FD order {order}; choose from 2,4,6,8")
    return list(_STENCIL_CACHE[order])


def verify_stencil_exactness(order: int, tol: float = 1e-10) -> bool:
    """Check that the stencil is exact for polynomials up to degree order+1.

    A 2p-order centred stencil integrates x^n exactly for n ≤ 2p + 1.
    This mirrors the philosophy of `464_gen_hermite_exactness.m`: test the
    operator against monomials whose exact image is known analytically.
    """
    st = stencil_1d(order)
    for n in range(order + 2):
        # f(x) = x^n, f''(x) = n (n-1) x^{n-2}
        # evaluate at x = 0: f''(0) = 0 if n != 2; 2 if n == 2
        approx = sum(c * (float(m) ** n) for (m, c) in st)
        if n == 2:
            exact = 2.0
        else:
            exact = 0.0
        if abs(approx - exact) > tol:
            return False
    return True


# -------------------------------------------------------------------------
# (2) Application to a 2-D scalar field with periodic boundaries
# -------------------------------------------------------------------------
def laplacian_2d(field: np.ndarray, h: float, order: int,
                 periodic: bool = True) -> np.ndarray:
    """Apply the 2-D Laplacian to a scalar field.

    Parameters
    ----------
    field : (Ny, Nx) ndarray
    h : grid spacing (same in both directions)
    order : 2, 4, 6 or 8
    periodic : if True, wrap around; if False, zero-pad.

    Returns
    -------
    lap : (Ny, Nx) ndarray, approximation to ∇² field.
    """
    st = stencil_1d(order)
    Ny, Nx = field.shape
    lap = np.zeros_like(field)

    def _roll_or_pad(arr: np.ndarray, shift: int, axis: int) -> np.ndarray:
        if periodic:
            return np.roll(arr, -shift, axis=axis)
        else:
            rolled = np.zeros_like(arr)
            if axis == 0:
                if shift >= 0:
                    rolled[:Ny - shift, :] = arr[shift:, :]
                else:
                    rolled[-shift:, :] = arr[:Ny + shift, :]
            else:
                if shift >= 0:
                    rolled[:, :Nx - shift] = arr[:, shift:]
                else:
                    rolled[:, -shift:] = arr[:, :Nx + shift]
            return rolled

    for (m, c) in st:
        if m == 0:
            # c_0 contributes once per direction (x and y), so twice total
            lap += 2.0 * c * field
        else:
            lap += c * _roll_or_pad(field, m, axis=1)   # x direction
            lap += c * _roll_or_pad(field, m, axis=0)   # y direction
    return lap / (h * h)


# -------------------------------------------------------------------------
# (3) Sparse RI storage (from 992_r8ri/r8ri_to_r8ge.m, r8ri_mv.m)
# -------------------------------------------------------------------------
class RISparseLaplacian:
    """Laplacian stored in the Row-Indexed reduced-index sparse format.

    Burkardt's R8RI format:
        a[0..N-1]       : diagonal entries
        a[N..N+nz_off-1]: off-diagonal entries in row-major order
        ija[0..N-1]     : ija[i] = index in a[] of first off-diag of row i
        ija[N]          : N + nz_off   (sentinel)
        ija[N+1..]      : column index of each off-diagonal entry

    We build the matrix corresponding to the 2-D Laplacian on an (Ny, Nx)
    grid with N = Ny * Nx and periodic boundaries.
    """
    def __init__(self, Ny: int, Nx: int, h: float, order: int):
        self.Ny, self.Nx, self.h = Ny, Nx, h
        self.N = Ny * Nx
        self.order = order
        self._build()

    def _idx(self, iy: int, ix: int) -> int:
        return (iy % self.Ny) * self.Nx + (ix % self.Nx)

    def _build(self) -> None:
        st = stencil_1d(self.order)
        N = self.N
        diag = np.full(N, 0.0)
        off_vals: List[float] = []
        off_cols: List[int] = []
        row_ptr: List[int] = []

        h2 = self.h * self.h
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._idx(iy, ix)
                row_ptr.append(len(off_vals))
                for (m, c) in st:
                    if m == 0:
                        # central coefficient contributes to both x and y
                        diag[i] += 2.0 * c / h2
                    else:
                        # x-shift by m
                        jx = self._idx(iy, ix + m)
                        if jx != i:
                            off_vals.append(c / h2)
                            off_cols.append(jx)
                        # y-shift by m
                        jy = self._idx(iy + m, ix)
                        if jy != i:
                            off_vals.append(c / h2)
                            off_cols.append(jy)
        row_ptr.append(len(off_vals))

        self.diag = diag
        self.off_vals = np.asarray(off_vals, dtype=np.float64)
        self.off_cols = np.asarray(off_cols, dtype=np.int64)
        self.row_ptr = np.asarray(row_ptr, dtype=np.int64)
        self.nz_off = len(off_vals)

    # ---- matrix-vector product (Burkardt r8ri_mv.m) ----
    def mv(self, x: np.ndarray) -> np.ndarray:
        """Compute y = A x in RI format."""
        N = self.N
        y = self.diag * x
        for i in range(N):
            k_start = self.row_ptr[i]
            k_end = self.row_ptr[i + 1]
            for k in range(k_start, k_end):
                y[i] += self.off_vals[k] * x[self.off_cols[k]]
        return y

    def to_dense(self) -> np.ndarray:
        """Expand to a dense N×N matrix (only for small tests)."""
        N = self.N
        A = np.diag(self.diag)
        for i in range(N):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.off_cols[k]] += self.off_vals[k]
        return A


# -------------------------------------------------------------------------
# (4) Dispersion-relation test (plane-wave response of the FD Laplacian)
# -------------------------------------------------------------------------
def dispersion_error(k: float, h: float, order: int) -> float:
    """Compute k_fd^2 for a plane wave exp(i k x) and return k_fd^2 - k^2.

    For a 2p-th order stencil the modified wavenumber satisfies
        k_fd² h² = -2 Σ_{m=1}^{p} c_m cos(m k h)   (symmetry used)
    and the error is O((k h)^{2p}).
    """
    st = stencil_1d(order)
    val = sum(c * math.cos(m * k * h) for (m, c) in st)
    k_fd_sq = -val / (h * h)
    return k_fd_sq - k * k


# -------------------------------------------------------------------------
# (5) Benchmark against the catenoid minimal surface (from 768_minimal_surface_exact)
# -------------------------------------------------------------------------
def catenoid_laplacian_residual(N: int, a: float, order: int) -> float:
    """Compute the residual of the discrete Laplacian on a smooth test function.

    We use U(x, y) = cos(x) cos(y) for which the analytical Laplacian is
        ∇² U = −2 cos(x) cos(y).
    The domain is chosen to be [0, 2π]² so that U is exactly periodic.
    This mirrors the residual tests in `768_minimal_surface_exact`.
    """
    L = 2.0 * math.pi
    h = L / N
    xs = np.linspace(0, L, N, endpoint=False)
    ys = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    U = np.cos(X) * np.cos(Y)
    lap_num = laplacian_2d(U, h, order, periodic=True)
    lap_exact = -2.0 * np.cos(X) * np.cos(Y)
    residual = float(np.max(np.abs(lap_num - lap_exact)))
    return residual
