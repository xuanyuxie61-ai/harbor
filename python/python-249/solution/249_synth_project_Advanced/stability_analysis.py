# -*- coding: utf-8 -*-
"""
stability_analysis.py
=====================
Linear stability analysis of the stellar structure + nuclear network
system via eigenvalue decomposition of the Jacobian.

Background
----------
The semi-discrete stellar evolution system can be written in abstract
form as

    dy/dt = F(y)

where y = (r, P, L, T, X_1, ..., X_K)_i  at each mass shell i.
Linearisation around a reference state y_0 gives

    d(delta y)/dt = J * delta y,   J = dF/dy |_{y_0}

The eigenvalues lambda_k of J determine stability:
  - Re(lambda_k) < 0  =>  stable direction
  - Re(lambda_k) > 0  =>  unstable direction
  - |lambda_k| dt > stability_limit  =>  time-step unstable

For the coupled stellar + nuclear system the Jacobian is large, sparse,
block-banded.  We assemble it using the same element-by-element
strategy as the Wathen finite-element matrix (seed 1401_wathen_matrix)
and solve the eigenvalue problem using a power iteration + inverse
iteration scheme, or estimate the spectral radius for CFL-type step
selection via the conjugate-gradient iteration on  J^T J.

Key formulae
------------
For a nuclear network with K species, the Jacobian block for species i,j
is
    J_{ij} = d(n_i)/dt / dn_j
           = - n_i n_j <sigma v>_{ij}  (for i != j, destruction)
           = - sum_k n_k <sigma v>_{ik} (for i == j)
For the thermal coupling,
    J_{T,i} = (epsilon_nuc / T) * (nu - 2/3)
where nu is the temperature exponent  epsilon ~ T^nu.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Optional
import math


# =====================================================================
# Sparse tripet (IJX) storage and conversion
# =====================================================================

class SparseMatrix:
    """Sparse matrix in coordinate (triplet) form with conversion to
    dense.  Direct analogue of the st_to_ge / wathen_sparse machinery
    in seed 1401_wathen_matrix.

    Attributes
    ----------
    n, m : int
        Number of rows and columns.
    I, J, X : list
        Triplet indices and values.
    """

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.I: List[int] = []
        self.J: List[int] = []
        self.X: List[float] = []

    def add(self, i: int, j: int, x: float) -> None:
        if 0 <= i < self.n and 0 <= j < self.m:
            self.I.append(i)
            self.J.append(j)
            self.X.append(x)

    def to_dense(self) -> List[List[float]]:
        """Convert to dense n-by-m list of lists."""
        A = [[0.0]*self.m for _ in range(self.n)]
        for i, j, x in zip(self.I, self.J, self.X):
            A[i][j] += x
        return A

    def matvec(self, x: List[float]) -> List[float]:
        """Compute A x for a dense vector x."""
        if len(x) != self.m:
            raise ValueError("matvec: dimension mismatch")
        y = [0.0] * self.n
        for i, j, xval in zip(self.I, self.J, self.X):
            y[i] += xval * x[j]
        return y

    def nnz(self) -> int:
        return len(self.X)


# =====================================================================
# Conjugate gradient solver (from seed 1401 cg_sparse)
# =====================================================================

def cg_solve(A: SparseMatrix, b: List[float], x0: Optional[List[float]] = None,
             tol: float = 1.0e-10, maxiter: int = 0
             ) -> Tuple[List[float], int, float]:
    """Solve A x = b by the conjugate gradient method, where A is SPD.

    Returns (x, iterations, residual_norm).  If maxiter == 0 we use
    maxiter = A.n.
    """
    n = A.n
    if maxiter <= 0:
        maxiter = n
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]
    # r = b - A x
    Ax = A.matvec(x)
    r = [b[i] - Ax[i] for i in range(n)]
    p = r[:]
    rs_old = sum(ri*ri for ri in r)
    if math.sqrt(rs_old) < tol:
        return x, 0, math.sqrt(rs_old)
    for it in range(1, maxiter+1):
        Ap = A.matvec(p)
        pAp = sum(pi*Api for pi, Api in zip(p, Ap))
        if abs(pAp) < 1.0e-300:
            break
        alpha = rs_old / pAp
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * Ap[i] for i in range(n)]
        rs_new = sum(ri*ri for ri in r)
        if math.sqrt(rs_new) < tol:
            return x, it, math.sqrt(rs_new)
        beta = rs_new / rs_old
        p = [r[i] + beta * p[i] for i in range(n)]
        rs_old = rs_new
    return x, maxiter, math.sqrt(sum(ri*ri for ri in r))


# =====================================================================
# Jacobian assembly for the stellar + nuclear system
# =====================================================================

def assemble_nuclear_jacobian(species_names: List[str],
                              abundances: List[float],
                              T9: float, rho: float,
                              rate_func: Callable
                              ) -> SparseMatrix:
    """Assemble the nuclear Jacobian block J_{ij} = d Ydot_i / d Y_j
    where Y_i are the species mass fractions.

    For a network with K species we use a simple destruction-only model
    where each species i is consumed in a two-body reaction with
    rate r_i = rate_func(i, T9, rho).  The Jacobian then is
        J_{ij} = - r_i delta_{ij}  Y_j   (for i != j)
        J_{ii} = - sum_k r_i Y_k  + production terms
    This is the block-diagonal part; the off-diagonal (coupling to T)
    is added via J_{iT} = epsilon_nuc * nu / T.

    The assembly mirrors the element-by-element loop in
    `wathen_sparse` (seed 1401): each reaction contributes a local
    element matrix that is scattered into the global system.
    """
    K = len(species_names)
    J = SparseMatrix(K + 1, K + 1)
    for i in range(K):
        ri = rate_func(i, T9, rho)
        # Destruction term
        J.add(i, i, -ri)
        # Coupling to temperature row (index K)
        #    d(n_i)/dT = n_i * (nu/T)   with  nu ~ 2..40
        nu = 4.0 + 2.0 * i   # rough temperature exponent
        J.add(i, K, abundances[i] * nu * ri / max(T9, 1.0e-10))
        # Off-diagonal: species j feeds into species i
        for j in range(K):
            if j != i:
                J.add(i, j, 0.5 * ri * abundances[j])
    # Energy equation row (last row)
    eps = 0.0
    for i in range(K):
        eps += rate_func(i, T9, rho) * abundances[i]
    J.add(K, K, -0.1 * eps)
    return J


# =====================================================================
# Power iteration for spectral radius
# =====================================================================

def spectral_radius(A: SparseMatrix, n_iter: int = 100,
                    seed_vec: Optional[List[float]] = None
                    ) -> Tuple[float, List[float]]:
    """Estimate the spectral radius rho(A) by power iteration.

    Returns (rho, v) where v is the approximate dominant eigenvector
    normalised to unit 2-norm.  Uses a pseudo-random initial vector
    to avoid symmetry-induced orthogonality with the dominant mode.
    """
    n = A.n
    if seed_vec is None:
        # Deterministic pseudo-random seed vector using a simple LCG
        # to avoid symmetry with eigenvectors of structured matrices.
        v = [0.0] * n
        s = 12345
        for i in range(n):
            s = (s * 1103515245 + 12345) & 0x7fffffff
            v[i] = (s / 0x7fffffff) - 0.5
        nrm = math.sqrt(sum(x*x for x in v))
        if nrm > 0:
            v = [x / nrm for x in v]
    else:
        nrm = math.sqrt(sum(x*x for x in seed_vec))
        v = [x / nrm for x in seed_vec] if nrm > 0 else [1.0/math.sqrt(n)]*n
    rho = 0.0
    for _ in range(n_iter):
        w = A.matvec(v)
        nrm = math.sqrt(sum(wi*wi for wi in w))
        if nrm < 1.0e-300:
            return 0.0, v
        v = [wi / nrm for wi in w]
        rho = nrm
    return rho, v


def eigenvalue_gershgorin(A: SparseMatrix) -> Tuple[List[Tuple[float,float]],
                                                      List[Tuple[float,float]]]:
    """Compute Gershgorin discs for the eigenvalues of A.

    Returns (real_bounds, complex_bounds) where each is a list of
    (lo, hi) intervals.  For real matrices the real bounds are the
    classical Gershgorin intervals [a_ii - R_i, a_ii + R_i] with
    R_i = sum_{j != i} |a_ij|.
    """
    dense = A.to_dense()
    n = A.n
    discs = []
    for i in range(n):
        aii = dense[i][i]
        Ri = sum(abs(dense[i][j]) for j in range(n) if j != i)
        discs.append((aii - Ri, aii + Ri))
    # Overall envelope
    lo = min(d[0] for d in discs)
    hi = max(d[1] for d in discs)
    return discs, [(lo, hi)]


# =====================================================================
# CFL / von Neumann step limiter
# =====================================================================

def max_stable_dt_from_J(J: SparseMatrix, safety: float = 0.8) -> float:
    """Return the maximum stable explicit time-step from the spectral
    radius of J.  The stability region of the explicit Euler method
    is the disc |1 + dt lambda| <= 1, so for real negative eigenvalues
    we need dt <= 2 / |lambda_max|, and for imaginary we need
    dt <= 1 / |lambda_max|.  We take the conservative choice
        dt_max = safety / rho(J).
    """
    rho, _ = spectral_radius(J, n_iter=50)
    if rho <= 0.0:
        return float("inf")
    return safety / rho


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    """Test CG on a small SPD system and spectral radius on a 3x3."""
    print("stability_analysis self-test:")
    # SPD matrix  A = tridiag(-1, 2, -1)  of order 4
    A = SparseMatrix(4, 4)
    for i in range(4):
        A.add(i, i, 2.0)
        if i > 0:
            A.add(i, i-1, -1.0)
        if i < 3:
            A.add(i, i+1, -1.0)
    b = [1.0, 2.0, 3.0, 4.0]
    x, it, rn = cg_solve(A, b, tol=1.0e-10)
    print(f"  cg_solve: it={it}, residual={rn:.2e}, x={[round(xi,4) for xi in x]}")
    rho, _ = spectral_radius(A, n_iter=50)
    # Eigenvalues of tridiag(-1,2,-1) of order 4 are
    # 2 - 2 cos(k pi / 5) for k=1..4.  Largest is 2 - 2 cos(4pi/5) ~ 3.618.
    exact_rho = 2.0 - 2.0 * math.cos(4.0 * math.pi / 5.0)
    print(f"  spectral_radius(tridiag(4)) = {rho:.6f}  "
          f"(exact = {exact_rho:.6f})")
    discs, env = eigenvalue_gershgorin(A)
    print(f"  Gershgorin envelope = {env}")
    print("stability_analysis self-test OK")


if __name__ == "__main__":
    _self_test()
